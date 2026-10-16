import type { CacheType } from "./validate.js";

export const GiB = 1024 ** 3;

/** Bytes per cached element. q8_0 = 34 B / 32, q4_0 = 18 B / 32. */
export const CACHE_BYTES_PER_ELEM: Readonly<Record<CacheType, number>> = Object.freeze({
  f16: 2,
  q8_0: 34 / 32,
  q4_0: 18 / 32,
});

/** The subset of ModelInfo the planner needs. */
export interface PlanModel {
  readonly layerBytes: readonly number[];
  readonly outputBytes: number;
  readonly tokenEmbdBytes: number;
  readonly kvKElems: readonly number[];
  readonly kvVElems: readonly number[];
}

export interface GpuSpec {
  readonly name: string;
  readonly vramGiB: number;
  readonly reservedGiB: number;
}

export interface PlanInput {
  readonly model: PlanModel;
  readonly gpus: readonly GpuSpec[];
  readonly ctx: number;
  readonly cacheK: CacheType;
  readonly cacheV: CacheType;
  /** Compute-buffer headroom per GPU. */
  readonly headroomGiB: number;
}

export interface GpuAllocation {
  readonly index: number;
  readonly name: string;
  readonly capacity: number;
  readonly reserved: number;
  readonly headroom: number;
  readonly weights: number;
  readonly kv: number;
  /** Half-open range of repeating layers [start, end), or null. */
  readonly layers: readonly [number, number] | null;
  readonly hasOutput: boolean;
  /** capacity - reserved - headroom - weights - kv (negative = overcommitted). */
  readonly free: number;
}

export interface Plan {
  readonly fits: boolean;
  /** Value for -ngl: repeating layers on GPU, +1 when the output layer is too. */
  readonly ngl: number;
  readonly nLayers: number;
  readonly gpus: readonly GpuAllocation[];
  /** Weights that would normally be on GPU but don't fit (layers + output). */
  readonly cpuSpillBytes: number;
  /** Everything that lives in system RAM: token_embd + spilled weights. */
  readonly cpuWeightBytes: number;
  readonly cpuKvBytes: number;
  readonly totalKvBytes: number;
  /** Per-GPU layer counts for --tensor-split (output counted on its GPU). */
  readonly tensorSplit: readonly number[];
  /** Quantized V cache needs flash attention. */
  readonly flashAttention: boolean;
}

/**
 * KV cache bytes per layer:
 *   ctx * (kElems * bytes(K) + vElems * bytes(V))
 * which for kElems = vElems = n_kv_heads * head_dim and K = V types is the
 * familiar 2 * n_kv_heads * head_dim * ctx * bytes_per_elem.
 */
export function kvBytesPerLayer(model: PlanModel, ctx: number, cacheK: CacheType, cacheV: CacheType): number[] {
  const bk = CACHE_BYTES_PER_ELEM[cacheK];
  const bv = CACHE_BYTES_PER_ELEM[cacheV];
  return model.layerBytes.map((_, i) => ctx * ((model.kvKElems[i] ?? 0) * bk + (model.kvVElems[i] ?? 0) * bv));
}

export function totalKvBytes(model: PlanModel, ctx: number, cacheK: CacheType, cacheV: CacheType): number {
  return kvBytesPerLayer(model, ctx, cacheK, cacheV).reduce((a, b) => a + b, 0);
}

export function needsFlashAttention(cacheV: CacheType): boolean {
  return cacheV !== "f16";
}

interface Assignment {
  /** Number of consecutive offloaded layers given to each GPU, in order. */
  counts: number[];
  outputGpu: number | null;
}

function usableBytes(g: GpuSpec, headroomGiB: number): number {
  return Math.max(0, (g.vramGiB - g.reservedGiB - headroomGiB) * GiB);
}

/** Split `total` into integer parts proportional to weights (largest remainder). */
export function proportionalCounts(total: number, weights: readonly number[]): number[] {
  const sum = weights.reduce((a, b) => a + b, 0);
  if (sum <= 0 || total <= 0) return weights.map(() => 0);
  const raw = weights.map((w) => (w / sum) * total);
  const counts = raw.map(Math.floor);
  let left = total - counts.reduce((a, b) => a + b, 0);
  const order = raw.map((r, i) => [r - Math.floor(r), i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  for (const [, i] of order) {
    if (left <= 0) break;
    counts[i] = (counts[i] ?? 0) + 1;
    left--;
  }
  return counts;
}

function fitsAssignment(costs: readonly number[], usable: readonly number[], a: Assignment, outputBytes: number): boolean {
  let j = 0;
  for (let g = 0; g < usable.length; g++) {
    let used = a.outputGpu === g ? outputBytes : 0;
    const n = a.counts[g] ?? 0;
    for (let k = 0; k < n; k++) used += costs[j++] ?? 0;
    if (used > (usable[g] ?? 0)) return false;
  }
  return true;
}

/** Try VRAM-weighted first, then a greedy in-order fill (optimal for contiguous splits). */
function assign(costs: readonly number[], usable: readonly number[], outputBytes: number | null): Assignment | null {
  let outputGpu: number | null = null;
  if (outputBytes !== null) {
    for (let g = usable.length - 1; g >= 0; g--) {
      if ((usable[g] ?? 0) > 0) {
        outputGpu = g;
        break;
      }
    }
    if (outputGpu === null) return null;
  }
  const out = outputBytes ?? 0;

  const proportional: Assignment = { counts: proportionalCounts(costs.length, usable), outputGpu };
  if (fitsAssignment(costs, usable, proportional, out)) return proportional;

  const counts = usable.map(() => 0);
  let j = 0;
  for (let g = 0; g < usable.length && j < costs.length; g++) {
    let used = outputGpu === g ? out : 0;
    if (used > (usable[g] ?? 0)) return null;
    while (j < costs.length && used + (costs[j] ?? 0) <= (usable[g] ?? 0)) {
      used += costs[j] ?? 0;
      counts[g] = (counts[g] ?? 0) + 1;
      j++;
    }
  }
  if (j < costs.length) return null;
  const greedy = { counts, outputGpu };
  return fitsAssignment(costs, usable, greedy, out) ? greedy : null;
}

export function plan(input: PlanInput): Plan {
  const { model, gpus, ctx, cacheK, cacheV, headroomGiB } = input;
  const L = model.layerBytes.length;
  const kv = kvBytesPerLayer(model, ctx, cacheK, cacheV);
  const costs = model.layerBytes.map((w, i) => w + (kv[i] ?? 0));
  const usable = gpus.map((g) => usableBytes(g, headroomGiB));
  const totalKv = kv.reduce((a, b) => a + b, 0);

  // llama.cpp offloads the *last* ngl repeating layers; ngl = L + 1 also offloads the output layer.
  let chosen: { ngl: number; a: Assignment } | null = null;
  for (let ngl = L + 1; ngl >= 0 && gpus.length > 0; ngl--) {
    const nRepeat = Math.min(ngl, L);
    const a = assign(costs.slice(L - nRepeat), usable, ngl > L ? model.outputBytes : null);
    if (a) {
      chosen = { ngl, a };
      break;
    }
  }
  if (!chosen) chosen = { ngl: 0, a: { counts: gpus.map(() => 0), outputGpu: null } };

  const { ngl, a } = chosen;
  const nRepeat = Math.min(ngl, L);
  const start = L - nRepeat;
  let cursor = start;
  const allocations: GpuAllocation[] = gpus.map((g, i) => {
    const n = a.counts[i] ?? 0;
    const range: [number, number] | null = n > 0 ? [cursor, cursor + n] : null;
    let weights = a.outputGpu === i ? model.outputBytes : 0;
    let kvBytes = 0;
    for (let l = cursor; l < cursor + n; l++) {
      weights += model.layerBytes[l] ?? 0;
      kvBytes += kv[l] ?? 0;
    }
    cursor += n;
    const capacity = g.vramGiB * GiB;
    const reserved = g.reservedGiB * GiB;
    const headroom = headroomGiB * GiB;
    return {
      index: i,
      name: g.name,
      capacity,
      reserved,
      headroom,
      weights,
      kv: kvBytes,
      layers: range,
      hasOutput: a.outputGpu === i,
      free: capacity - reserved - headroom - weights - kvBytes,
    };
  });

  let spill = 0;
  let cpuKv = 0;
  for (let l = 0; l < start; l++) {
    spill += model.layerBytes[l] ?? 0;
    cpuKv += kv[l] ?? 0;
  }
  if (ngl <= L) spill += model.outputBytes;

  const tensorSplit = gpus.map((_, i) => (a.counts[i] ?? 0) + (a.outputGpu === i ? 1 : 0));

  return {
    fits: ngl === L + 1,
    ngl,
    nLayers: L,
    gpus: allocations,
    cpuSpillBytes: spill,
    cpuWeightBytes: spill + model.tokenEmbdBytes,
    cpuKvBytes: cpuKv,
    totalKvBytes: totalKv,
    tensorSplit,
    flashAttention: needsFlashAttention(cacheV),
  };
}

export const CTX_MIN = 512;
export const CTX_MAX = 1_048_576;
const CTX_STEP = 256;

/**
 * Largest context (multiple of 256, within [512, upper]) at which the whole
 * model plus KV cache fits on the GPUs. Null if even 512 doesn't fit.
 */
export function maxContext(input: Omit<PlanInput, "ctx">, upper = CTX_MAX): number | null {
  const fits = (ctx: number) => plan({ ...input, ctx }).fits;
  const hi0 = Math.max(CTX_MIN, Math.min(CTX_MAX, Math.floor(upper / CTX_STEP) * CTX_STEP));
  if (!fits(CTX_MIN)) return null;
  if (fits(hi0)) return hi0;
  let lo = CTX_MIN / CTX_STEP;
  let hi = hi0 / CTX_STEP;
  // Invariant: lo fits, hi does not.
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (fits(mid * CTX_STEP)) lo = mid;
    else hi = mid;
  }
  return lo * CTX_STEP;
}
