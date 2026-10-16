import { findPreset } from "./presets.js";
import {
  BOUNDS,
  isCacheType,
  isValidOrgOrRepo,
  isValidQuant,
  isValidRepoPath,
  isValidRevision,
  parseDecimal,
  parseInteger,
  type CacheType,
} from "./validate.js";

/** A GPU card: a preset id, or "custom" with an explicit size. */
export interface GpuCard {
  readonly preset: string;
  readonly vramGiB: number;
  readonly reservedGiB: number;
}

export interface HfSelection {
  readonly org: string;
  readonly repo: string;
  readonly revision: string;
  readonly quant: string | null;
  readonly path: string | null;
}

export interface ShareState {
  readonly hf: HfSelection | null;
  readonly gpus: readonly GpuCard[];
  readonly ctx: number;
  readonly cache: CacheType;
  readonly headroomGiB: number;
}

export const DEFAULT_STATE: ShareState = Object.freeze({
  hf: null,
  gpus: Object.freeze([{ preset: "rx9070xt", vramGiB: 16, reservedGiB: 0.5 }]),
  ctx: 16384,
  cache: "f16",
  headroomGiB: 1,
});

const MAX_PARAM_LENGTH = 700;

function get(params: URLSearchParams, key: string): string | null {
  const v = params.get(key);
  return v !== null && v.length <= MAX_PARAM_LENGTH ? v : null;
}

/** g=<presetId|GB>[:<reservedGB>] */
function parseGpu(raw: string): GpuCard | null {
  if (raw.length > 40) return null;
  const [head, reservedRaw, ...extra] = raw.split(":");
  if (head === undefined || extra.length > 0) return null;
  const reservedGiB = reservedRaw === undefined ? 0 : parseDecimal(reservedRaw, 0, BOUNDS.reservedGbMax);
  if (reservedGiB === null) return null;
  const preset = findPreset(head);
  if (preset) {
    return reservedGiB < preset.vramGiB ? { preset: preset.id, vramGiB: preset.vramGiB, reservedGiB } : null;
  }
  const gb = parseDecimal(head, BOUNDS.gpuGbMin, BOUNDS.gpuGbMax);
  if (gb === null || reservedGiB >= gb) return null;
  return { preset: "custom", vramGiB: gb, reservedGiB };
}

function fmt(n: number): string {
  return String(Math.round(n * 1000) / 1000);
}

function formatGpu(g: GpuCard): string {
  const head = g.preset === "custom" ? fmt(g.vramGiB) : g.preset;
  return g.reservedGiB > 0 ? `${head}:${fmt(g.reservedGiB)}` : head;
}

/** Parse untrusted query params. Anything invalid falls back to defaults. */
export function parseShareState(params: URLSearchParams): ShareState {
  let hf: HfSelection | null = null;
  const repoRaw = get(params, "hf");
  if (repoRaw !== null) {
    const parts = repoRaw.split("/");
    if (parts.length === 2 && isValidOrgOrRepo(parts[0]) && isValidOrgOrRepo(parts[1])) {
      const rev = get(params, "rev");
      const quant = get(params, "q");
      const path = get(params, "f");
      hf = {
        org: parts[0],
        repo: parts[1],
        revision: rev !== null && isValidRevision(rev) ? rev : "main",
        quant: quant !== null && isValidQuant(quant) ? quant : null,
        path: path !== null && isValidRepoPath(path) ? path : null,
      };
    }
  }

  const gpus: GpuCard[] = [];
  for (const raw of params.getAll("g").slice(0, 64)) {
    if (gpus.length >= BOUNDS.gpuCountMax) break;
    const g = parseGpu(raw);
    if (g) gpus.push(g);
  }

  const ctx = parseInteger(get(params, "ctx"), BOUNDS.ctxMin, BOUNDS.ctxMax);
  const cache = get(params, "kv");
  const headroom = parseDecimal(get(params, "hr"), 0, BOUNDS.headroomGbMax);

  return {
    hf,
    gpus: gpus.length > 0 ? gpus : DEFAULT_STATE.gpus,
    ctx: ctx ?? DEFAULT_STATE.ctx,
    cache: isCacheType(cache) ? cache : DEFAULT_STATE.cache,
    headroomGiB: headroom ?? DEFAULT_STATE.headroomGiB,
  };
}

/** Serialize state to query params. Never includes tokens or local file names. */
export function serializeShareState(s: ShareState): URLSearchParams {
  const p = new URLSearchParams();
  if (s.hf) {
    p.set("hf", `${s.hf.org}/${s.hf.repo}`);
    if (s.hf.revision !== "main") p.set("rev", s.hf.revision);
    if (s.hf.path) p.set("f", s.hf.path);
    else if (s.hf.quant) p.set("q", s.hf.quant);
  }
  for (const g of s.gpus.slice(0, BOUNDS.gpuCountMax)) p.append("g", formatGpu(g));
  p.set("ctx", String(Math.round(s.ctx)));
  p.set("kv", s.cache);
  if (s.headroomGiB !== DEFAULT_STATE.headroomGiB) p.set("hr", fmt(s.headroomGiB));
  return p;
}
