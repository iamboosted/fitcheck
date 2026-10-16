import { ParseError } from "./errors.js";
import { fileTypeName, ggmlType } from "./ggml-types.js";
import type { GgufArray, GgufValue, TensorInfo } from "./gguf.js";
import { LIMITS } from "./limits.js";
import type { GgufFile } from "./reader.js";

export interface ModelInfo {
  readonly architecture: string;
  readonly name: string | null;
  readonly sizeLabel: string | null;
  /** Quant label from general.file_type, else the dominant tensor type. */
  readonly quant: string | null;
  readonly paramCount: number;
  readonly totalBytes: number;
  readonly nLayers: number;
  /** Weight bytes of each repeating block `blk.N.*`. */
  readonly layerBytes: readonly number[];
  readonly tokenEmbdBytes: number;
  /** Output head + output_norm. With tied embeddings, a copy of token_embd. */
  readonly outputBytes: number;
  readonly tiedOutput: boolean;
  /** Non-layer tensors other than token_embd / output (always tiny). */
  readonly otherBytes: number;
  /** KV cache elements per token per layer, for K and V. */
  readonly kvKElems: readonly number[];
  readonly kvVElems: readonly number[];
  readonly contextLength: number | null;
  readonly embeddingLength: number | null;
  readonly headCount: number | null;
  readonly headCountKv: number | null;
  readonly headDim: number | null;
  readonly expertCount: number | null;
  readonly expertUsedCount: number | null;
  readonly shardCount: number;
  readonly warnings: readonly string[];
  /** Summary of hidden tokenizer/chat-template metadata (sizes only). */
  readonly hiddenMetadata: { readonly tokens: number | null; readonly chatTemplateBytes: number | null };
}

const ARCH_RE = /^[a-z0-9][a-z0-9_.-]{0,63}$/i;
const LAYER_RE = /^blk\.(\d{1,5})\./;
const MAX_LAYERS = 4096;

function num(v: GgufValue | undefined): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function posInt(v: GgufValue | undefined, max = 1e9): number | null {
  const n = num(v);
  return n !== null && Number.isInteger(n) && n >= 0 && n <= max ? n : null;
}

function str(v: GgufValue | undefined): string | null {
  return typeof v === "string" ? v : null;
}

function isArray(v: GgufValue | undefined): v is GgufArray {
  return typeof v === "object" && v !== null && (v as GgufArray).kind === "array";
}

/** Value that may be a scalar or a per-layer array (e.g. head_count_kv). */
function perLayer(v: GgufValue | undefined, nLayers: number, fallback: number): number[] {
  if (isArray(v) && v.values && v.values.length === nLayers) {
    return v.values.map((x) => posInt(x, 1e6) ?? fallback);
  }
  const s = posInt(v, 1e6);
  return new Array<number>(nLayers).fill(s ?? fallback);
}

/** Order shards by split.no and verify the set is complete and consistent. */
function orderShards(files: readonly GgufFile[]): GgufFile[] {
  if (files.length === 0) throw new ParseError("INVALID", "No GGUF files.");
  if (files.length > LIMITS.maxShards) throw new ParseError("LIMIT", "Too many shard files.");
  if (files.length === 1) {
    const count = posInt(files[0]?.header.metadata.get("split.count"));
    if (count !== null && count > 1) {
      throw new ParseError("INVALID", `This is one shard of a ${count}-part model; add every shard.`);
    }
    return [...files];
  }
  const ordered = new Array<GgufFile | undefined>(files.length);
  for (const f of files) {
    const count = posInt(f.header.metadata.get("split.count"));
    const no = posInt(f.header.metadata.get("split.no"));
    if (count !== files.length || no === null || no >= files.length) {
      throw new ParseError("INVALID", "Shard files are missing or belong to different models.");
    }
    if (ordered[no]) throw new ParseError("INVALID", "Duplicate shard.");
    ordered[no] = f;
  }
  return ordered as GgufFile[];
}

export function summarizeModel(files: readonly GgufFile[]): ModelInfo {
  const shards = orderShards(files);
  const first = shards[0] as GgufFile;
  const md = first.header.metadata;
  const warnings: string[] = [];
  for (const s of shards) for (const w of s.header.warnings) if (!warnings.includes(w)) warnings.push(w);

  const archRaw = str(md.get("general.architecture"));
  const architecture = archRaw !== null && ARCH_RE.test(archRaw) ? archRaw : "unknown";
  if (architecture === "unknown") warnings.push("Missing or invalid general.architecture.");
  const a = (k: string) => md.get(`${architecture}.${k}`);

  const tensors: TensorInfo[] = [];
  for (const s of shards) tensors.push(...s.header.tensors);
  if (tensors.length > LIMITS.maxTensorCount) throw new ParseError("LIMIT", "Tensor count exceeds the limit.");

  let maxLayerSeen = -1;
  for (const t of tensors) {
    const m = LAYER_RE.exec(t.name);
    if (m) maxLayerSeen = Math.max(maxLayerSeen, Number(m[1]));
  }
  const blockCount = posInt(a("block_count"), MAX_LAYERS);
  let nLayers = blockCount ?? maxLayerSeen + 1;
  if (blockCount === null) warnings.push("block_count missing; inferred from tensor names.");
  if (maxLayerSeen >= nLayers && maxLayerSeen < MAX_LAYERS) {
    nLayers = maxLayerSeen + 1;
    warnings.push("Tensor names reference more layers than block_count.");
  }
  nLayers = Math.max(0, Math.min(nLayers, MAX_LAYERS));

  const layerBytes = new Array<number>(nLayers).fill(0);
  let tokenEmbdBytes = 0;
  let outputHeadBytes = 0;
  let outputNormBytes = 0;
  let otherBytes = 0;
  let totalBytes = 0;
  let paramCount = 0;
  const bytesByType = new Map<number, number>();

  for (const t of tensors) {
    const bytes = t.bytes ?? 0;
    totalBytes += bytes;
    paramCount += t.elements;
    bytesByType.set(t.type, (bytesByType.get(t.type) ?? 0) + bytes);
    const m = LAYER_RE.exec(t.name);
    if (m && Number(m[1]) < nLayers) {
      const i = Number(m[1]);
      layerBytes[i] = (layerBytes[i] ?? 0) + bytes;
    } else if (t.name === "token_embd.weight") {
      tokenEmbdBytes += bytes;
    } else if (t.name === "output.weight" || t.name === "output.bias") {
      outputHeadBytes += bytes;
    } else if (t.name === "output_norm.weight" || t.name === "output_norm.bias") {
      outputNormBytes += bytes;
    } else {
      otherBytes += bytes;
    }
  }
  if (totalBytes > Number(LIMITS.maxModelBytes)) throw new ParseError("LIMIT", "Model exceeds the 2 TB limit.");

  const tiedOutput = outputHeadBytes === 0 && tokenEmbdBytes > 0;
  const outputBytes = (tiedOutput ? tokenEmbdBytes : outputHeadBytes) + outputNormBytes;

  // Attention geometry.
  const embeddingLength = posInt(a("embedding_length"), 1e7);
  const headCount = posInt(a("attention.head_count"), 1e5) ?? firstOf(a("attention.head_count"));
  const kvHeads = perLayer(a("attention.head_count_kv"), nLayers, headCount ?? 0);
  const derivedHeadDim = embeddingLength !== null && headCount ? Math.floor(embeddingLength / headCount) : null;
  const keyLength = posInt(a("attention.key_length"), 1e5) ?? derivedHeadDim;
  const valueLength = posInt(a("attention.value_length"), 1e5) ?? derivedHeadDim;
  if (keyLength === null || valueLength === null) {
    warnings.push("Could not determine attention head size; KV cache estimate is zero.");
  }

  let kvKElems: number[];
  let kvVElems: number[];
  const kvLoraRank = posInt(a("attention.kv_lora_rank"), 1e6);
  if (kvLoraRank) {
    // Multi-head latent attention (DeepSeek-style): one compressed head per layer.
    const ropeDim = posInt(a("rope.dimension_count"), 1e5) ?? 0;
    kvKElems = new Array<number>(nLayers).fill(kvLoraRank + ropeDim);
    kvVElems = new Array<number>(nLayers).fill(kvLoraRank);
    warnings.push("MLA attention detected; KV cache size is approximate.");
  } else {
    kvKElems = kvHeads.map((h) => h * (keyLength ?? 0));
    kvVElems = kvHeads.map((h) => h * (valueLength ?? 0));
  }

  const ftype = posInt(md.get("general.file_type"), 1e4);
  let quant = ftype !== null ? (fileTypeName(ftype) ?? null) : null;
  if (quant === null) {
    let best: [number, number] | null = null;
    for (const [type, bytes] of bytesByType) if (!best || bytes > best[1]) best = [type, bytes];
    quant = best ? (ggmlType(best[0])?.name ?? null) : null;
  }

  const tokens = md.get("tokenizer.ggml.tokens");
  const tmpl = md.get("tokenizer.chat_template");

  const commonKv = mostCommon(kvHeads);
  return {
    architecture,
    name: str(md.get("general.name")),
    sizeLabel: str(md.get("general.size_label")),
    quant,
    paramCount,
    totalBytes,
    nLayers,
    layerBytes,
    tokenEmbdBytes,
    outputBytes,
    tiedOutput,
    otherBytes,
    kvKElems,
    kvVElems,
    contextLength: posInt(a("context_length"), 1e9),
    embeddingLength,
    headCount: headCount ?? null,
    headCountKv: commonKv,
    headDim: keyLength,
    expertCount: posInt(a("expert_count"), 1e6),
    expertUsedCount: posInt(a("expert_used_count"), 1e6),
    shardCount: shards.length,
    warnings,
    hiddenMetadata: {
      tokens: isArray(tokens) ? tokens.length : null,
      chatTemplateBytes: typeof tmpl === "string" ? new TextEncoder().encode(tmpl).length : null,
    },
  };
}

function firstOf(v: GgufValue | undefined): number | null {
  return isArray(v) && v.values ? posInt(v.values[0], 1e5) : null;
}

function mostCommon(xs: readonly number[]): number | null {
  const counts = new Map<number, number>();
  for (const x of xs) counts.set(x, (counts.get(x) ?? 0) + 1);
  let best: number | null = null;
  let bestCount = 0;
  for (const [x, c] of counts) {
    if (x > 0 && c > bestCount) {
      best = x;
      bestCount = c;
    }
  }
  return best;
}
