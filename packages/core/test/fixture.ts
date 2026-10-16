/** Synthetic GGUF writer for tests. Produces header bytes (and optional zero data). */
import { GgufValueType as T, ggmlType } from "../src/index.js";

export class Writer {
  private chunks: Uint8Array[] = [];
  private len = 0;

  push(b: Uint8Array): this {
    this.chunks.push(b);
    this.len += b.length;
    return this;
  }
  get length(): number {
    return this.len;
  }
  u8(v: number): this {
    return this.push(Uint8Array.of(v & 0xff));
  }
  u32(v: number): this {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setUint32(0, v >>> 0, true);
    return this.push(b);
  }
  i32(v: number): this {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setInt32(0, v, true);
    return this.push(b);
  }
  f32(v: number): this {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setFloat32(0, v, true);
    return this.push(b);
  }
  u64(v: number | bigint): this {
    const b = new Uint8Array(8);
    new DataView(b.buffer).setBigUint64(0, BigInt(v), true);
    return this.push(b);
  }
  str(s: string): this {
    const bytes = new TextEncoder().encode(s);
    return this.u64(bytes.length).push(bytes);
  }
  bytes(): Uint8Array<ArrayBuffer> {
    const out = new Uint8Array(this.len);
    let p = 0;
    for (const c of this.chunks) {
      out.set(c, p);
      p += c.length;
    }
    return out;
  }
}

export type KvSpec =
  | { key: string; type: "u32"; value: number }
  | { key: string; type: "i32"; value: number }
  | { key: string; type: "f32"; value: number }
  | { key: string; type: "u64"; value: number | bigint }
  | { key: string; type: "bool"; value: boolean }
  | { key: string; type: "str"; value: string }
  | { key: string; type: "arr_i32"; value: number[] }
  | { key: string; type: "arr_str"; value: string[] };

export interface TensorSpec {
  name: string;
  dims: number[];
  type: number;
}

export function writeKv(w: Writer, kv: KvSpec): void {
  w.str(kv.key);
  switch (kv.type) {
    case "u32":
      w.u32(T.UINT32).u32(kv.value);
      break;
    case "i32":
      w.u32(T.INT32).i32(kv.value);
      break;
    case "f32":
      w.u32(T.FLOAT32).f32(kv.value);
      break;
    case "u64":
      w.u32(T.UINT64).u64(kv.value);
      break;
    case "bool":
      w.u32(T.BOOL).u8(kv.value ? 1 : 0);
      break;
    case "str":
      w.u32(T.STRING).str(kv.value);
      break;
    case "arr_i32":
      w.u32(T.ARRAY).u32(T.INT32).u64(kv.value.length);
      for (const v of kv.value) w.i32(v);
      break;
    case "arr_str":
      w.u32(T.ARRAY).u32(T.STRING).u64(kv.value.length);
      for (const v of kv.value) w.str(v);
      break;
  }
}

export function tensorBytes(t: TensorSpec): number {
  const info = ggmlType(t.type);
  if (!info) return 0;
  const ne0 = t.dims[0] ?? 1;
  const rows = t.dims.slice(1).reduce((a, b) => a * b, 1);
  return rows * Math.ceil(ne0 / info.blockSize) * info.typeSize;
}

export interface BuildOptions {
  version?: number;
  kv: KvSpec[];
  tensors: TensorSpec[];
  alignment?: number;
  /** Append this many zero bytes after the header (simulated tensor data). */
  dataBytes?: number;
}

export function buildGguf(opts: BuildOptions): Uint8Array<ArrayBuffer> {
  const w = new Writer();
  const alignment = opts.alignment ?? 32;
  w.u32(0x46554747).u32(opts.version ?? 3).u64(opts.tensors.length).u64(opts.kv.length);
  for (const kv of opts.kv) writeKv(w, kv);
  let offset = 0;
  for (const t of opts.tensors) {
    w.str(t.name).u32(t.dims.length);
    for (const d of t.dims) w.u64(d);
    w.u32(t.type).u64(offset);
    offset += Math.ceil(Math.max(tensorBytes(t), 1) / alignment) * alignment;
  }
  if (opts.dataBytes) {
    const pad = (alignment - (w.length % alignment)) % alignment;
    w.push(new Uint8Array(pad + opts.dataBytes));
  }
  return w.bytes();
}

export interface LlamaFixtureOptions {
  layers?: number;
  embd?: number;
  heads?: number;
  kvHeads?: number;
  ff?: number;
  vocab?: number;
  weightType?: number;
  tied?: boolean;
  ctx?: number;
  extraKv?: KvSpec[];
  tokenList?: number;
  dataBytes?: number;
}

/** A plausible llama-architecture model header. */
export function llamaFixture(o: LlamaFixtureOptions = {}): { bytes: Uint8Array<ArrayBuffer>; tensors: TensorSpec[] } {
  const layers = o.layers ?? 4;
  const embd = o.embd ?? 256;
  const heads = o.heads ?? 8;
  const kvHeads = o.kvHeads ?? 2;
  const ff = o.ff ?? 512;
  const vocab = o.vocab ?? 1024;
  const wt = o.weightType ?? 12; // Q4_K
  const headDim = embd / heads;
  const tensors: TensorSpec[] = [{ name: "token_embd.weight", dims: [embd, vocab], type: wt }];
  for (let i = 0; i < layers; i++) {
    tensors.push(
      { name: `blk.${i}.attn_norm.weight`, dims: [embd], type: 0 },
      { name: `blk.${i}.attn_q.weight`, dims: [embd, embd], type: wt },
      { name: `blk.${i}.attn_k.weight`, dims: [embd, kvHeads * headDim], type: wt },
      { name: `blk.${i}.attn_v.weight`, dims: [embd, kvHeads * headDim], type: wt },
      { name: `blk.${i}.attn_output.weight`, dims: [embd, embd], type: wt },
      { name: `blk.${i}.ffn_norm.weight`, dims: [embd], type: 0 },
      { name: `blk.${i}.ffn_gate.weight`, dims: [embd, ff], type: wt },
      { name: `blk.${i}.ffn_up.weight`, dims: [embd, ff], type: wt },
      { name: `blk.${i}.ffn_down.weight`, dims: [ff, embd], type: 14 },
    );
  }
  tensors.push({ name: "output_norm.weight", dims: [embd], type: 0 });
  if (!o.tied) tensors.push({ name: "output.weight", dims: [embd, vocab], type: 14 });

  const kv: KvSpec[] = [
    { key: "general.architecture", type: "str", value: "llama" },
    { key: "general.name", type: "str", value: "Test Model" },
    { key: "general.file_type", type: "u32", value: 15 },
    { key: "llama.block_count", type: "u32", value: layers },
    { key: "llama.context_length", type: "u32", value: o.ctx ?? 8192 },
    { key: "llama.embedding_length", type: "u32", value: embd },
    { key: "llama.feed_forward_length", type: "u32", value: ff },
    { key: "llama.attention.head_count", type: "u32", value: heads },
    { key: "llama.attention.head_count_kv", type: "u32", value: kvHeads },
    { key: "tokenizer.ggml.model", type: "str", value: "gpt2" },
    {
      key: "tokenizer.ggml.tokens",
      type: "arr_str",
      value: Array.from({ length: o.tokenList ?? 100 }, (_, i) => `tok${i}`),
    },
    { key: "tokenizer.chat_template", type: "str", value: "{% for m in messages %}{{ m.content }}{% endfor %}" },
    ...(o.extraKv ?? []),
  ];
  return { bytes: buildGguf({ kv, tensors, dataBytes: o.dataBytes ?? 0 }), tensors };
}
