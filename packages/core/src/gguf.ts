import { ParseError } from "./errors.js";
import { ggmlType } from "./ggml-types.js";
import { LIMITS } from "./limits.js";

/** GGUF metadata value types (gguf.h). */
export const GgufValueType = Object.freeze({
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
});

export interface GgufArray {
  readonly kind: "array";
  readonly elemType: number;
  readonly length: number;
  /** Present only for short arrays; long arrays keep just the summary. */
  readonly values?: readonly GgufValue[];
}

/** 64-bit values outside the safe integer range stay as bigint. */
export type GgufValue = number | bigint | boolean | string | GgufArray;

export interface TensorInfo {
  readonly name: string;
  readonly dims: readonly number[];
  readonly type: number;
  /** Offset relative to the start of the tensor data section. */
  readonly offset: number;
  readonly elements: number;
  /** Computed size in bytes; null when the ggml type is unknown. */
  readonly bytes: number | null;
}

export interface GgufHeader {
  readonly version: number;
  readonly metadata: ReadonlyMap<string, GgufValue>;
  readonly tensors: readonly TensorInfo[];
  readonly alignment: number;
  /** Absolute file offset of the tensor data section. */
  readonly dataOffset: number;
  /** Bytes consumed by magic + metadata + tensor infos. */
  readonly headerBytes: number;
  readonly warnings: readonly string[];
}

const MAGIC = 0x46554747; // "GGUF" little-endian
const SUPPORTED_VERSIONS = new Set([2, 3]);
const FORBIDDEN_KEYS = new Set(["__proto__", "constructor", "prototype"]);
const MAX_SAFE = BigInt(Number.MAX_SAFE_INTEGER);

/** Thrown internally when the buffer ends before the header does. */
export class NeedMoreData {
  constructor(readonly required: number) {}
}

const utf8 = new TextDecoder("utf-8", { fatal: false, ignoreBOM: true });

class Cursor {
  pos = 0;
  private readonly view: DataView;

  constructor(
    private readonly buf: Uint8Array,
    /** True when buf holds the whole file (no more bytes exist). */
    private readonly complete: boolean,
    private readonly maxHeader: number,
  ) {
    this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  }

  private need(n: number): void {
    if (!Number.isSafeInteger(n) || n < 0) {
      throw new ParseError("INVALID", "Invalid read length.");
    }
    const end = this.pos + n;
    if (end > this.maxHeader) {
      throw new ParseError("LIMIT", "GGUF header exceeds the 64 MB limit.");
    }
    if (end > this.buf.length) {
      if (this.complete) throw new ParseError("TRUNCATED", "File ends before the GGUF header is complete.");
      throw new NeedMoreData(end);
    }
  }

  u8(): number {
    this.need(1);
    return this.view.getUint8(this.pos++);
  }
  i8(): number {
    this.need(1);
    return this.view.getInt8(this.pos++);
  }
  u16(): number {
    this.need(2);
    const v = this.view.getUint16(this.pos, true);
    this.pos += 2;
    return v;
  }
  i16(): number {
    this.need(2);
    const v = this.view.getInt16(this.pos, true);
    this.pos += 2;
    return v;
  }
  u32(): number {
    this.need(4);
    const v = this.view.getUint32(this.pos, true);
    this.pos += 4;
    return v;
  }
  i32(): number {
    this.need(4);
    const v = this.view.getInt32(this.pos, true);
    this.pos += 4;
    return v;
  }
  f32(): number {
    this.need(4);
    const v = this.view.getFloat32(this.pos, true);
    this.pos += 4;
    return v;
  }
  f64(): number {
    this.need(8);
    const v = this.view.getFloat64(this.pos, true);
    this.pos += 8;
    return v;
  }
  u64(): bigint {
    this.need(8);
    const v = this.view.getBigUint64(this.pos, true);
    this.pos += 8;
    return v;
  }
  i64(): bigint {
    this.need(8);
    const v = this.view.getBigInt64(this.pos, true);
    this.pos += 8;
    return v;
  }

  /** uint64 that must fit in a JS number. */
  safeU64(what: string): number {
    const v = this.u64();
    if (v > MAX_SAFE) throw new ParseError("LIMIT", `${what} exceeds the safe integer range.`);
    return Number(v);
  }

  skip(n: number): void {
    this.need(n);
    this.pos += n;
  }

  string(maxBytes: number): string {
    const len = this.stringLength(maxBytes);
    this.need(len);
    const s = utf8.decode(this.buf.subarray(this.pos, this.pos + len));
    this.pos += len;
    return s;
  }

  skipString(maxBytes: number): void {
    this.skip(this.stringLength(maxBytes));
  }

  private stringLength(maxBytes: number): number {
    const len = this.u64();
    if (len > BigInt(maxBytes)) throw new ParseError("LIMIT", "String length exceeds the limit.");
    return Number(len);
  }
}

function fixedSize(type: number): number | null {
  switch (type) {
    case GgufValueType.UINT8:
    case GgufValueType.INT8:
    case GgufValueType.BOOL:
      return 1;
    case GgufValueType.UINT16:
    case GgufValueType.INT16:
      return 2;
    case GgufValueType.UINT32:
    case GgufValueType.INT32:
    case GgufValueType.FLOAT32:
      return 4;
    case GgufValueType.UINT64:
    case GgufValueType.INT64:
    case GgufValueType.FLOAT64:
      return 8;
    default:
      return null;
  }
}

function toNumberIfSafe(v: bigint): number | bigint {
  return v <= MAX_SAFE && v >= -MAX_SAFE ? Number(v) : v;
}

function readScalar(c: Cursor, type: number): GgufValue {
  switch (type) {
    case GgufValueType.UINT8:
      return c.u8();
    case GgufValueType.INT8:
      return c.i8();
    case GgufValueType.UINT16:
      return c.u16();
    case GgufValueType.INT16:
      return c.i16();
    case GgufValueType.UINT32:
      return c.u32();
    case GgufValueType.INT32:
      return c.i32();
    case GgufValueType.FLOAT32:
      return c.f32();
    case GgufValueType.BOOL:
      return c.u8() !== 0;
    case GgufValueType.STRING:
      return c.string(LIMITS.maxStringBytes);
    case GgufValueType.UINT64:
      return toNumberIfSafe(c.u64());
    case GgufValueType.INT64:
      return toNumberIfSafe(c.i64());
    case GgufValueType.FLOAT64:
      return c.f64();
    default:
      throw new ParseError("INVALID", "Unknown metadata value type.");
  }
}

function readValue(c: Cursor, type: number, depth: number): GgufValue {
  if (type !== GgufValueType.ARRAY) return readScalar(c, type);

  if (depth + 1 > LIMITS.maxArrayDepth) {
    throw new ParseError("LIMIT", "Metadata arrays are nested too deeply.");
  }
  const elemType = c.u32();
  const lenBig = c.u64();
  if (lenBig > BigInt(LIMITS.maxArrayLength)) {
    throw new ParseError("LIMIT", "Metadata array length exceeds the limit.");
  }
  const length = Number(lenBig);

  const size = fixedSize(elemType);
  if (size !== null) {
    if (length > LIMITS.keepNumericArrayMax) {
      c.skip(length * size);
      return { kind: "array", elemType, length };
    }
    const values: GgufValue[] = [];
    for (let i = 0; i < length; i++) values.push(readScalar(c, elemType));
    return { kind: "array", elemType, length, values };
  }

  if (elemType === GgufValueType.STRING) {
    if (length > LIMITS.keepStringArrayMax) {
      for (let i = 0; i < length; i++) c.skipString(LIMITS.maxStringBytes);
      return { kind: "array", elemType, length };
    }
    const values: GgufValue[] = [];
    for (let i = 0; i < length; i++) values.push(c.string(LIMITS.maxStringBytes));
    return { kind: "array", elemType, length, values };
  }

  if (elemType === GgufValueType.ARRAY) {
    // Nested arrays: parse to validate and advance, keep only a summary.
    for (let i = 0; i < length; i++) readValue(c, GgufValueType.ARRAY, depth + 1);
    return { kind: "array", elemType, length };
  }

  throw new ParseError("INVALID", "Unknown metadata array element type.");
}

/** Multiply non-negative bigints, failing once the product passes a cap. */
function mulCapped(a: bigint, b: bigint, cap: bigint): bigint {
  const p = a * b;
  if (p > cap) throw new ParseError("LIMIT", "Tensor size exceeds the 2 TB model limit.");
  return p;
}

export interface ParseOptions {
  /** True when `buf` contains every byte of the file. */
  complete: boolean;
  /** Total file size if known, used to estimate unknown tensor types. */
  fileSize?: number | null;
  maxHeaderBytes?: number;
}

/**
 * Parse a GGUF header from a prefix of the file. Throws `NeedMoreData` when the
 * prefix is too short (and more bytes exist), or `ParseError` on bad input.
 */
export function parseGgufHeader(buf: Uint8Array, opts: ParseOptions): GgufHeader {
  const maxHeader = Math.min(opts.maxHeaderBytes ?? LIMITS.maxHeaderBytes, LIMITS.maxHeaderBytes);
  const c = new Cursor(buf, opts.complete, maxHeader);
  const warnings: string[] = [];

  if (c.u32() !== MAGIC) throw new ParseError("BAD_MAGIC", "Not a GGUF file (bad magic).");
  const version = c.u32();
  if (!SUPPORTED_VERSIONS.has(version)) {
    throw new ParseError("BAD_VERSION", "Unsupported GGUF version.");
  }

  const tensorCountBig = c.u64();
  if (tensorCountBig > BigInt(LIMITS.maxTensorCount)) {
    throw new ParseError("LIMIT", "Tensor count exceeds the limit.");
  }
  const kvCountBig = c.u64();
  if (kvCountBig > BigInt(LIMITS.maxKvCount)) {
    throw new ParseError("LIMIT", "Metadata count exceeds the limit.");
  }
  const tensorCount = Number(tensorCountBig);
  const kvCount = Number(kvCountBig);

  const metadata = new Map<string, GgufValue>();
  for (let i = 0; i < kvCount; i++) {
    const key = c.string(LIMITS.maxKeyBytes);
    if (FORBIDDEN_KEYS.has(key)) throw new ParseError("INVALID", "Forbidden metadata key.");
    if (metadata.has(key)) throw new ParseError("INVALID", "Duplicate metadata key.");
    const type = c.u32();
    metadata.set(key, readValue(c, type, 0));
  }

  let alignment = 32;
  const alignValue = metadata.get("general.alignment");
  if (alignValue !== undefined) {
    if (
      typeof alignValue !== "number" ||
      !Number.isInteger(alignValue) ||
      alignValue < 1 ||
      alignValue > LIMITS.maxAlignment ||
      (alignValue & (alignValue - 1)) !== 0
    ) {
      throw new ParseError("INVALID", "Invalid general.alignment.");
    }
    alignment = alignValue;
  }

  const tensors: TensorInfo[] = [];
  let totalBytes = 0n;
  let unknownTypes = 0;
  for (let i = 0; i < tensorCount; i++) {
    const name = c.string(LIMITS.maxStringBytes);
    const nDims = c.u32();
    if (nDims < LIMITS.minDims || nDims > LIMITS.maxDims) {
      throw new ParseError("INVALID", "Tensor has an invalid number of dimensions.");
    }
    const dimsBig: bigint[] = [];
    for (let d = 0; d < nDims; d++) {
      const dim = c.u64();
      if (dim < 1n || dim > LIMITS.maxDim) throw new ParseError("INVALID", "Tensor dimension out of range.");
      dimsBig.push(dim);
    }
    const type = c.u32();
    const offsetBig = c.u64();
    if (offsetBig > MAX_SAFE) throw new ParseError("LIMIT", "Tensor offset exceeds the safe integer range.");

    let elementsBig = 1n;
    for (const d of dimsBig) elementsBig = mulCapped(elementsBig, d, LIMITS.maxModelBytes * 8n);
    if (elementsBig > MAX_SAFE) throw new ParseError("LIMIT", "Tensor element count exceeds the limit.");

    let bytes: number | null = null;
    const info = ggmlType(type);
    if (info) {
      const ne0 = dimsBig[0] as bigint;
      const rows = elementsBig / ne0;
      const blocksPerRow = (ne0 + BigInt(info.blockSize) - 1n) / BigInt(info.blockSize);
      const rowBytes = mulCapped(blocksPerRow, BigInt(info.typeSize), LIMITS.maxModelBytes);
      const tensorBytes = mulCapped(rows, rowBytes, LIMITS.maxModelBytes);
      totalBytes += tensorBytes;
      if (totalBytes > LIMITS.maxModelBytes) {
        throw new ParseError("LIMIT", "Total tensor size exceeds the 2 TB limit.");
      }
      bytes = Number(tensorBytes);
    } else {
      unknownTypes++;
    }

    tensors.push({
      name,
      dims: dimsBig.map(Number),
      type,
      offset: Number(offsetBig),
      elements: Number(elementsBig),
      bytes,
    });
  }

  const headerBytes = c.pos;
  const dataOffset = Math.ceil(headerBytes / alignment) * alignment;

  if (unknownTypes > 0) {
    warnings.push(`${unknownTypes} tensor(s) use an unknown ggml type; their sizes are estimated.`);
  }

  return {
    version,
    metadata,
    tensors: unknownTypes > 0 ? estimateUnknown(tensors, dataOffset, opts.fileSize ?? null) : tensors,
    alignment,
    dataOffset,
    headerBytes,
    warnings,
  };
}

/** Estimate unknown-type tensor sizes from the gap to the next tensor offset. */
function estimateUnknown(tensors: TensorInfo[], dataOffset: number, fileSize: number | null): TensorInfo[] {
  const byOffset = [...tensors].sort((a, b) => a.offset - b.offset);
  const nextOffset = new Map<TensorInfo, number | null>();
  for (let i = 0; i < byOffset.length; i++) {
    const t = byOffset[i] as TensorInfo;
    const next = byOffset[i + 1];
    nextOffset.set(t, next ? next.offset : fileSize !== null ? fileSize - dataOffset : null);
  }
  return tensors.map((t) => {
    if (t.bytes !== null) return t;
    const next = nextOffset.get(t) ?? null;
    let bytes = next !== null ? next - t.offset : NaN;
    // Fall back to ~4.5 bits per weight when the gap is unusable.
    if (!Number.isFinite(bytes) || bytes <= 0 || bytes > t.elements * 8) {
      bytes = Math.ceil(t.elements * 0.5625);
    }
    return { ...t, bytes };
  });
}
