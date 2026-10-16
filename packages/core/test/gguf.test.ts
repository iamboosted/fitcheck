import { describe, expect, it } from "vitest";
import {
  GgufValueType as T,
  LIMITS,
  ParseError,
  blobSource,
  bufferSource,
  parseGgufHeader,
  readGgufHeader,
  type ByteSource,
  type GgufArray,
} from "../src/index.js";
import { Writer, buildGguf, llamaFixture, tensorBytes, type KvSpec } from "./fixture.js";

function header(tensorCount: number | bigint, kvCount: number | bigint, version = 3): Writer {
  return new Writer().u32(0x46554747).u32(version).u64(tensorCount).u64(kvCount);
}

async function parse(bytes: Uint8Array) {
  return readGgufHeader(bufferSource(bytes));
}

async function expectParseError(bytes: Uint8Array | Promise<unknown>, code?: string) {
  const p = bytes instanceof Uint8Array ? parse(bytes) : bytes;
  const err = await p.then(
    () => null,
    (e: unknown) => e,
  );
  expect(err).toBeInstanceOf(ParseError);
  if (code) expect((err as ParseError).code).toBe(code);
  return err as ParseError;
}

describe("parseGgufHeader: valid files", () => {
  it("parses metadata and tensor infos of a llama fixture", async () => {
    const { bytes, tensors } = llamaFixture({ layers: 3 });
    const { header: h, fileSize, fetches } = await parse(bytes);
    expect(h.version).toBe(3);
    expect(fileSize).toBe(bytes.length);
    expect(fetches).toBe(1);
    expect(h.metadata.get("general.architecture")).toBe("llama");
    expect(h.metadata.get("llama.block_count")).toBe(3);
    expect(h.tensors).toHaveLength(tensors.length);
    for (let i = 0; i < tensors.length; i++) {
      expect(h.tensors[i]?.bytes).toBe(tensorBytes(tensors[i]!));
      expect(h.tensors[i]?.name).toBe(tensors[i]!.name);
    }
    expect(h.dataOffset % 32).toBe(0);
    expect(h.dataOffset).toBeGreaterThanOrEqual(h.headerBytes);
    expect(h.warnings).toEqual([]);
  });

  it("stores metadata in a Map, never on a plain object", async () => {
    const { header: h } = await parse(llamaFixture().bytes);
    expect(h.metadata).toBeInstanceOf(Map);
  });

  it("summarizes long string arrays instead of keeping them", async () => {
    const { header: h } = await parse(llamaFixture({ tokenList: 5000 }).bytes);
    const tokens = h.metadata.get("tokenizer.ggml.tokens") as GgufArray;
    expect(tokens.kind).toBe("array");
    expect(tokens.length).toBe(5000);
    expect(tokens.values).toBeUndefined();
  });

  it("keeps short numeric arrays (per-layer head_count_kv)", async () => {
    const bytes = buildGguf({
      kv: [{ key: "x.per_layer", type: "arr_i32", value: [1, 2, 3] }],
      tensors: [],
    });
    const { header: h } = await parse(bytes);
    expect((h.metadata.get("x.per_layer") as GgufArray).values).toEqual([1, 2, 3]);
  });

  it("accepts GGUF v2", async () => {
    const bytes = buildGguf({ version: 2, kv: [], tensors: [] });
    expect((await parse(bytes)).header.version).toBe(2);
  });

  it("computes sizes for every K-quant and IQ type", async () => {
    const types = [2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 34, 35, 39];
    const tensors = types.map((type, i) => ({ name: `t${i}`, dims: [512, 3], type }));
    const { header: h } = await parse(buildGguf({ kv: [], tensors }));
    for (let i = 0; i < types.length; i++) expect(h.tensors[i]?.bytes).toBe(tensorBytes(tensors[i]!));
    // Spot check: Q4_K is 144 bytes per 256 weights.
    expect(h.tensors[types.indexOf(12)]?.bytes).toBe(2 * 144 * 3);
    // MXFP4 is 17 bytes per 32 weights.
    expect(h.tensors[types.indexOf(39)]?.bytes).toBe(16 * 17 * 3);
  });

  it("warns and estimates unknown tensor types from offsets", async () => {
    const bytes = buildGguf({
      kv: [],
      tensors: [
        { name: "a", dims: [64, 64], type: 99 },
        { name: "b", dims: [64, 64], type: 0 },
      ],
    });
    const { header: h } = await parse(bytes);
    expect(h.warnings.join(" ")).toMatch(/unknown ggml type/);
    expect(h.tensors[0]?.bytes).toBeGreaterThan(0);
  });

  it("honors a custom power-of-two alignment", async () => {
    const bytes = buildGguf({ kv: [{ key: "general.alignment", type: "u32", value: 64 }], tensors: [] });
    const { header: h } = await parse(bytes);
    expect(h.alignment).toBe(64);
    expect(h.dataOffset % 64).toBe(0);
  });
});

describe("parseGgufHeader: rejects hostile input", () => {
  it("bad magic", async () => {
    await expectParseError(new Uint8Array([1, 2, 3, 4, 3, 0, 0, 0]), "BAD_MAGIC");
  });

  it("unknown versions", async () => {
    for (const v of [0, 1, 4, 0xffffffff]) await expectParseError(header(0, 0, v).bytes(), "BAD_VERSION");
  });

  it("tensor_count over 100k", async () => {
    await expectParseError(header(LIMITS.maxTensorCount + 1, 0).bytes(), "LIMIT");
    await expectParseError(header(2n ** 64n - 1n, 0).bytes(), "LIMIT");
  });

  it("kv_count over 10k", async () => {
    await expectParseError(header(0, LIMITS.maxKvCount + 1).bytes(), "LIMIT");
  });

  it("strings longer than 1 MB", async () => {
    const w = header(0, 1).str("k").u32(T.STRING).u64(LIMITS.maxStringBytes + 1);
    await expectParseError(w.bytes(), "LIMIT");
    const w2 = header(0, 1).str("k").u32(T.STRING).u64(2n ** 63n);
    await expectParseError(w2.bytes(), "LIMIT");
  });

  it("huge key length", async () => {
    await expectParseError(header(0, 1).u64(2n ** 62n).bytes(), "LIMIT");
  });

  it("arrays longer than 1M elements", async () => {
    const w = header(0, 1).str("k").u32(T.ARRAY).u32(T.UINT8).u64(LIMITS.maxArrayLength + 1);
    await expectParseError(w.bytes(), "LIMIT");
  });

  it("arrays nested deeper than 2", async () => {
    const w = header(0, 1).str("k").u32(T.ARRAY).u32(T.ARRAY).u64(1).u32(T.ARRAY).u64(1).u32(T.UINT8).u64(0);
    await expectParseError(w.bytes(), "LIMIT");
    // Depth 2 is fine.
    const ok = header(0, 1).str("k").u32(T.ARRAY).u32(T.ARRAY).u64(1).u32(T.UINT8).u64(1).u8(7);
    expect((await parse(ok.bytes())).header.metadata.get("k")).toMatchObject({ kind: "array", length: 1 });
  });

  it.each(["__proto__", "constructor", "prototype"])("forbidden key %s", async (key) => {
    const w = header(0, 1).str(key).u32(T.UINT32).u32(1);
    const err = await expectParseError(w.bytes(), "INVALID");
    expect(err.message).not.toContain(key);
    expect(({} as Record<string, unknown>)["polluted"]).toBeUndefined();
  });

  it("duplicate keys", async () => {
    const w = header(0, 2).str("a").u32(T.UINT32).u32(1).str("a").u32(T.UINT32).u32(2);
    await expectParseError(w.bytes(), "INVALID");
  });

  it("unknown value types", async () => {
    await expectParseError(header(0, 1).str("k").u32(13).u32(0).bytes(), "INVALID");
    await expectParseError(header(0, 1).str("k").u32(T.ARRAY).u32(99).u64(1).bytes(), "INVALID");
  });

  it("n_dims outside 1..4", async () => {
    for (const n of [0, 5, 0xffffffff]) {
      const w = header(1, 0).str("t").u32(n);
      await expectParseError(w.bytes(), "INVALID");
    }
  });

  it("dims outside 1..2^40", async () => {
    for (const d of [0n, 2n ** 40n + 1n, 2n ** 64n - 1n]) {
      const w = header(1, 0).str("t").u32(1).u64(d).u32(0).u64(0);
      await expectParseError(w.bytes(), "INVALID");
    }
  });

  it("element count / size overflow", async () => {
    const w = header(1, 0).str("t").u32(4).u64(2n ** 40n).u64(2n ** 40n).u64(2n ** 40n).u64(2n ** 40n).u32(0).u64(0);
    await expectParseError(w.bytes(), "LIMIT");
  });

  it("total model size over 2 TB", async () => {
    // Each tensor: 2^39 F32 elements = 2 TiB; two of them is too much even if one would pass.
    const w = header(2, 0);
    for (let i = 0; i < 2; i++) w.str(`t${i}`).u32(2).u64(2n ** 20n).u64(2n ** 19n).u32(0).u64(0);
    await expectParseError(w.bytes(), "LIMIT");
  });

  it("tensor offset above MAX_SAFE_INTEGER", async () => {
    const w = header(1, 0).str("t").u32(1).u64(32).u32(0).u64(2n ** 60n);
    await expectParseError(w.bytes(), "LIMIT");
  });

  it("invalid alignment", async () => {
    for (const a of [0, 3, 2 ** 30]) {
      const bytes = buildGguf({ kv: [{ key: "general.alignment", type: "u32", value: a }], tensors: [] });
      await expectParseError(bytes, "INVALID");
    }
  });

  it("truncated files", async () => {
    const { bytes } = llamaFixture();
    for (const n of [0, 3, 7, 20, 100, bytes.length - 1]) {
      await expectParseError(bytes.slice(0, n), "TRUNCATED");
    }
  });

  it("error messages never echo input strings", async () => {
    const evil = "<img src=x onerror=alert(1)>";
    const w = header(0, 1).str(evil).u32(77);
    const err = await expectParseError(w.bytes());
    expect(err.message).not.toContain("<img");
  });
});

describe("readGgufHeader: chunked reads", () => {
  it("never reads the tensor data region", async () => {
    const { bytes } = llamaFixture({ dataBytes: 20 * 1024 * 1024 });
    const src = bufferSource(bytes);
    let maxEnd = 0;
    const spy: ByteSource = {
      size: src.size,
      read: (o, l) => {
        maxEnd = Math.max(maxEnd, o + l);
        return src.read(o, l);
      },
    };
    const r = await readGgufHeader(spy);
    expect(r.fetches).toBe(1);
    expect(maxEnd).toBeLessThanOrEqual(LIMITS.initialChunkBytes);
    expect(r.bytesRead).toBeLessThan(bytes.length);
  });

  it("grows the buffer when the header is larger than the first chunk", async () => {
    // ~9 MB of tokenizer strings forces at least two extensions past 4 MB.
    const big = "x".repeat(1000);
    const extraKv: KvSpec[] = [{ key: "big.list", type: "arr_str", value: Array.from({ length: 9000 }, () => big) }];
    const { bytes } = llamaFixture({ extraKv, dataBytes: 1024 * 1024 });
    const r = await readGgufHeader(bufferSource(bytes));
    expect(r.fetches).toBeGreaterThanOrEqual(2);
    expect(r.header.metadata.get("general.architecture")).toBe("llama");
  });

  it("works with sources of unknown size (HTTP without Content-Range)", async () => {
    const { bytes } = llamaFixture();
    const src: ByteSource = { size: null, read: async (o, l) => bytes.slice(o, o + l) };
    const r = await readGgufHeader(src, { initialChunkBytes: 64 });
    expect(r.header.tensors.length).toBeGreaterThan(0);
  });

  it("aborts after 32 fetches", async () => {
    const { bytes } = llamaFixture();
    const src: ByteSource = { size: bytes.length, read: async (o) => bytes.slice(o, o + 1) };
    await expectParseError(readGgufHeader(src), "TOO_MANY_FETCHES");
  });

  it("truncates sources that return more than requested", async () => {
    const { bytes } = llamaFixture();
    const requested: number[] = [];
    // Hostile source: ignores the requested length and returns everything from offset.
    const src: ByteSource = {
      size: null,
      read: async (o, l) => {
        requested.push(l);
        return bytes.slice(o);
      },
    };
    const r = await readGgufHeader(src, { initialChunkBytes: 32 });
    expect(r.bytesRead).toBeLessThanOrEqual(requested.reduce((a, b) => a + b, 0));
    expect(r.header.tensors.length).toBeGreaterThan(0);
  });

  it("rejects headers over 64 MB", async () => {
    // 65 metadata strings of ~1 MB each.
    const w = header(0, 65);
    const s = "a".repeat(LIMITS.maxStringBytes);
    for (let i = 0; i < 65; i++) w.str(`k${i}`).u32(T.STRING).str(s);
    await expectParseError(parse(w.bytes()), "LIMIT");
  });

  it("reads Blobs through slice() only", async () => {
    const { bytes } = llamaFixture({ dataBytes: 8 * 1024 * 1024 });
    const blob = new Blob([bytes]);
    const sliced: Array<[number, number]> = [];
    const orig = blob.slice.bind(blob);
    blob.slice = (start?: number, end?: number) => {
      sliced.push([start ?? 0, end ?? blob.size]);
      return orig(start, end);
    };
    const r = await readGgufHeader(blobSource(blob));
    expect(r.header.metadata.get("general.name")).toBe("Test Model");
    expect(sliced.every(([s, e]) => e - s <= LIMITS.initialChunkBytes)).toBe(true);
  });

  it("parseGgufHeader rejects reading past the cap even with more data", () => {
    const { bytes } = llamaFixture();
    expect(() => parseGgufHeader(bytes, { complete: false, maxHeaderBytes: 16 })).toThrow(ParseError);
  });
});
