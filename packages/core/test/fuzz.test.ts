import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { GgufValueType as T, ParseError, bufferSource, readGgufHeader, summarizeModel } from "../src/index.js";
import { Writer, buildGguf, llamaFixture } from "./fixture.js";

/** Parse must either succeed or throw ParseError; anything else is a bug. */
async function safeParse(bytes: Uint8Array): Promise<"ok" | "error"> {
  const started = performance.now();
  try {
    const file = await readGgufHeader(bufferSource(bytes));
    // Model summarization must also hold up on whatever parsed.
    try {
      summarizeModel([file]);
    } catch (e) {
      if (!(e instanceof ParseError)) throw e;
    }
    return "ok";
  } catch (e) {
    if (!(e instanceof ParseError)) throw new Error(`non-ParseError thrown: ${String(e)}`);
    return "error";
  } finally {
    // Never hang: every input in these tests is small, so parsing must be fast.
    expect(performance.now() - started).toBeLessThan(2000);
  }
}

const valid = llamaFixture({ layers: 2, tokenList: 20 }).bytes;
const magicAndVersion = new Writer().u32(0x46554747).u32(3).bytes();

describe("fuzz: GGUF parser", () => {
  it("random bytes", async () => {
    await fc.assert(fc.asyncProperty(fc.uint8Array({ maxLength: 4096 }), async (b) => void (await safeParse(b))), {
      numRuns: 500,
    });
  });

  it("random bytes after a valid magic + version", async () => {
    await fc.assert(
      fc.asyncProperty(fc.uint8Array({ maxLength: 4096 }), async (b) => {
        const bytes = new Uint8Array(magicAndVersion.length + b.length);
        bytes.set(magicAndVersion);
        bytes.set(b, magicAndVersion.length);
        await safeParse(bytes);
      }),
      { numRuns: 500 },
    );
  });

  it("truncated valid files", async () => {
    await fc.assert(
      fc.asyncProperty(fc.integer({ min: 0, max: valid.length }), async (n) => {
        const r = await safeParse(valid.slice(0, n));
        if (n < valid.length) expect(r).toBe("error");
      }),
      { numRuns: 300 },
    );
  });

  it("mutated valid files (byte flips)", async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.array(fc.tuple(fc.nat({ max: valid.length - 1 }), fc.integer({ min: 0, max: 255 })), { minLength: 1, maxLength: 16 }),
        async (muts) => {
          const b = valid.slice();
          for (const [i, v] of muts) b[i] = v;
          await safeParse(b);
        },
      ),
      { numRuns: 500 },
    );
  });

  it("mutated valid files (64-bit fields set to extreme values)", async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.nat({ max: valid.length - 8 }),
        fc.constantFrom(0n, 1n, 2n ** 31n, 2n ** 32n - 1n, 2n ** 53n, 2n ** 63n, 2n ** 64n - 1n),
        async (pos, v) => {
          const b = valid.slice();
          new DataView(b.buffer).setBigUint64(pos, v, true);
          await safeParse(b);
        },
      ),
      { numRuns: 500 },
    );
  });

  it("hand-crafted malicious headers", async () => {
    const h = (tc: bigint | number, kc: bigint | number) => new Writer().u32(0x46554747).u32(3).u64(tc).u64(kc);
    const cases: Uint8Array[] = [
      h(2n ** 64n - 1n, 0).bytes(),
      h(0, 2n ** 64n - 1n).bytes(),
      h(0, 1).u64(2n ** 63n).bytes(), // giant key
      h(0, 1).str("k").u32(T.STRING).u64(2n ** 40n).bytes(), // giant string
      h(0, 1).str("k").u32(T.ARRAY).u32(T.STRING).u64(2n ** 60n).bytes(), // giant array
      h(0, 1).str("k").u32(T.ARRAY).u32(T.ARRAY).u64(1).u32(T.ARRAY).u64(1).u32(T.ARRAY).u64(1).bytes(), // deep
      h(0, 1).str("__proto__").u32(T.STRING).str("polluted").bytes(),
      h(0, 1).str("constructor").u32(T.UINT32).u32(1).bytes(),
      h(0, 1).str("prototype").u32(T.UINT32).u32(1).bytes(),
      h(1, 0).str("t").u32(4).u64(2n ** 40n).u64(2n ** 40n).u64(2n ** 40n).u64(2n ** 40n).u32(0).u64(0).bytes(), // overflow
      h(1, 0).str("t").u32(1).u64(0).u32(0).u64(0).bytes(), // zero dim
      h(1, 0).str("t").u32(9).bytes(), // n_dims
      // One million empty strings in an array: allowed, but bounded and fast.
      (() => {
        const w = h(0, 1).str("k").u32(T.ARRAY).u32(T.STRING).u64(1_000_000);
        for (let i = 0; i < 1_000_000; i++) w.u64(0);
        return w.bytes();
      })(),
      // Array of 1M nested empty arrays.
      (() => {
        const w = h(0, 1).str("k").u32(T.ARRAY).u32(T.ARRAY).u64(1_000_000);
        for (let i = 0; i < 1_000_000; i++) w.u32(T.UINT8).u64(0);
        return w.bytes();
      })(),
      buildGguf({ kv: [{ key: "general.architecture", type: "str", value: "\u202Eevil\u0000" }], tensors: [] }),
      buildGguf({ kv: [{ key: "llama.block_count", type: "u64", value: 2n ** 63n }], tensors: [] }),
    ];
    for (const c of cases) await safeParse(c);
    expect(Object.prototype).not.toHaveProperty("polluted");
    expect(({} as Record<string, unknown>)["polluted"]).toBeUndefined();
  });

  it("valid structures with random metadata", async () => {
    const kvArb = fc.oneof(
      fc.record({ key: fc.string({ maxLength: 40 }), type: fc.constant("u32" as const), value: fc.nat() }),
      fc.record({ key: fc.string({ maxLength: 40 }), type: fc.constant("str" as const), value: fc.string({ maxLength: 200 }) }),
      fc.record({
        key: fc.string({ maxLength: 40 }),
        type: fc.constant("arr_i32" as const),
        value: fc.array(fc.integer(), { maxLength: 50 }),
      }),
    );
    const tensorArb = fc.record({
      name: fc.string({ maxLength: 40 }),
      dims: fc.array(fc.integer({ min: 1, max: 4096 }), { minLength: 1, maxLength: 4 }),
      type: fc.integer({ min: 0, max: 45 }),
    });
    await fc.assert(
      fc.asyncProperty(fc.array(kvArb, { maxLength: 20 }), fc.array(tensorArb, { maxLength: 20 }), async (kv, tensors) => {
        await safeParse(buildGguf({ kv, tensors }));
      }),
      { numRuns: 300 },
    );
  });
});
