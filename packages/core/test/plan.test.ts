import { describe, expect, it } from "vitest";
import {
  CACHE_BYTES_PER_ELEM,
  GiB,
  ParseError,
  bufferSource,
  kvBytesPerLayer,
  maxContext,
  plan,
  proportionalCounts,
  readGgufHeader,
  summarizeModel,
  totalKvBytes,
  type PlanModel,
} from "../src/index.js";
import { llamaFixture } from "./fixture.js";

/** Uniform synthetic model: L layers of `layerGiB` each, n_kv_heads=8, head_dim=128. */
function model(L: number, layerGiB: number, outputGiB = 0.5): PlanModel {
  return {
    layerBytes: new Array(L).fill(layerGiB * GiB),
    outputBytes: outputGiB * GiB,
    tokenEmbdBytes: 0.5 * GiB,
    kvKElems: new Array(L).fill(8 * 128),
    kvVElems: new Array(L).fill(8 * 128),
  };
}

describe("KV cache math", () => {
  it("matches 2 * n_layers * n_kv_heads * head_dim * ctx * bytes", () => {
    const m = model(32, 0.1);
    const ctx = 8192;
    expect(totalKvBytes(m, ctx, "f16", "f16")).toBe(2 * 32 * 8 * 128 * ctx * 2);
    // Llama-3-8B at 8K f16 is the well-known 1 GiB.
    expect(totalKvBytes(m, ctx, "f16", "f16")).toBe(GiB);
  });

  it("uses block sizes for quantized caches", () => {
    expect(CACHE_BYTES_PER_ELEM.q8_0).toBe(34 / 32);
    expect(CACHE_BYTES_PER_ELEM.q4_0).toBe(18 / 32);
    const m = model(32, 0.1);
    expect(totalKvBytes(m, 8192, "q8_0", "q8_0")).toBeCloseTo((GiB * 34) / 64, 0);
  });

  it("handles per-layer KV heads and mixed K/V types", () => {
    const m: PlanModel = { ...model(2, 0.1), kvKElems: [1024, 0], kvVElems: [1024, 0] };
    const kv = kvBytesPerLayer(m, 100, "f16", "q4_0");
    expect(kv[0]).toBeCloseTo(100 * (1024 * 2 + 1024 * (18 / 32)));
    expect(kv[1]).toBe(0);
  });
});

describe("planner", () => {
  const gpu = (vramGiB: number, reservedGiB = 0) => ({ name: "gpu", vramGiB, reservedGiB });

  it("fits a small model on one GPU", () => {
    const p = plan({ model: model(10, 0.5), gpus: [gpu(16)], ctx: 4096, cacheK: "f16", cacheV: "f16", headroomGiB: 1 });
    expect(p.fits).toBe(true);
    expect(p.ngl).toBe(11);
    expect(p.gpus[0]?.layers).toEqual([0, 10]);
    expect(p.gpus[0]?.hasOutput).toBe(true);
    expect(p.cpuSpillBytes).toBe(0);
    expect(p.cpuWeightBytes).toBe(0.5 * GiB); // token_embd stays on CPU
    expect(p.tensorSplit).toEqual([11]);
  });

  it("weights layer assignment by usable VRAM and puts output on the last GPU", () => {
    const p = plan({ model: model(30, 0.3), gpus: [gpu(24), gpu(12)], ctx: 2048, cacheK: "f16", cacheV: "f16", headroomGiB: 1 });
    expect(p.fits).toBe(true);
    const [a, b] = p.gpus;
    expect(a!.layers).toEqual([0, 20]);
    expect(b!.layers).toEqual([20, 30]);
    expect(b!.hasOutput).toBe(true);
    expect(a!.hasOutput).toBe(false);
    expect(p.tensorSplit).toEqual([20, 11]);
    expect(p.tensorSplit.reduce((x, y) => x + y)).toBe(31);
  });

  it("subtracts reserved VRAM and headroom", () => {
    const m = model(10, 1.4, 0.1);
    const base = { model: m, ctx: 512, cacheK: "f16" as const, cacheV: "f16" as const };
    // 16 GiB - 1 headroom = 15 >= 14.1 (+ tiny KV): fits.
    expect(plan({ ...base, gpus: [gpu(16)], headroomGiB: 1 }).fits).toBe(true);
    // Reserve 1.5 GiB for the desktop: no longer fits.
    expect(plan({ ...base, gpus: [gpu(16, 1.5)], headroomGiB: 1 }).fits).toBe(false);
  });

  it("falls back to a greedy split when proportional overflows a card", () => {
    // Uneven layers: big ones at the front.
    const m: PlanModel = { ...model(4, 1), layerBytes: [6, 6, 1, 1].map((g) => g * GiB) };
    const p = plan({ model: m, gpus: [gpu(13), gpu(13)], ctx: 512, cacheK: "f16", cacheV: "f16", headroomGiB: 0 });
    expect(p.fits).toBe(true);
    for (const g of p.gpus) expect(g.free).toBeGreaterThanOrEqual(0);
  });

  it("reports max -ngl and CPU spill when it doesn't fit", () => {
    const m = model(40, 1, 1);
    const p = plan({ model: m, gpus: [gpu(16)], ctx: 4096, cacheK: "f16", cacheV: "f16", headroomGiB: 1 });
    expect(p.fits).toBe(false);
    // 15 GiB usable, each layer ~1 GiB + 16 MiB of KV: 14 layers.
    expect(p.ngl).toBe(14);
    expect(p.gpus[0]?.layers).toEqual([26, 40]); // llama.cpp offloads the last layers
    expect(p.cpuSpillBytes).toBe(26 * GiB + 1 * GiB);
    expect(p.cpuKvBytes).toBeGreaterThan(0);
    for (const g of p.gpus) expect(g.free).toBeGreaterThanOrEqual(0);
  });

  it("returns ngl 0 when nothing fits", () => {
    const p = plan({ model: model(4, 50), gpus: [gpu(8)], ctx: 512, cacheK: "f16", cacheV: "f16", headroomGiB: 1 });
    expect(p.ngl).toBe(0);
    expect(p.fits).toBe(false);
  });

  it("flags flash attention for quantized V cache only", () => {
    const args = { model: model(4, 0.1), gpus: [gpu(16)], ctx: 512, headroomGiB: 1 };
    expect(plan({ ...args, cacheK: "q8_0", cacheV: "f16" }).flashAttention).toBe(false);
    expect(plan({ ...args, cacheK: "q8_0", cacheV: "q8_0" }).flashAttention).toBe(true);
    expect(plan({ ...args, cacheK: "f16", cacheV: "q4_0" }).flashAttention).toBe(true);
  });

  it("never overcommits a GPU across many configurations", () => {
    for (const L of [1, 7, 32, 61]) {
      for (const n of [1, 2, 3, 5]) {
        for (const ctx of [512, 32768, 262144]) {
          const gpus = Array.from({ length: n }, (_, i) => gpu(8 + 4 * i, i === 0 ? 0.5 : 0));
          const p = plan({ model: model(L, 0.7), gpus, ctx, cacheK: "q8_0", cacheV: "q8_0", headroomGiB: 1 });
          for (const g of p.gpus) expect(g.free).toBeGreaterThanOrEqual(0);
          expect(p.tensorSplit.reduce((a, b) => a + b, 0)).toBe(p.ngl);
        }
      }
    }
  });
});

describe("proportionalCounts", () => {
  it("sums to the total and respects weights", () => {
    expect(proportionalCounts(10, [1, 1])).toEqual([5, 5]);
    expect(proportionalCounts(10, [2, 1])).toEqual([7, 3]);
    expect(proportionalCounts(0, [1, 1])).toEqual([0, 0]);
    expect(proportionalCounts(5, [0, 0])).toEqual([0, 0]);
    expect(proportionalCounts(7, [1, 1, 1]).reduce((a, b) => a + b)).toBe(7);
  });
});

describe("maxContext", () => {
  const base = { gpus: [{ name: "g", vramGiB: 24, reservedGiB: 0 }], cacheK: "f16" as const, cacheV: "f16" as const, headroomGiB: 1 };

  it("finds the largest fitting context", () => {
    const m = model(32, 0.5);
    const c = maxContext({ ...base, model: m });
    expect(c).not.toBeNull();
    expect(c! % 256).toBe(0);
    expect(plan({ ...base, model: m, ctx: c! }).fits).toBe(true);
    expect(plan({ ...base, model: m, ctx: c! + 256 }).fits).toBe(false);
  });

  it("returns null when even 512 doesn't fit and caps at the upper bound", () => {
    expect(maxContext({ ...base, model: model(32, 1) })).toBeNull();
    expect(maxContext({ ...base, model: { ...model(1, 0.1), kvKElems: [0], kvVElems: [0] } })).toBe(1_048_576);
  });

  it("allows more context with a quantized cache", () => {
    const m = model(32, 0.5);
    const f16 = maxContext({ ...base, model: m })!;
    const q8 = maxContext({ ...base, model: m, cacheK: "q8_0", cacheV: "q8_0" })!;
    expect(q8).toBeGreaterThan(f16 * 1.8);
  });
});

describe("summarizeModel", () => {
  it("groups bytes by layer and finds attention geometry", async () => {
    const { bytes } = llamaFixture({ layers: 4, embd: 256, heads: 8, kvHeads: 2 });
    const info = summarizeModel([await readGgufHeader(bufferSource(bytes))]);
    expect(info.architecture).toBe("llama");
    expect(info.nLayers).toBe(4);
    expect(info.layerBytes.every((b) => b > 0)).toBe(true);
    expect(info.headDim).toBe(32);
    expect(info.kvKElems).toEqual([64, 64, 64, 64]);
    expect(info.quant).toBe("Q4_K_M");
    expect(info.tiedOutput).toBe(false);
    expect(info.hiddenMetadata.tokens).toBe(100);
    expect(info.hiddenMetadata.chatTemplateBytes).toBeGreaterThan(0);
    const sum = info.layerBytes.reduce((a, b) => a + b, 0) + info.tokenEmbdBytes + info.outputBytes + info.otherBytes;
    expect(sum).toBe(info.totalBytes);
  });

  it("treats tied embeddings as an output copy", async () => {
    const { bytes } = llamaFixture({ tied: true });
    const info = summarizeModel([await readGgufHeader(bufferSource(bytes))]);
    expect(info.tiedOutput).toBe(true);
    expect(info.outputBytes).toBeGreaterThanOrEqual(info.tokenEmbdBytes);
  });

  it("merges shards and rejects incomplete sets", async () => {
    const mk = (no: number, count: number) =>
      llamaFixture({
        extraKv: [
          { key: "split.no", type: "u32", value: no },
          { key: "split.count", type: "u32", value: count },
        ],
      }).bytes;
    const a = await readGgufHeader(bufferSource(mk(0, 2)));
    const b = await readGgufHeader(bufferSource(mk(1, 2)));
    const merged = summarizeModel([b, a]);
    expect(merged.shardCount).toBe(2);
    expect(() => summarizeModel([a])).toThrow(ParseError);
    expect(() => summarizeModel([a, a])).toThrow(ParseError);
  });

  it("ignores hostile architecture strings", async () => {
    const { bytes } = llamaFixture({ extraKv: [] });
    const file = await readGgufHeader(bufferSource(bytes));
    const md = new Map(file.header.metadata);
    md.set("general.architecture", "../../evil<script>");
    const info = summarizeModel([{ ...file, header: { ...file.header, metadata: md } }]);
    expect(info.architecture).toBe("unknown");
  });
});
