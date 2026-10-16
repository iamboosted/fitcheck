import { describe, expect, it } from "vitest";
import {
  DEFAULT_STATE,
  GiB,
  ValidationError,
  bashQuote,
  buildCommands,
  isShellSafeFileName,
  parseShareState,
  plan,
  psQuote,
  sanitizeText,
  serializeShareState,
  type ModelSource,
} from "../src/index.js";

describe("shell quoting", () => {
  it("bash single-quote escaping", () => {
    expect(bashQuote("model.gguf")).toBe("'model.gguf'");
    expect(bashQuote("my model.gguf")).toBe("'my model.gguf'");
    expect(bashQuote("it's")).toBe("'it'\\''s'");
    expect(bashQuote("$(rm -rf /)")).toBe("'$(rm -rf /)'");
  });

  it("PowerShell single-quote doubling, including Unicode quotes", () => {
    expect(psQuote("my model.gguf")).toBe("'my model.gguf'");
    expect(psQuote("it's")).toBe("'it''s'");
    expect(psQuote("a\u2019b")).toBe("'a\u2019\u2019b'");
    expect(psQuote("$env:X")).toBe("'$env:X'");
  });
});

describe("local file name validation", () => {
  it("allows spaces and ordinary names", () => {
    for (const n of ["model.gguf", "My Model Q4_K_M.gguf", "qwen3 (8b).gguf", "m\u00FCde.gguf"]) {
      expect(isShellSafeFileName(n)).toBe(true);
    }
  });

  const rejected = [
    "\n", "\r", "\t", "\u0000", "\u001b", "\u007f", "\u0085",
    "'", "\"", "`", "$", ";", "|", "&", "<", ">",
    "\u2018", "\u2019", "\u201C", "\u201D", "\u202E", "\u2066", "\u200B", "\u2028",
  ];
  it.each(rejected)("rejects names containing %j", (ch) => {
    expect(isShellSafeFileName(`a${ch}b.gguf`)).toBe(false);
  });

  it("rejects paths, option-like and padded names", () => {
    for (const n of ["a/b.gguf", "a\\b.gguf", "-rf.gguf", " a.gguf", "a.gguf ", "", "x".repeat(300)]) {
      expect(isShellSafeFileName(n)).toBe(false);
    }
  });
});

describe("buildCommands", () => {
  const model = {
    layerBytes: new Array(8).fill(0.5 * GiB),
    outputBytes: 0.2 * GiB,
    tokenEmbdBytes: 0.2 * GiB,
    kvKElems: new Array(8).fill(1024),
    kvVElems: new Array(8).fill(1024),
  };
  const gpus = [
    { name: "a", vramGiB: 16, reservedGiB: 0.5 },
    { name: "b", vramGiB: 16, reservedGiB: 0 },
  ];
  const p = plan({ model, gpus, ctx: 8192, cacheK: "q8_0", cacheV: "q8_0", headroomGiB: 1 });
  const cmd = (source: ModelSource) => buildCommands({ source, plan: p, ctx: 8192, cacheK: "q8_0", cacheV: "q8_0" });

  it("local file with spaces", () => {
    const c = cmd({ kind: "local", fileName: "My Model.gguf" });
    expect(c.bash).toBe(`llama-server -m 'My Model.gguf' -ngl 999 -c 8192 -ctk q8_0 -ctv q8_0 -fa on -sm layer -ts ${p.tensorSplit.join(",")}`);
    expect(c.powershell).toContain("-m 'My Model.gguf'");
  });

  it("HF quant and HF file forms", () => {
    expect(cmd({ kind: "hf-quant", org: "org", repo: "repo", quant: "Q4_K_M" }).bash).toContain("-hf 'org/repo:Q4_K_M'");
    const f = cmd({ kind: "hf-file", org: "org", repo: "repo", path: "Q8_0/m-00001-of-00002.gguf" });
    expect(f.bash).toContain("--hf-repo 'org/repo' --hf-file 'Q8_0/m-00001-of-00002.gguf'");
  });

  it("refuses unsafe file names instead of emitting them", () => {
    for (const name of ["a';rm -rf ~;'.gguf", "a`whoami`.gguf", "a$(id).gguf", "a\nb.gguf", "a\u2019.gguf"]) {
      expect(() => cmd({ kind: "local", fileName: name })).toThrow(ValidationError);
    }
  });

  it("refuses unvalidated HF references", () => {
    expect(() => cmd({ kind: "hf-quant", org: "o;rm", repo: "r", quant: "Q4" })).toThrow(ValidationError);
    expect(() => cmd({ kind: "hf-file", org: "o", repo: "r", path: "../x.gguf" })).toThrow(ValidationError);
  });

  it("emits numbers from numbers only", () => {
    expect(() =>
      buildCommands({ source: { kind: "local", fileName: "a.gguf" }, plan: p, ctx: Number.NaN, cacheK: "f16", cacheV: "f16" }),
    ).toThrow(ValidationError);
    expect(() =>
      buildCommands({
        source: { kind: "local", fileName: "a.gguf" },
        plan: p,
        ctx: 8192,
        cacheK: "f16; rm" as "f16",
        cacheV: "f16",
      }),
    ).toThrow(ValidationError);
  });

  it("omits cache flags and -fa for f16", () => {
    const p16 = plan({ model, gpus: [gpus[0]!], ctx: 4096, cacheK: "f16", cacheV: "f16", headroomGiB: 1 });
    const c = buildCommands({ source: { kind: "local", fileName: "a.gguf" }, plan: p16, ctx: 4096, cacheK: "f16", cacheV: "f16" });
    expect(c.bash).toBe("llama-server -m 'a.gguf' -ngl 999 -c 4096");
  });
});

describe("URL state", () => {
  const parse = (qs: string) => parseShareState(new URLSearchParams(qs));

  it("round-trips valid state", () => {
    const s = parse("hf=org/repo&q=Q4_K_M&g=rx7900xtx:0.5&g=24&g=12.5:1&ctx=32768&kv=q8_0&hr=1.5");
    expect(s.hf).toEqual({ org: "org", repo: "repo", revision: "main", quant: "Q4_K_M", path: null });
    expect(s.gpus).toEqual([
      { preset: "rx7900xtx", vramGiB: 24, reservedGiB: 0.5 },
      { preset: "custom", vramGiB: 24, reservedGiB: 0 },
      { preset: "custom", vramGiB: 12.5, reservedGiB: 1 },
    ]);
    expect(s.ctx).toBe(32768);
    expect(s.cache).toBe("q8_0");
    expect(s.headroomGiB).toBe(1.5);
    expect(parseShareState(serializeShareState(s))).toEqual(s);
  });

  it("drops invalid params silently and uses defaults", () => {
    const s = parse(
      "hf=../../etc&q=Q4;rm&g=0&g=513&g=-1&g=1e3&g=nope&g=24:24&g=24:x&ctx=100&kv=f32&hr=-1",
    );
    expect(s.hf).toBeNull();
    expect(s.gpus).toEqual(DEFAULT_STATE.gpus);
    expect(s.ctx).toBe(DEFAULT_STATE.ctx);
    expect(s.cache).toBe(DEFAULT_STATE.cache);
    expect(s.headroomGiB).toBe(DEFAULT_STATE.headroomGiB);
  });

  it("validates hf parts individually", () => {
    const s = parse("hf=org/repo&q=Q4;rm&f=../x.gguf&rev=a/b");
    expect(s.hf).toEqual({ org: "org", repo: "repo", revision: "main", quant: null, path: null });
    expect(parse("hf=org/repo/extra").hf).toBeNull();
    expect(parse("hf=" + "a".repeat(800) + "/b").hf).toBeNull();
  });

  it("caps cards at 16 and bounds ctx", () => {
    expect(parse(Array.from({ length: 40 }, () => "g=24").join("&")).gpus).toHaveLength(16);
    expect(parse("ctx=1048577").ctx).toBe(DEFAULT_STATE.ctx);
    expect(parse("ctx=1048576").ctx).toBe(1048576);
    expect(parse("ctx=512").ctx).toBe(512);
    expect(parse("ctx=4096.5").ctx).toBe(DEFAULT_STATE.ctx);
  });

  it("never serializes tokens", () => {
    const qs = serializeShareState({ ...DEFAULT_STATE, hf: { org: "o", repo: "r", revision: "main", quant: "Q4_0", path: null } }).toString();
    expect(qs).not.toMatch(/token|hf_/i);
  });
});

describe("sanitizeText", () => {
  it("strips controls, bidi overrides and zero-width characters", () => {
    expect(sanitizeText("a\u0000b\u001bc")).toBe("a b c");
    expect(sanitizeText("abc\u202Edcba")).toBe("abcdcba");
    expect(sanitizeText("x\u2066y\u2069z")).toBe("xyz");
    expect(sanitizeText("zero\u200Bwidth\uFEFF")).toBe("zerowidth");
    expect(sanitizeText("line\nbreak\u2028x")).toBe("line breakx");
  });

  it("truncates and handles non-strings", () => {
    expect(sanitizeText("x".repeat(500), 10)).toBe("xxxxxxxxx\u2026");
    expect(Array.from(sanitizeText("\u{1F600}".repeat(50), 5))).toHaveLength(5);
    expect(sanitizeText(42)).toBe("");
    expect(sanitizeText(undefined)).toBe("");
  });
});
