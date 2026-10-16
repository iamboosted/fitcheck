import { describe, expect, it } from "vitest";
import {
  ParseError,
  ValidationError,
  buildResolveUrl,
  groupGgufFiles,
  hfFetch,
  hfListGgufFiles,
  hfRangeSource,
  isAllowedHfHost,
  matchQuant,
  parseHfInput,
  parseTreeListing,
  quantFromFileName,
  readGgufHeader,
  shardPaths,
  type FetchLike,
} from "../src/index.js";
import { llamaFixture } from "./fixture.js";

const ref = { org: "org", repo: "repo", revision: "main" };

describe("parseHfInput", () => {
  it("accepts blob and resolve URLs", () => {
    expect(parseHfInput("https://huggingface.co/unsloth/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q4_K_M.gguf")).toEqual({
      kind: "file",
      org: "unsloth",
      repo: "Qwen3-8B-GGUF",
      revision: "main",
      path: "Qwen3-8B-Q4_K_M.gguf",
    });
    expect(parseHfInput("https://huggingface.co/a/b/resolve/main/Q4_K_M/m-00001-of-00002.gguf?download=true")).toMatchObject({
      kind: "file",
      path: "Q4_K_M/m-00001-of-00002.gguf",
    });
  });

  it("accepts shorthand forms", () => {
    expect(parseHfInput("org/repo:Q4_K_M")).toEqual({ kind: "quant", org: "org", repo: "repo", revision: "main", quant: "Q4_K_M" });
    expect(parseHfInput("  org/repo  ")).toEqual({ kind: "repo", org: "org", repo: "repo", revision: "main" });
    expect(parseHfInput("https://huggingface.co/org/repo")).toMatchObject({ kind: "repo" });
    expect(parseHfInput("https://huggingface.co/org/repo/tree/main")).toMatchObject({ kind: "repo" });
  });

  it.each([
    "http://huggingface.co/a/b/blob/main/x.gguf",
    "https://huggingface.co.evil.com/a/b/blob/main/x.gguf",
    "https://evil.com/a/b/blob/main/x.gguf",
    "https://hf.co/a/b/blob/main/x.gguf",
    "https://user:pass@huggingface.co/a/b/blob/main/x.gguf",
    "https://huggingface.co:8443/a/b/blob/main/x.gguf",
    "https://huggingface.co/a/b/blob/main/x.bin",
    "https://huggingface.co/a/b/blob/main/../../x.gguf",
    "https://huggingface.co/a/b/blob/main/%2e%2e/x.gguf",
    "https://huggingface.co/a/b/blob/main/sub%2Fdir/x.gguf",
    "https://huggingface.co/a/b/blob/main/x%00.gguf",
    "https://huggingface.co/a/b/blob/main/x y.gguf",
    "https://huggingface.co/a/b/blob/main/x\".gguf",
    "https://huggingface.co/a/b/blob/ma%ZZin/x.gguf",
    "https://huggingface.co/a/b/raw/main/x.gguf",
    ["javascript", "alert(1)"].join(":"),
    "data:text/html,hi",
    "file:///etc/passwd",
    "org/repo:Q4_K_M;rm -rf /",
    "org/repo:UD-Q4_K_XL",
    "-org/repo",
    ".org/repo",
    "org/re..po",
    "org/repo/extra",
    "a".repeat(97) + "/repo",
    "",
    "x".repeat(5000),
  ])("rejects %s", (s) => {
    expect(() => parseHfInput(s)).toThrow(ValidationError);
  });
});

describe("host allowlist", () => {
  it.each([
    ["huggingface.co", true],
    ["cdn-lfs.huggingface.co", true],
    ["cas-bridge.xethub.hf.co", true],
    ["cdn-lfs-us-1.hf.co", true],
    ["evilhuggingface.co", false],
    ["huggingface.co.evil.com", false],
    ["evilhf.co", false],
    ["hf.co.evil.com", false],
    ["example.com", false],
  ])("%s -> %s", (host, ok) => {
    expect(isAllowedHfHost(host)).toBe(ok);
  });
});

interface Recorded {
  url: string;
  init: RequestInit;
}

function mockFetch(handler: (url: string, init: RequestInit) => Response, log: Recorded[] = []): FetchLike {
  return async (url, init) => {
    log.push({ url, init });
    return handler(url, init);
  };
}

function withUrl(res: Response, url: string): Response {
  Object.defineProperty(res, "url", { value: url });
  return res;
}

describe("hfFetch", () => {
  it("sends the token only to huggingface.co with credentials omitted and no referrer", async () => {
    const log: Recorded[] = [];
    const f = mockFetch((u) => withUrl(new Response("ok"), u), log);
    await hfFetch(new URL("https://huggingface.co/api/x"), { fetch: f, token: "hf_secret" });
    await hfFetch(new URL("https://cdn-lfs.hf.co/x"), { fetch: f, token: "hf_secret" });
    const h0 = new Headers(log[0]!.init.headers);
    const h1 = new Headers(log[1]!.init.headers);
    expect(h0.get("authorization")).toBe("Bearer hf_secret");
    expect(h1.get("authorization")).toBeNull();
    for (const r of log) {
      expect(r.init.credentials).toBe("omit");
      expect(r.init.referrerPolicy).toBe("no-referrer");
      expect(r.url).not.toContain("hf_secret");
    }
  });

  it("refuses non-HF request URLs before fetching", async () => {
    const log: Recorded[] = [];
    const f = mockFetch(() => new Response("x"), log);
    await expect(hfFetch(new URL("https://evil.com/x"), { fetch: f })).rejects.toBeInstanceOf(ParseError);
    await expect(hfFetch(new URL("http://huggingface.co/x"), { fetch: f })).rejects.toBeInstanceOf(ParseError);
    expect(log).toHaveLength(0);
  });

  it("aborts when redirected to a non-HF host", async () => {
    const f = mockFetch(() => withUrl(new Response("x"), "https://evil.example/steal"));
    const err = await hfFetch(new URL("https://huggingface.co/a"), { fetch: f, token: "hf_secret" }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ParseError);
    expect((err as Error).message).toMatch(/non-Hugging Face/);
    expect((err as Error).message).not.toContain("hf_secret");
    expect((err as Error).message).not.toContain("evil");
  });

  it("allows redirects to *.hf.co and *.huggingface.co", async () => {
    for (const host of ["https://cas-bridge.xethub.hf.co/x", "https://cdn-lfs.huggingface.co/x"]) {
      const f = mockFetch(() => withUrl(new Response("x"), host));
      await expect(hfFetch(new URL("https://huggingface.co/a"), { fetch: f })).resolves.toBeInstanceOf(Response);
    }
  });

  it("rejects tokens with control characters", async () => {
    const f = mockFetch((u) => withUrl(new Response("x"), u));
    await expect(hfFetch(new URL("https://huggingface.co/a"), { fetch: f, token: "a\r\nX-Evil: 1" })).rejects.toBeInstanceOf(
      ValidationError,
    );
  });

  it("maps network failures to a generic error without leaking the token", async () => {
    const f: FetchLike = async () => {
      throw new TypeError("Failed to fetch hf_secret");
    };
    const err = await hfFetch(new URL("https://huggingface.co/a"), { fetch: f, token: "hf_secret" }).catch((e: unknown) => e);
    expect((err as Error).message).not.toContain("hf_secret");
  });
});

describe("hfRangeSource", () => {
  const { bytes } = llamaFixture({ dataBytes: 6 * 1024 * 1024 });

  function rangeServer(opts: { ignoreRange?: boolean; exposeContentRange?: boolean } = {}) {
    const log: Recorded[] = [];
    const f = mockFetch((url, init) => {
      const range = new Headers(init.headers).get("range");
      const m = range ? /^bytes=(\d+)-(\d+)$/.exec(range) : null;
      if (!m || opts.ignoreRange) {
        return withUrl(new Response(bytes, { status: 200, headers: { "content-length": String(bytes.length) } }), url);
      }
      const start = Number(m[1]);
      const end = Math.min(Number(m[2]), bytes.length - 1);
      if (start >= bytes.length) return withUrl(new Response(null, { status: 416 }), url);
      const headers: Record<string, string> = {};
      if (opts.exposeContentRange !== false) headers["content-range"] = `bytes ${start}-${end}/${bytes.length}`;
      return withUrl(new Response(bytes.slice(start, end + 1), { status: 206, headers }), "https://cas-bridge.xethub.hf.co/obj");
    }, log);
    return { f, log };
  }

  it("parses a remote header with Range requests", async () => {
    const { f, log } = rangeServer();
    const src = hfRangeSource(ref, "model.gguf", { fetch: f });
    const r = await readGgufHeader(src);
    expect(r.header.metadata.get("general.architecture")).toBe("llama");
    expect(r.fileSize).toBe(bytes.length);
    expect(log[0]!.url).toBe("https://huggingface.co/org/repo/resolve/main/model.gguf");
    expect(new Headers(log[0]!.init.headers).get("range")).toBe(`bytes=0-${4 * 1024 * 1024 - 1}`);
  });

  it("works when Content-Range is not exposed (size unknown)", async () => {
    const { f } = rangeServer({ exposeContentRange: false });
    const r = await readGgufHeader(hfRangeSource(ref, "model.gguf", { fetch: f }));
    expect(r.header.tensors.length).toBeGreaterThan(0);
  });

  it("caps the body when a server ignores Range on the first chunk", async () => {
    const { f } = rangeServer({ ignoreRange: true });
    const src = hfRangeSource(ref, "model.gguf", { fetch: f });
    const chunk = await src.read(0, 1024);
    expect(chunk.length).toBe(1024);
    await expect(src.read(4096, 1024)).rejects.toBeInstanceOf(ParseError);
  });

  it("rejects a wrong byte range", async () => {
    const f = mockFetch((u) =>
      withUrl(new Response(new Uint8Array(10), { status: 206, headers: { "content-range": "bytes 5-14/100" } }), u),
    );
    await expect(hfRangeSource(ref, "m.gguf", { fetch: f }).read(0, 10)).rejects.toBeInstanceOf(ParseError);
  });

  it("maps auth errors to a helpful message", async () => {
    const f = mockFetch((u) => withUrl(new Response("no", { status: 401 }), u));
    await expect(hfRangeSource(ref, "m.gguf", { fetch: f }).read(0, 10)).rejects.toThrow(/token/);
  });

  it("refuses invalid paths when building URLs", () => {
    expect(() => buildResolveUrl(ref, "../x.gguf")).toThrow(ValidationError);
    expect(() => buildResolveUrl({ ...ref, org: "a/b" }, "x.gguf")).toThrow(ValidationError);
  });
});

describe("tree listing", () => {
  it("validates hostile listing entries", () => {
    const files = parseTreeListing([
      { type: "file", path: "good-Q4_K_M.gguf", size: 100, lfs: { size: 5_000_000_000 } },
      { type: "file", path: "../evil.gguf", size: 1 },
      { type: "file", path: "x<script>.gguf", size: 1 },
      { type: "file", path: "readme.md", size: 1 },
      { type: "directory", path: "Q8_0" },
      { type: "file", path: "Q8_0/m-Q8_0-00001-of-00002.gguf", size: 10 },
      { type: "file", path: "Q8_0/m-Q8_0-00002-of-00002.gguf", size: 11 },
      { type: "file", path: "neg.gguf", size: -5 },
      null,
      "string",
      { __proto__: { type: "file" }, path: "proto.gguf" },
    ]);
    expect(files.map((f) => f.path)).toEqual([
      "good-Q4_K_M.gguf",
      "Q8_0/m-Q8_0-00001-of-00002.gguf",
      "Q8_0/m-Q8_0-00002-of-00002.gguf",
      "neg.gguf",
    ]);
    expect(files[0]!.size).toBe(5_000_000_000);
    expect(files[3]!.size).toBe(0);
    expect(() => parseTreeListing({ files: [] })).toThrow(ParseError);
  });

  it("groups shards and matches quants like llama.cpp -hf", () => {
    const groups = groupGgufFiles([
      { path: "m-Q4_K_M.gguf", size: 5 },
      { path: "m-Q8_0-00001-of-00002.gguf", size: 5 },
      { path: "m-Q8_0-00002-of-00002.gguf", size: 5 },
      { path: "m-F16-00001-of-00003.gguf", size: 5 },
      { path: "mmproj-F16.gguf", size: 1 },
    ]);
    const q8 = groups.find((g) => g.label === "m-Q8_0")!;
    expect(q8.paths).toHaveLength(2);
    expect(q8.complete).toBe(true);
    expect(groups.find((g) => g.label === "m-F16")!.complete).toBe(false);
    expect(groups.find((g) => g.label === "mmproj-F16")!.isProjector).toBe(true);
    expect(matchQuant(groups, "q4_k_m")).toHaveLength(1);
    expect(matchQuant(groups, "F16")).toHaveLength(0); // incomplete + projector excluded
    expect(quantFromFileName("Qwen3-8B-UD-Q4_K_XL.gguf")).toBe("UD-Q4_K_XL");
    expect(quantFromFileName("dir/model.Q5_K_S.gguf")).toBe("Q5_K_S");
  });

  it("derives every shard path from one shard URL", () => {
    expect(shardPaths("d/m-00002-of-00003.gguf")).toEqual([
      "d/m-00001-of-00003.gguf",
      "d/m-00002-of-00003.gguf",
      "d/m-00003-of-00003.gguf",
    ]);
    expect(shardPaths("m.gguf")).toEqual(["m.gguf"]);
  });

  it("caps oversized listings", async () => {
    const huge = new Uint8Array(9 * 1024 * 1024).fill(0x20);
    const f = mockFetch((u) => withUrl(new Response(huge, { status: 200 }), u));
    await expect(hfListGgufFiles(ref, { fetch: f })).rejects.toBeInstanceOf(ParseError);
  });

  it("rejects invalid JSON", async () => {
    const f = mockFetch((u) => withUrl(new Response("{not json", { status: 200 }), u));
    await expect(hfListGgufFiles(ref, { fetch: f })).rejects.toBeInstanceOf(ParseError);
  });
});
