/**
 * Live check against a small public GGUF on Hugging Face. Skipped unless
 * FITCHECK_LIVE=1 (npm run test:live), since CI and sandboxes may be offline.
 */
import { describe, expect, it } from "vitest";
import { groupGgufFiles, hfListGgufFiles, hfRangeSource, readGgufHeader, summarizeModel } from "../src/index.js";

const live = process.env["FITCHECK_LIVE"] === "1";
const ref = { org: "Qwen", repo: "Qwen2.5-0.5B-Instruct-GGUF", revision: "main" };
const ORIGIN = "https://example.github.io";

describe.skipIf(!live)("live Hugging Face", () => {
  it("lists files, Range-fetches only the header, and passes CORS for a browser origin", async () => {
    // Send an Origin header like a browser would, so CORS headers are returned.
    const withOrigin = (input: string, init: RequestInit) => {
      const headers = new Headers(init.headers);
      headers.set("Origin", ORIGIN);
      return fetch(input, { ...init, headers });
    };
    const groups = groupGgufFiles(await hfListGgufFiles(ref, { fetch: withOrigin })).filter((g) => !g.isProjector);
    expect(groups.length).toBeGreaterThan(0);
    const smallest = groups[0]!;

    const seen: Response[] = [];
    const spy = async (input: string, init: RequestInit) => {
      const res = await withOrigin(input, init);
      seen.push(res.clone());
      return res;
    };
    const file = await readGgufHeader(hfRangeSource(ref, smallest.paths[0]!, { fetch: spy }));
    const info = summarizeModel([file]);
    expect(info.architecture).toBe("qwen2");
    expect(info.nLayers).toBeGreaterThan(0);
    expect(file.bytesRead).toBeLessThan(smallest.totalSize);

    const first = seen[0]!;
    expect(first.status).toBe(206);
    const acao = first.headers.get("access-control-allow-origin");
    expect(acao === "*" || acao === ORIGIN, `ACAO was ${acao}`).toBe(true);
  });
});
