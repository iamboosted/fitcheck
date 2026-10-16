/**
 * Checks the production bundle, not the sources. Run `npm run build` first
 * (CI does). Fails loudly if dist is missing so it can't silently pass.
 */
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { CSP } from "../vite.config.js";

const dist = fileURLToPath(new URL("../dist", import.meta.url));

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

describe("built output", () => {
  it("exists (run npm run build first)", () => {
    expect(existsSync(join(dist, "index.html")), "apps/web/dist is missing; run `npm run build`").toBe(true);
  });

  const files = existsSync(dist) ? walk(dist) : [];
  const text = files.filter((f) => /\.(js|html|css|svg)$/.test(f));

  it("contains no HTML-string sinks or dynamic code evaluation", () => {
    const banned = [
      /\.innerHTML\b/, /\.outerHTML\b/, /insertAdjacentHTML/, /document\.write/, /\beval\s*\(/,
      /new\s+Function\s*\(/, /createContextualFragment/, /\bsrcdoc\b/, /\bsetTimeout\(\s*["'`]/,
      /\blocalStorage\b/, /\bsessionStorage\b/, /\bindexedDB\b/,
    ];
    expect(text.length).toBeGreaterThan(0);
    for (const f of text) {
      const src = readFileSync(f, "utf8");
      for (const re of banned) expect(re.test(src), `${re} found in ${f}`).toBe(false);
    }
  });

  it("ships the CSP and referrer meta tags before any script", () => {
    const html = readFileSync(join(dist, "index.html"), "utf8");
    const decoded = html.replace(/&#39;/g, "'").replace(/&quot;/g, '"');
    const cspIdx = decoded.indexOf(`<meta http-equiv="Content-Security-Policy" content="${CSP}">`);
    expect(cspIdx).toBeGreaterThan(-1);
    expect(cspIdx).toBeLessThan(decoded.indexOf("<script"));
    expect(CSP).toContain("default-src 'none'");
    expect(CSP).toContain("require-trusted-types-for 'script'");
    expect(CSP).not.toMatch(/unsafe-(inline|eval)/);
    expect(html).toContain('<meta name="referrer" content="no-referrer"');
  });

  it("has no inline scripts, inline styles, or third-party resources", () => {
    const html = readFileSync(join(dist, "index.html"), "utf8");
    expect(html).not.toMatch(/<script(?![^>]*\bsrc=)[^>]*>/);
    expect(html).not.toMatch(/<style\b/);
    expect(html).not.toMatch(/\sstyle=/);
    expect(html).not.toMatch(/\son[a-z]+=/i);
    for (const m of html.matchAll(/\b(?:src|href)="([^"]+)"/g)) {
      expect(m[1]!.startsWith("http"), `external resource ${m[1]}`).toBe(false);
    }
    for (const f of text) {
      const src = readFileSync(f, "utf8");
      // Only huggingface.co may appear as a network origin in the bundle (github.com as a link).
      for (const m of src.matchAll(/https:\/\/([a-z0-9.-]+)/gi)) {
        const host = m[1]!.toLowerCase();
        // github.com appears only as a plain link (navigation), never fetched; CSP connect-src stays HF-only.
        const ok =
          host === "huggingface.co" || host.endsWith(".huggingface.co") || host.endsWith(".hf.co") ||
          host === "www.w3.org" || host === "github.com";
        expect(ok, `unexpected origin ${host} in ${f}`).toBe(true);
      }
    }
  });

  it("emits no data: or blob: workers and no source maps", () => {
    for (const f of files) expect(f.endsWith(".map"), f).toBe(false);
    for (const f of text.filter((x) => x.endsWith(".js"))) {
      const src = readFileSync(f, "utf8");
      expect(src).not.toMatch(/new Worker\(\s*["'`](?:data|blob):/);
    }
  });
});
