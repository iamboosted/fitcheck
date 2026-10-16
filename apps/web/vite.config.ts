import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

/**
 * GitHub Pages can't send response headers, so the policy ships as a meta tag.
 * Only added to production builds: Vite's dev server injects inline styles.
 */
export const CSP = [
  "default-src 'none'",
  "script-src 'self'",
  "style-src 'self'",
  "img-src 'self' data:",
  "font-src 'self'",
  "connect-src https://huggingface.co https://*.huggingface.co https://*.hf.co",
  "worker-src 'self'",
  "base-uri 'none'",
  "form-action 'none'",
  "object-src 'none'",
  "require-trusted-types-for 'script'",
].join("; ");

function cspMeta(): Plugin {
  return {
    name: "fitcheck-csp-meta",
    apply: "build",
    transformIndexHtml: {
      order: "post",
      handler() {
        return [
          {
            tag: "meta",
            attrs: { "http-equiv": "Content-Security-Policy", content: CSP },
            injectTo: "head-prepend",
          },
        ];
      },
    },
  };
}

// Project page lives at https://<user>.github.io/<repo>/ ; override for a rename or custom domain.
const base = process.env["FITCHECK_BASE"] ?? "/fitcheck/";

export default defineConfig({
  base,
  plugins: [cspMeta()],
  resolve: {
    alias: {
      "@fitcheck/core": fileURLToPath(new URL("../../packages/core/src/index.ts", import.meta.url)),
    },
  },
  worker: { format: "es" },
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    // Never inline assets as data:/blob: URLs; everything is a same-origin file.
    assetsInlineLimit: 0,
    modulePreload: { polyfill: false },
  },
});
