# fitcheck

Plan **llama.cpp multi-GPU splits** from a GGUF header and get a working
`llama-server` command, entirely in your browser.

**Live:** https://iamboosted.github.io/fitcheck/ _(placeholder)_

- Drop a local `.gguf` (or every shard of one). Only the header is read, via
  `File.slice()`. The weights are never loaded and nothing is uploaded.
- Or paste a Hugging Face model: a `blob`/`resolve` URL, `org/repo:QUANT`, or
  bare `org/repo` to pick from its GGUF files. Headers are fetched with HTTP
  Range requests.
- Pick your GPUs (presets or custom size, with VRAM reserved for the desktop),
  context size and KV cache type.
- See a per-GPU bar of weights / KV cache / headroom against capacity, the
  max context that fits, and a copy-ready command for bash and PowerShell.
- Settings live in the URL, so a result is shareable (local file names are
  never put in the URL).

## Screenshots

_Add screenshots here._

## How the math works

**Weights.** Each tensor's size is computed from its dimensions and its ggml
type's block layout (e.g. Q4_K = 144 bytes per 256 weights). Bytes are grouped
by repeating layer (`blk.N.*`) plus the non-layer tensors (`token_embd`,
`output`, `output_norm`). If `output.weight` is absent, the embeddings are
tied and the output layer is counted as a copy of `token_embd`.

**KV cache.** Per layer:

```
kv_bytes = ctx × n_kv_heads × (key_length × bytes(K) + value_length × bytes(V))
```

With equal K/V sizes and types this is the familiar
`2 × n_layers × n_kv_heads × head_dim × ctx × bytes_per_elem`.
`f16` = 2 bytes, `q8_0` = 34/32, `q4_0` = 18/32. Per-layer `head_count_kv`
arrays are honored. MLA models (DeepSeek-style) use an approximation. A
quantized V cache requires flash attention, so `-fa on` is added automatically.

**Where things go.** This mirrors llama.cpp's layer split mode:

- `-ngl N` offloads the *last* N repeating layers. `N = n_layers + 1` also
  offloads the output layer, which goes on the last GPU.
- `token_embd` always stays in system RAM.
- Each layer's KV cache lives on the GPU holding that layer.
- Each GPU's usable memory is `VRAM − reserved − compute headroom` (headroom
  defaults to 1 GB per GPU).
- Layers are split in device order, weighted by usable VRAM. If that overflows
  a card, a greedy in-order fill is tried instead.
- `-ts` is emitted as layer counts per GPU (the output layer counts on the
  last card), so llama.cpp reproduces the planned split.

**When it doesn't fit,** the planner searches down for the largest `-ngl`
that fits, and reports how many bytes of weights and KV cache spill to system
RAM.

**Max context** is a binary search (in steps of 256) for the largest context
at which the whole model still fits on the GPUs.

These are estimates. Compute buffers vary by backend, batch size and
architecture, so keep the headroom honest. List GPUs in the same order as
`llama-server --list-devices`.

## Repo layout

```
packages/core   Pure TS: GGUF parser, planner, HF client, validation, shell quoting (no DOM)
apps/web        Vite + vanilla TS UI, parser Web Worker
```

`@fitcheck/core` has zero runtime dependencies and builds to `dist/` with
`npm run build -w @fitcheck/core`, ready to publish for reuse elsewhere.
`@huggingface/gguf` was considered, but it's built around fetch/Node paths
rather than an abstract `read(offset, length)`. Our parser serves `Blob.slice`
and Range requests through one code path, with hard limits on hostile input.

## Development

```sh
npm ci
npm run dev          # http://localhost:5173/fitcheck/
npm run build        # lint + typecheck + build (core and web)
npm test             # unit, fuzz, and built-output checks (run build first)
npm run test:live    # optional: real Range/CORS check against Hugging Face
```

The base path defaults to `/fitcheck/`. Set `FITCHECK_BASE` to change it. CI
sets it from the repository name automatically.

## Security

Every input is treated as hostile. See [SECURITY.md](SECURITY.md) for how to
report issues. Highlights:

- **Parser limits.** Bounds-checked reads and BigInt for 64-bit fields. Caps:
  64 MB header, 100k tensors, 10k KV pairs, 1 MB strings, 1M-element arrays,
  nesting depth 2, 1–4 dims, each dim ≤ 2^40, 2 TB total, and 32 reads per file.
  Parsing runs in a worker that is killed after 15 s. Metadata lives in a `Map`,
  and `__proto__`/`constructor`/`prototype` keys are rejected.
- **Rendering.** DOM APIs and `textContent` only; ESLint bans the HTML sinks
  and `eval`. Untrusted strings are stripped of control, bidi and zero-width
  characters. The chat template and tokenizer contents are never rendered.
- **Network.** URLs are rebuilt from allowlisted parts. Only `huggingface.co`
  is fetched, and redirects are only accepted to `*.huggingface.co` / `*.hf.co`.
  Requests use `credentials: "omit"` and `no-referrer`.
- **HF token.** Kept in memory only, sent only to `huggingface.co`, and cleared
  on page unload.
- **Generated command.** Built from validated values only, with per-shell quoting.
- **Page hardening.** Strict CSP with Trusted Types, and no third-party anything.
- **CI.** Actions are SHA-pinned, permissions minimal, `npm ci`, `npm audit`,
  plus CodeQL and Dependabot.

## License

Copyright © 2026 iAmBoosted. Released under the [MIT License](LICENSE).
