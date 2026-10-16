# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report privately through GitHub's private vulnerability reporting:
**Security → Advisories → Report a vulnerability** on this repository
(`/security/advisories/new`). You'll get an acknowledgement within a few days,
and we'll coordinate a fix and disclosure date with you.

Useful things to include: affected version or commit, a minimal reproducing
input (e.g. a crafted `.gguf` header or URL), and what you observed.

## Scope

In scope:

- The web app (`apps/web`) and the planner/parser package (`packages/core`).
- Malformed or malicious GGUF headers that hang, crash, or escape the parser's
  limits, or that get rendered unsanitized.
- Anything that causes the Hugging Face token to leave memory, reach a host
  other than `huggingface.co`, or appear in URLs, storage, logs or errors.
- Shell injection through the generated `llama-server` command.
- CSP / Trusted Types bypasses.

Out of scope: accuracy of VRAM estimates (please open a normal issue), and
vulnerabilities in llama.cpp itself.

## Design notes

- Everything runs client-side; nothing is uploaded. Only header bytes are read.
- Parsing runs in a Web Worker with a 15 s timeout and hard caps on header
  size, counts, string/array lengths, dimensions and total model size.
- The page ships a strict CSP (`default-src 'none'`, `require-trusted-types-for 'script'`)
  as a meta tag, since GitHub Pages can't set headers.
- No `innerHTML`/`eval`-family APIs; enforced by ESLint and by a test over the built bundle.
