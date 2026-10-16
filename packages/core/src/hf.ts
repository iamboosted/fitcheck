import { ParseError, ValidationError } from "./errors.js";
import type { ByteSource } from "./reader.js";
import {
  isValidOrgOrRepo,
  isValidQuant,
  isValidRepoPath,
  isValidRevision,
} from "./validate.js";

export const HF_ORIGIN = "https://huggingface.co";
const HF_HOST = "huggingface.co";

export interface HfRepoRef {
  readonly org: string;
  readonly repo: string;
  readonly revision: string;
}

export type HfInput =
  | (HfRepoRef & { readonly kind: "file"; readonly path: string })
  | (HfRepoRef & { readonly kind: "quant"; readonly quant: string })
  | (HfRepoRef & { readonly kind: "repo" });

const MAX_INPUT_LENGTH = 2048;

function safeDecode(seg: string): string | null {
  try {
    return decodeURIComponent(seg);
  } catch {
    return null;
  }
}

/**
 * Accepts:
 *   https://huggingface.co/<org>/<repo>/(blob|resolve)/<rev>/<path>.gguf
 *   https://huggingface.co/<org>/<repo>[/tree/<rev>]
 *   <org>/<repo>:<QUANT>
 *   <org>/<repo>
 * Every part is validated; nothing from the raw input is used to fetch.
 */
export function parseHfInput(raw: string): HfInput {
  if (typeof raw !== "string") throw new ValidationError("Enter a Hugging Face URL or org/repo.");
  const input = raw.trim();
  if (input.length === 0 || input.length > MAX_INPUT_LENGTH) {
    throw new ValidationError("Enter a Hugging Face URL or org/repo.");
  }

  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(input)) return parseHfUrl(input);

  const m = /^([^/:\s]+)\/([^/:\s]+)(?::([^/:\s]+))?$/.exec(input);
  if (!m) throw new ValidationError("Use org/repo, org/repo:QUANT, or a huggingface.co URL.");
  const [, org, repo, quant] = m;
  if (!isValidOrgOrRepo(org) || !isValidOrgOrRepo(repo)) {
    throw new ValidationError("Invalid organization or repository name.");
  }
  if (quant !== undefined) {
    if (!isValidQuant(quant)) throw new ValidationError("Invalid quant tag.");
    return { kind: "quant", org, repo, revision: "main", quant };
  }
  return { kind: "repo", org, repo, revision: "main" };
}

function parseHfUrl(input: string): HfInput {
  let url: URL;
  try {
    url = new URL(input);
  } catch {
    throw new ValidationError("Not a valid URL.");
  }
  if (url.protocol !== "https:" || url.hostname !== HF_HOST || url.port !== "" || url.username || url.password) {
    throw new ValidationError("Only https://huggingface.co URLs are accepted.");
  }
  const segs = url.pathname.split("/").filter((s) => s.length > 0).map(safeDecode);
  // Decoded segments must not smuggle separators (%2F, %5C) or be empty.
  if (segs.some((s) => s === null || s.length === 0 || s.includes("/") || s.includes("\\"))) {
    throw new ValidationError("Malformed URL path.");
  }
  const parts = segs as string[];
  const [org, repo, mode, revision, ...rest] = parts;
  if (!isValidOrgOrRepo(org) || !isValidOrgOrRepo(repo)) {
    throw new ValidationError("Invalid organization or repository name.");
  }
  if (mode === undefined) return { kind: "repo", org, repo, revision: "main" };
  if (!isValidRevision(revision)) throw new ValidationError("Invalid revision.");
  if (mode === "tree" && rest.length === 0) return { kind: "repo", org, repo, revision };
  if (mode !== "blob" && mode !== "resolve") throw new ValidationError("Unsupported huggingface.co URL.");
  const path = rest.join("/");
  if (!isValidRepoPath(path)) throw new ValidationError("Invalid .gguf file path.");
  return { kind: "file", org, repo, revision, path };
}

/** Host allowlist for any response we accept (including after redirects). */
export function isAllowedHfHost(hostname: string): boolean {
  const h = hostname.toLowerCase();
  return h === HF_HOST || h.endsWith(".huggingface.co") || h.endsWith(".hf.co");
}

export function isAllowedHfUrl(url: URL): boolean {
  return url.protocol === "https:" && url.port === "" && !url.username && !url.password && isAllowedHfHost(url.hostname);
}

function assertRef(ref: HfRepoRef): void {
  if (!isValidOrgOrRepo(ref.org) || !isValidOrgOrRepo(ref.repo) || !isValidRevision(ref.revision)) {
    throw new ValidationError("Invalid repository reference.");
  }
}

/** Build the file URL from validated parts only. */
export function buildResolveUrl(ref: HfRepoRef, path: string): URL {
  assertRef(ref);
  if (!isValidRepoPath(path)) throw new ValidationError("Invalid file path.");
  const p = path.split("/").map(encodeURIComponent).join("/");
  return new URL(
    `${HF_ORIGIN}/${encodeURIComponent(ref.org)}/${encodeURIComponent(ref.repo)}/resolve/${encodeURIComponent(ref.revision)}/${p}`,
  );
}

export function buildTreeUrl(ref: HfRepoRef): URL {
  assertRef(ref);
  return new URL(
    `${HF_ORIGIN}/api/models/${encodeURIComponent(ref.org)}/${encodeURIComponent(ref.repo)}/tree/${encodeURIComponent(ref.revision)}?recursive=true`,
  );
}

export type FetchLike = (input: string, init: RequestInit) => Promise<Response>;

export interface HfFetchOptions {
  fetch?: FetchLike;
  /** Bearer token; only ever attached to requests for huggingface.co itself. */
  token?: string | undefined;
  signal?: AbortSignal | undefined;
  headers?: Record<string, string>;
}

/**
 * The only way this codebase talks to the network. Enforces the host
 * allowlist before the request and on the final (post-redirect) URL.
 */
export async function hfFetch(url: URL, opts: HfFetchOptions = {}): Promise<Response> {
  if (!isAllowedHfUrl(url)) throw new ParseError("IO", "Blocked request to a non-Hugging Face host.");
  const headers = new Headers(opts.headers);
  const token = opts.token?.trim();
  if (token && url.hostname === HF_HOST) {
    if (!/^[\x21-\x7E]{1,512}$/.test(token)) throw new ValidationError("The access token contains invalid characters.");
    headers.set("Authorization", `Bearer ${token}`);
  }
  const doFetch: FetchLike = opts.fetch ?? ((i, init) => fetch(i, init));
  let res: Response;
  try {
    res = await doFetch(url.href, {
      method: "GET",
      headers,
      credentials: "omit",
      referrerPolicy: "no-referrer",
      redirect: "follow",
      mode: "cors",
      cache: "no-store",
      signal: opts.signal ?? null,
    });
  } catch (e) {
    if (opts.signal?.aborted) throw new ParseError("TIMEOUT", "The request was cancelled or timed out.");
    throw new ParseError("IO", "Network request failed (offline, blocked, or CORS).");
  }
  const finalUrl = res.url ? safeUrl(res.url) : url;
  if (!finalUrl || !isAllowedHfUrl(finalUrl)) {
    await cancelBody(res);
    throw new ParseError("IO", "Redirected to a non-Hugging Face host; aborted.");
  }
  if (res.type === "opaqueredirect") {
    throw new ParseError("IO", "Unexpected redirect response.");
  }
  return res;
}

function safeUrl(s: string): URL | null {
  try {
    return new URL(s);
  } catch {
    return null;
  }
}

async function cancelBody(res: Response): Promise<void> {
  try {
    await res.body?.cancel();
  } catch {
    /* ignore */
  }
}

function statusError(status: number): ParseError {
  if (status === 401 || status === 403) {
    return new ParseError("IO", `Access denied (HTTP ${status}). Gated repo? Accept its terms and supply a token.`);
  }
  if (status === 404) return new ParseError("IO", "Not found (HTTP 404).");
  if (status === 429) return new ParseError("IO", "Rate limited by Hugging Face (HTTP 429).");
  return new ParseError("IO", `Unexpected HTTP status ${Number.isInteger(status) ? status : 0}.`);
}

/** Read at most `max` bytes of a body, cancelling the rest. */
export async function readBodyCapped(res: Response, max: number): Promise<Uint8Array> {
  if (!res.body) return new Uint8Array(0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (total < max) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!(value instanceof Uint8Array)) throw new ParseError("IO", "Invalid response body.");
      const take = Math.min(value.length, max - total);
      chunks.push(take === value.length ? value : value.subarray(0, take));
      total += take;
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      /* ignore */
    }
  }
  const out = new Uint8Array(total);
  let pos = 0;
  for (const c of chunks) {
    out.set(c, pos);
    pos += c.length;
  }
  return out;
}

const CONTENT_RANGE_RE = /^bytes (\d{1,16})-(\d{1,16})\/(\d{1,16}|\*)$/;

/** ByteSource over HTTP Range requests against a validated HF file URL. */
export function hfRangeSource(ref: HfRepoRef, path: string, opts: HfFetchOptions = {}): ByteSource & { readonly url: URL } {
  const url = buildResolveUrl(ref, path);
  let size: number | null = null;
  return {
    url,
    get size() {
      return size;
    },
    async read(offset, length, signal) {
      if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length <= 0) {
        throw new ParseError("IO", "Invalid range.");
      }
      const res = await hfFetch(url, {
        ...opts,
        signal: signal ?? opts.signal,
        headers: { Range: `bytes=${offset}-${offset + length - 1}` },
      });
      if (res.status === 416) {
        await cancelBody(res);
        return new Uint8Array(0);
      }
      if (res.status === 206) {
        const cr = res.headers.get("content-range");
        const m = cr ? CONTENT_RANGE_RE.exec(cr) : null;
        if (m) {
          if (Number(m[1]) !== offset) {
            await cancelBody(res);
            throw new ParseError("IO", "Server returned the wrong byte range.");
          }
          if (m[3] !== "*") size = Number(m[3]);
        }
        return readBodyCapped(res, length);
      }
      if (res.status === 200) {
        // Server ignored Range. Only usable for the first chunk; never read past it.
        if (offset !== 0) {
          await cancelBody(res);
          throw new ParseError("IO", "Server does not support Range requests.");
        }
        const cl = Number(res.headers.get("content-length"));
        if (Number.isSafeInteger(cl) && cl > 0) size = cl;
        return readBodyCapped(res, length);
      }
      await cancelBody(res);
      throw statusError(res.status);
    },
  };
}

export interface HfFileEntry {
  readonly path: string;
  readonly size: number;
}

const MAX_TREE_BYTES = 8 * 1024 * 1024;
const MAX_TREE_ENTRIES = 20_000;

/** List .gguf files in a repo via the HF tree API. The response is untrusted. */
export async function hfListGgufFiles(ref: HfRepoRef, opts: HfFetchOptions = {}): Promise<HfFileEntry[]> {
  const res = await hfFetch(buildTreeUrl(ref), { ...opts, headers: { Accept: "application/json" } });
  if (res.status !== 200) {
    await cancelBody(res);
    throw statusError(res.status);
  }
  const bytes = await readBodyCapped(res, MAX_TREE_BYTES + 1);
  if (bytes.length > MAX_TREE_BYTES) throw new ParseError("LIMIT", "Repository listing is too large.");
  let data: unknown;
  try {
    data = JSON.parse(new TextDecoder("utf-8", { fatal: false }).decode(bytes));
  } catch {
    throw new ParseError("INVALID", "Repository listing is not valid JSON.");
  }
  return parseTreeListing(data);
}

/** Read an own property only; never trust anything inherited. */
function own(obj: unknown, key: string): unknown {
  return typeof obj === "object" && obj !== null && Object.hasOwn(obj, key) ? (obj as Record<string, unknown>)[key] : undefined;
}

export function parseTreeListing(data: unknown): HfFileEntry[] {
  if (!Array.isArray(data)) throw new ParseError("INVALID", "Unexpected repository listing format.");
  const out: HfFileEntry[] = [];
  for (const item of data.slice(0, MAX_TREE_ENTRIES)) {
    if (typeof item !== "object" || item === null) continue;
    if (own(item, "type") !== "file") continue;
    const path = own(item, "path");
    if (!isValidRepoPath(path)) continue;
    const lfsSize = own(own(item, "lfs"), "size");
    const rawSize = typeof lfsSize === "number" ? lfsSize : own(item, "size");
    const size = typeof rawSize === "number" && Number.isSafeInteger(rawSize) && rawSize >= 0 ? rawSize : 0;
    out.push({ path, size });
  }
  return out;
}

const SHARD_RE = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/;

export interface ShardInfo {
  readonly base: string;
  readonly index: number;
  readonly count: number;
}

export function shardInfo(path: string): ShardInfo | null {
  const m = SHARD_RE.exec(path);
  if (!m) return null;
  const index = Number(m[2]);
  const count = Number(m[3]);
  if (index < 1 || count < 1 || index > count || count > 256) return null;
  return { base: m[1] as string, index, count };
}

/** All shard paths for a sharded file path (validated), in order. */
export function shardPaths(path: string): string[] {
  const s = shardInfo(path);
  if (!s) return [path];
  const pad = (n: number) => String(n).padStart(5, "0");
  const paths: string[] = [];
  for (let i = 1; i <= s.count; i++) {
    const p = `${s.base}-${pad(i)}-of-${pad(s.count)}.gguf`;
    if (!isValidRepoPath(p)) throw new ValidationError("Invalid shard path.");
    paths.push(p);
  }
  return paths;
}

export interface GgufGroup {
  /** Display label: path without the shard suffix. */
  readonly label: string;
  /** Ordered file paths (one, or every shard). */
  readonly paths: readonly string[];
  readonly totalSize: number;
  readonly complete: boolean;
  readonly isProjector: boolean;
  /** Quant tag guessed from the file name, if it matches QUANT_RE. */
  readonly quant: string | null;
}

const QUANT_IN_NAME_RE =
  /(?:^|[-_.])((?:UD-)?(?:IQ\d_[A-Z]{1,4}|Q\d_K(?:_[A-Z]{1,2})?|Q\d_\d|BF16|F16|F32|MXFP4(?:_MOE)?|TQ\d_\d))(?=[-_.]|$)/i;

export function quantFromFileName(path: string): string | null {
  const name = path.split("/").pop() ?? "";
  const m = QUANT_IN_NAME_RE.exec(name.replace(/-\d{5}-of-\d{5}\.gguf$|\.gguf$/i, ""));
  const q = m?.[1]?.toUpperCase() ?? null;
  return q;
}

export function groupGgufFiles(files: readonly HfFileEntry[]): GgufGroup[] {
  const groups = new Map<string, { label: string; count: number; byIndex: Map<number, HfFileEntry> }>();
  const singles: GgufGroup[] = [];
  for (const f of files) {
    const s = shardInfo(f.path);
    if (!s) {
      singles.push(makeGroup(f.path.replace(/\.gguf$/, ""), [f.path], f.size, true));
      continue;
    }
    const key = `${s.base}\u0000${s.count}`;
    let g = groups.get(key);
    if (!g) {
      g = { label: s.base, count: s.count, byIndex: new Map() };
      groups.set(key, g);
    }
    g.byIndex.set(s.index, f);
  }
  const sharded = [...groups.values()].map((g) => {
    const entries = [...g.byIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, e]) => e);
    const total = entries.reduce((n, e) => n + e.size, 0);
    return makeGroup(g.label, entries.map((e) => e.path), total, entries.length === g.count);
  });
  return [...singles, ...sharded].sort((a, b) => Number(a.isProjector) - Number(b.isProjector) || a.totalSize - b.totalSize);
}

function makeGroup(label: string, paths: string[], totalSize: number, complete: boolean): GgufGroup {
  const name = (label.split("/").pop() ?? "").toLowerCase();
  const quant = quantFromFileName(paths[0] ?? "");
  return {
    label,
    paths,
    totalSize,
    complete,
    isProjector: name.includes("mmproj"),
    quant: quant !== null && isValidQuant(quant) ? quant : null,
  };
}

/** Groups whose file name contains the quant tag (how llama.cpp's -hf picks). */
export function matchQuant(groups: readonly GgufGroup[], quant: string): GgufGroup[] {
  const q = quant.toLowerCase();
  return groups.filter((g) => !g.isProjector && g.complete && (g.paths[0] ?? "").toLowerCase().includes(q));
}
