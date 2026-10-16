/** Strict allowlists for every externally supplied identifier. */

export const ORG_REPO_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
export const FILENAME_RE = /^[A-Za-z0-9._-]{1,200}\.gguf$/;
export const QUANT_RE = /^[A-Za-z0-9_]{1,32}$/;
/** Directory segment inside a repo (subfolder of quants). */
export const DIR_SEGMENT_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$/;
export const REVISION_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
export const MAX_REPO_PATH_DEPTH = 4;

export function isValidOrgOrRepo(s: unknown): s is string {
  return typeof s === "string" && ORG_REPO_RE.test(s) && !s.includes("..");
}

export function isValidFileName(s: unknown): s is string {
  return typeof s === "string" && FILENAME_RE.test(s) && !s.startsWith(".");
}

export function isValidQuant(s: unknown): s is string {
  return typeof s === "string" && QUANT_RE.test(s);
}

export function isValidRevision(s: unknown): s is string {
  return typeof s === "string" && REVISION_RE.test(s) && !s.includes("..");
}

/** A path inside a repo: optional validated folders + a validated .gguf name. */
export function isValidRepoPath(s: unknown): s is string {
  if (typeof s !== "string" || s.length > 600) return false;
  const parts = s.split("/");
  if (parts.length > MAX_REPO_PATH_DEPTH + 1) return false;
  const file = parts.pop();
  if (!isValidFileName(file)) return false;
  return parts.every((p) => DIR_SEGMENT_RE.test(p) && !p.includes(".."));
}

export const CACHE_TYPES = ["f16", "q8_0", "q4_0"] as const;
export type CacheType = (typeof CACHE_TYPES)[number];

export function isCacheType(s: unknown): s is CacheType {
  return typeof s === "string" && (CACHE_TYPES as readonly string[]).includes(s);
}

export const BOUNDS = Object.freeze({
  gpuGbMin: 1,
  gpuGbMax: 512,
  reservedGbMax: 64,
  headroomGbMax: 32,
  gpuCountMin: 1,
  gpuCountMax: 16,
  ctxMin: 512,
  ctxMax: 1_048_576,
});

const DECIMAL_RE = /^\d{1,7}(?:\.\d{1,3})?$/;
const INT_RE = /^\d{1,8}$/;

/** Parse a decimal string strictly, returning null when invalid or out of range. */
export function parseDecimal(s: string | null | undefined, min: number, max: number): number | null {
  if (typeof s !== "string" || !DECIMAL_RE.test(s)) return null;
  const v = Number(s);
  return Number.isFinite(v) && v >= min && v <= max ? v : null;
}

export function parseInteger(s: string | null | undefined, min: number, max: number): number | null {
  if (typeof s !== "string" || !INT_RE.test(s)) return null;
  const v = Number(s);
  return Number.isSafeInteger(v) && v >= min && v <= max ? v : null;
}

export function clampNumber(v: unknown, min: number, max: number, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? Math.min(max, Math.max(min, v)) : fallback;
}

/**
 * Characters never allowed in a local file name that ends up in a shell
 * command: controls, newlines, every kind of quote, backtick and shell
 * metacharacters. Unicode quotes are included because PowerShell treats
 * them as quote characters.
 */
const SHELL_UNSAFE_RE = /[\u0000-\u001F\u007F-\u009F"'`$;|&<>\u2018\u2019\u201A\u201B\u201C\u201D\u201E\u201F\u2028\u2029\u202A-\u202E\u2066-\u2069\u200B-\u200F\uFEFF]/;

export function isShellSafeFileName(s: unknown): s is string {
  return (
    typeof s === "string" &&
    s.length >= 1 &&
    s.length <= 255 &&
    !SHELL_UNSAFE_RE.test(s) &&
    !s.includes("/") &&
    !s.includes("\\") &&
    s.trim() === s &&
    !s.startsWith("-")
  );
}
