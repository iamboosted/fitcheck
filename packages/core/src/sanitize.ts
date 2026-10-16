/**
 * Display sanitization for strings that originate from untrusted input
 * (GGUF metadata, file names, HF responses). Output is meant for textContent
 * only; this is not an HTML escaper and must never be used as one.
 */

// C0/C1 controls and DEL.
const CONTROL = /[\u0000-\u001F\u007F-\u009F]/g;
// Bidi embeddings/overrides/isolates and marks.
const BIDI = /[\u202A-\u202E\u2066-\u2069\u200E\u200F\u061C]/g;
// Zero-width and invisible formatting characters, BOM, line/para separators.
const INVISIBLE = /[\u200B-\u200D\u2060-\u2064\u180E\uFEFF\u2028\u2029\u00AD]/g;
// Lone surrogates.
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g;

export const DEFAULT_DISPLAY_MAX = 120;

export function sanitizeText(input: unknown, maxLength = DEFAULT_DISPLAY_MAX): string {
  if (typeof input !== "string") return "";
  // Bound work before running regexes on hostile input.
  let s = input.length > maxLength * 4 + 64 ? input.slice(0, maxLength * 4 + 64) : input;
  s = s
    .replace(CONTROL, " ")
    .replace(BIDI, "")
    .replace(INVISIBLE, "")
    .replace(LONE_SURROGATE, "\uFFFD")
    .replace(/\s+/g, " ")
    .trim();
  const chars = Array.from(s);
  if (chars.length > maxLength) {
    return chars.slice(0, Math.max(0, maxLength - 1)).join("") + "\u2026";
  }
  return s;
}

/** Metadata keys whose values are never rendered, only summarized by size. */
export function isHiddenMetadataKey(key: string): boolean {
  return key.startsWith("tokenizer.") || key.includes("chat_template");
}
