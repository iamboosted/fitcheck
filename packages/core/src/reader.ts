import { ParseError } from "./errors.js";
import { NeedMoreData, parseGgufHeader, type GgufHeader } from "./gguf.js";
import { LIMITS } from "./limits.js";

/**
 * Abstract random-access byte source. `Blob.slice` and HTTP Range requests both
 * implement this, so local files and remote files share one parsing path.
 */
export interface ByteSource {
  /** Total size if known. May become known after the first read. */
  readonly size: number | null;
  /**
   * Read up to `length` bytes at `offset`. Returning fewer bytes is allowed;
   * returning zero bytes means end of file.
   */
  read(offset: number, length: number, signal?: AbortSignal): Promise<Uint8Array>;
}

export interface ReadHeaderOptions {
  initialChunkBytes?: number;
  maxFetches?: number;
  maxHeaderBytes?: number;
  signal?: AbortSignal;
}

export interface GgufFile {
  readonly header: GgufHeader;
  readonly fileSize: number | null;
  /** Bytes actually read from the source. */
  readonly bytesRead: number;
  readonly fetches: number;
}

/**
 * Read just enough of a GGUF file to parse its header: start with a 4 MB chunk
 * and grow (at least doubling) until metadata and tensor infos parse.
 */
export async function readGgufHeader(src: ByteSource, opts: ReadHeaderOptions = {}): Promise<GgufFile> {
  const maxHeader = Math.min(opts.maxHeaderBytes ?? LIMITS.maxHeaderBytes, LIMITS.maxHeaderBytes);
  const maxFetches = Math.min(opts.maxFetches ?? LIMITS.maxFetches, LIMITS.maxFetches);
  let want = Math.min(opts.initialChunkBytes ?? LIMITS.initialChunkBytes, maxHeader);

  let buf = new Uint8Array(0);
  let fetches = 0;
  let eof = false;

  for (;;) {
    opts.signal?.throwIfAborted();
    const size = src.size;
    const target = Math.min(want, maxHeader, size ?? Number.POSITIVE_INFINITY);

    if (target > buf.length && !eof) {
      if (fetches >= maxFetches) {
        throw new ParseError("TOO_MANY_FETCHES", "Gave up after too many read requests.");
      }
      fetches++;
      const requested = target - buf.length;
      let chunk = await src.read(buf.length, requested, opts.signal);
      if (!(chunk instanceof Uint8Array)) throw new ParseError("IO", "Read returned invalid data.");
      if (chunk.length > requested) chunk = chunk.subarray(0, requested);
      if (chunk.length === 0) {
        eof = true;
      } else {
        const next = new Uint8Array(buf.length + chunk.length);
        next.set(buf, 0);
        next.set(chunk, buf.length);
        buf = next;
      }
      // Short read: keep reading until the target is reached or EOF.
      if (!eof && buf.length < target) continue;
    }

    const knownSize = src.size;
    const complete = eof || (knownSize !== null && buf.length >= knownSize);
    try {
      const header = parseGgufHeader(buf, {
        complete: complete || buf.length >= maxHeader,
        fileSize: knownSize,
        maxHeaderBytes: maxHeader,
      });
      return { header, fileSize: knownSize, bytesRead: buf.length, fetches };
    } catch (e) {
      if (e instanceof NeedMoreData) {
        if (e.required > maxHeader) throw new ParseError("LIMIT", "GGUF header exceeds the 64 MB limit.");
        want = Math.min(maxHeader, Math.max(e.required, buf.length * 2));
        continue;
      }
      if (e instanceof ParseError) throw e;
      throw new ParseError("INVALID", "Malformed GGUF header.");
    }
  }
}

/** Byte source over a Blob/File; only the requested slice is ever read. */
export function blobSource(blob: Blob): ByteSource {
  return {
    size: blob.size,
    async read(offset, length) {
      if (offset >= blob.size) return new Uint8Array(0);
      const end = Math.min(blob.size, offset + length);
      return new Uint8Array(await blob.slice(offset, end).arrayBuffer());
    },
  };
}

/** Byte source over an in-memory buffer (tests, fixtures). */
export function bufferSource(bytes: Uint8Array): ByteSource {
  return {
    size: bytes.length,
    async read(offset, length) {
      return bytes.slice(offset, Math.min(bytes.length, offset + length));
    },
  };
}
