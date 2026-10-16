/** Hard caps applied while parsing untrusted GGUF headers. */
export const LIMITS = Object.freeze({
  /** Largest header region (metadata + tensor infos) we will ever read. */
  maxHeaderBytes: 64 * 1024 * 1024,
  /** First chunk size; later chunks at least double the buffer. */
  initialChunkBytes: 4 * 1024 * 1024,
  /** Maximum number of read() calls per file. */
  maxFetches: 32,
  maxTensorCount: 100_000,
  maxKvCount: 10_000,
  maxStringBytes: 1024 * 1024,
  maxKeyBytes: 64 * 1024,
  maxArrayLength: 1_000_000,
  /** Top-level array = depth 1, array of arrays = depth 2. */
  maxArrayDepth: 2,
  minDims: 1,
  maxDims: 4,
  maxDim: 2n ** 40n,
  /** Total computed tensor bytes across a model (all shards). */
  maxModelBytes: 2n * 1024n ** 4n,
  maxAlignment: 1024 * 1024,
  /** Shards per model. */
  maxShards: 256,
  /** Arrays longer than this keep only a summary (numeric arrays). */
  keepNumericArrayMax: 4096,
  /** String arrays longer than this keep only a summary. */
  keepStringArrayMax: 64,
});

export const PARSE_TIMEOUT_MS = 15_000;
