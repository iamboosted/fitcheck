import { PARSE_TIMEOUT_MS, ParseError, type GgufFile, type ParseErrorCode } from "@fitcheck/core";
import workerUrl from "./parser.worker.ts?worker&url";
import type { WorkerRequest, WorkerResponse } from "./worker-protocol.js";

interface TrustedTypePolicyLike {
  createScriptURL(input: string): unknown;
}
interface TrustedTypesLike {
  createPolicy(name: string, rules: { createScriptURL: (input: string) => string }): TrustedTypePolicyLike;
}

/**
 * The CSP requires Trusted Types for script sinks, and `new Worker(url)` is
 * one. This policy only admits same-origin script URLs.
 */
let policy: TrustedTypePolicyLike | null | undefined;
function workerScriptUrl(): string {
  if (policy === undefined) {
    const tt = (globalThis as { trustedTypes?: TrustedTypesLike }).trustedTypes;
    policy = tt
      ? tt.createPolicy("fitcheck-worker", {
          createScriptURL(input) {
            const u = new URL(input, location.href);
            if (u.origin !== location.origin) throw new TypeError("Blocked cross-origin worker URL.");
            return u.href;
          },
        })
      : null;
  }
  return (policy ? policy.createScriptURL(workerUrl) : workerUrl) as string;
}

const CODES: ReadonlySet<string> = new Set<ParseErrorCode>([
  "BAD_MAGIC", "BAD_VERSION", "TRUNCATED", "LIMIT", "INVALID", "TOO_MANY_FETCHES", "IO", "TIMEOUT",
]);

/** Parse one header in a fresh worker; the worker is always terminated afterwards. */
export function parseInWorker(req: WorkerRequest, timeoutMs = PARSE_TIMEOUT_MS): Promise<GgufFile> {
  return new Promise<GgufFile>((resolve, reject) => {
    let worker: Worker;
    try {
      worker = new Worker(workerScriptUrl(), { type: "module", name: "gguf-parser" });
    } catch {
      reject(new ParseError("IO", "Could not start the parser worker."));
      return;
    }
    let settled = false;
    const finish = (fn: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      worker.terminate();
      fn();
    };
    const timer = setTimeout(
      () => finish(() => reject(new ParseError("TIMEOUT", "Parsing took too long and was stopped."))),
      timeoutMs,
    );
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const res = e.data;
      finish(() => {
        if (res && res.ok === true && res.file && res.file.header) resolve(res.file);
        else if (res && res.ok === false && CODES.has(res.code)) reject(new ParseError(res.code, String(res.message)));
        else reject(new ParseError("INVALID", "Malformed GGUF header."));
      });
    };
    worker.onerror = (e) => {
      e.preventDefault();
      finish(() => reject(new ParseError("INVALID", "The parser crashed on this input.")));
    };
    worker.onmessageerror = () => finish(() => reject(new ParseError("INVALID", "Malformed GGUF header.")));
    worker.postMessage(req);
  });
}

/** Run jobs with bounded concurrency, failing fast. */
export async function parseAll(reqs: readonly WorkerRequest[], concurrency = 4): Promise<GgufFile[]> {
  const out = new Array<GgufFile>(reqs.length);
  let next = 0;
  const run = async () => {
    while (next < reqs.length) {
      const i = next++;
      out[i] = await parseInWorker(reqs[i] as WorkerRequest);
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, reqs.length) }, run));
  return out;
}
