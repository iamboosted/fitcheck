/**
 * Parses one GGUF header off the main thread. The main thread enforces a
 * timeout and terminates this worker when done, on error, or on timeout.
 */
import {
  ParseError,
  ValidationError,
  blobSource,
  hfRangeSource,
  isValidOrgOrRepo,
  isValidRepoPath,
  isValidRevision,
  readGgufHeader,
} from "@fitcheck/core";
import type { WorkerRequest, WorkerResponse } from "./worker-protocol.js";

interface WorkerScope {
  onmessage: ((e: MessageEvent<unknown>) => void) | null;
  postMessage(msg: WorkerResponse): void;
}
const scope = self as unknown as WorkerScope;

function fail(e: unknown): WorkerResponse {
  if (e instanceof ParseError) return { ok: false, code: e.code, message: e.message };
  if (e instanceof ValidationError) return { ok: false, code: "INVALID", message: e.message };
  return { ok: false, code: "INVALID", message: "Malformed GGUF header." };
}

async function handle(req: WorkerRequest): Promise<WorkerResponse> {
  if (req.kind === "blob") {
    if (!(req.file instanceof Blob)) throw new ValidationError("Invalid file.");
    const file = await readGgufHeader(blobSource(req.file));
    return { ok: true, file };
  }
  if (req.kind === "hf") {
    const { org, repo, revision } = req.ref;
    if (!isValidOrgOrRepo(org) || !isValidOrgOrRepo(repo) || !isValidRevision(revision) || !isValidRepoPath(req.path)) {
      throw new ValidationError("Invalid Hugging Face reference.");
    }
    const token = typeof req.token === "string" ? req.token : undefined;
    const file = await readGgufHeader(hfRangeSource({ org, repo, revision }, req.path, { token }));
    return { ok: true, file };
  }
  throw new ValidationError("Unknown request.");
}

scope.onmessage = (e) => {
  const req = e.data as WorkerRequest;
  handle(req).then(
    (res) => scope.postMessage(res),
    (err: unknown) => scope.postMessage(fail(err)),
  );
};
