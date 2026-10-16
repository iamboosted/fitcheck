import type { GgufFile, HfRepoRef, ParseErrorCode } from "@fitcheck/core";

export type WorkerRequest =
  | { readonly kind: "blob"; readonly file: Blob }
  | { readonly kind: "hf"; readonly ref: HfRepoRef; readonly path: string; readonly token?: string };

export type WorkerResponse =
  | { readonly ok: true; readonly file: GgufFile }
  | { readonly ok: false; readonly code: ParseErrorCode; readonly message: string };
