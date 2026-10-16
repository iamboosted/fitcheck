import { ValidationError } from "./errors.js";
import type { Plan } from "./plan.js";
import {
  isCacheType,
  isShellSafeFileName,
  isValidOrgOrRepo,
  isValidQuant,
  isValidRepoPath,
  type CacheType,
} from "./validate.js";

/** POSIX shell single-quoting: ' becomes '\'' . */
export function bashQuote(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`;
}

/**
 * PowerShell single-quoted string: every quote character (ASCII and the
 * Unicode single quotes PowerShell also honors) is doubled.
 */
export function psQuote(s: string): string {
  return `'${s.replace(/['\u2018\u2019\u201A\u201B]/g, (q) => q + q)}'`;
}

export type ModelSource =
  | { readonly kind: "local"; readonly fileName: string }
  | { readonly kind: "hf-quant"; readonly org: string; readonly repo: string; readonly quant: string }
  | { readonly kind: "hf-file"; readonly org: string; readonly repo: string; readonly path: string };

export interface CommandInput {
  readonly source: ModelSource;
  readonly plan: Plan;
  readonly ctx: number;
  readonly cacheK: CacheType;
  readonly cacheV: CacheType;
}

export interface Commands {
  readonly bash: string;
  readonly powershell: string;
}

type Arg = { readonly literal: string } | { readonly value: string };

function int(n: number, min: number, max: number, what: string): string {
  if (!Number.isSafeInteger(n) || n < min || n > max) throw new ValidationError(`Invalid ${what}.`);
  return String(n);
}

/** Build llama-server argv. Only validated values are interpolated. */
export function buildArgs(input: CommandInput): Arg[] {
  const { source, plan, ctx, cacheK, cacheV } = input;
  const args: Arg[] = [{ literal: "llama-server" }];

  switch (source.kind) {
    case "local":
      if (!isShellSafeFileName(source.fileName)) {
        throw new ValidationError("File name contains characters that are unsafe in a shell command; rename the file.");
      }
      args.push({ literal: "-m" }, { value: source.fileName });
      break;
    case "hf-quant":
      if (!isValidOrgOrRepo(source.org) || !isValidOrgOrRepo(source.repo) || !isValidQuant(source.quant)) {
        throw new ValidationError("Invalid Hugging Face reference.");
      }
      args.push({ literal: "-hf" }, { value: `${source.org}/${source.repo}:${source.quant}` });
      break;
    case "hf-file":
      if (!isValidOrgOrRepo(source.org) || !isValidOrgOrRepo(source.repo) || !isValidRepoPath(source.path)) {
        throw new ValidationError("Invalid Hugging Face reference.");
      }
      args.push(
        { literal: "--hf-repo" },
        { value: `${source.org}/${source.repo}` },
        { literal: "--hf-file" },
        { value: source.path },
      );
      break;
  }

  if (!isCacheType(cacheK) || !isCacheType(cacheV)) throw new ValidationError("Invalid cache type.");

  const ngl = plan.fits ? 999 : plan.ngl;
  args.push({ literal: "-ngl" }, { literal: int(ngl, 0, 999, "layer count") });
  args.push({ literal: "-c" }, { literal: int(ctx, 512, 1_048_576, "context size") });
  if (cacheK !== "f16") args.push({ literal: "-ctk" }, { literal: cacheK });
  if (cacheV !== "f16") args.push({ literal: "-ctv" }, { literal: cacheV });
  if (plan.flashAttention) args.push({ literal: "-fa" }, { literal: "on" });

  const used = plan.tensorSplit.filter((n) => n > 0).length;
  if (plan.tensorSplit.length > 1 && used > 0) {
    const ts = plan.tensorSplit.map((n) => int(n, 0, 100_000, "tensor split")).join(",");
    args.push({ literal: "-sm" }, { literal: "layer" }, { literal: "-ts" }, { literal: ts });
  }
  return args;
}

export function buildCommands(input: CommandInput): Commands {
  const args = buildArgs(input);
  const render = (q: (s: string) => string) =>
    args.map((a) => ("literal" in a ? a.literal : q(a.value))).join(" ");
  return { bash: render(bashQuote), powershell: render(psQuote) };
}
