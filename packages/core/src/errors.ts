/**
 * Error codes are stable identifiers; messages are fixed English text that
 * never contains bytes, strings or names taken from the parsed input.
 */
export type ParseErrorCode =
  | "BAD_MAGIC"
  | "BAD_VERSION"
  | "TRUNCATED"
  | "LIMIT"
  | "INVALID"
  | "TOO_MANY_FETCHES"
  | "IO"
  | "TIMEOUT";

export class ParseError extends Error {
  readonly code: ParseErrorCode;

  constructor(code: ParseErrorCode, message: string) {
    super(message);
    this.name = "ParseError";
    this.code = code;
  }
}

export class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ValidationError";
  }
}
