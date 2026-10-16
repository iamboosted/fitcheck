export function formatBytes(n: number): string {
  if (!Number.isFinite(n)) return "?";
  const neg = n < 0;
  let v = Math.abs(n);
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  const s = u === 0 ? String(Math.round(v)) : v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2);
  return `${neg ? "-" : ""}${s} ${units[u]}`;
}

export function formatCount(n: number): string {
  if (!Number.isFinite(n)) return "?";
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(Math.round(n));
}

export function formatCtx(n: number): string {
  if (n >= 1048576 && n % 1048576 === 0) return `${n / 1048576}M`;
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}K`;
  return String(n);
}
