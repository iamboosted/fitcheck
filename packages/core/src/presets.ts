export interface GpuPreset {
  readonly id: string;
  readonly name: string;
  readonly vramGiB: number;
}

export const GPU_PRESETS: readonly GpuPreset[] = Object.freeze([
  { id: "rx9070xt", name: "RX 9070 XT", vramGiB: 16 },
  { id: "rx9060xt", name: "RX 9060 XT 16GB", vramGiB: 16 },
  { id: "rx7900xtx", name: "RX 7900 XTX", vramGiB: 24 },
  { id: "rx7900xt", name: "RX 7900 XT", vramGiB: 20 },
  { id: "rx7800xt", name: "RX 7800 XT", vramGiB: 16 },
  { id: "rtx3060", name: "RTX 3060 12GB", vramGiB: 12 },
  { id: "rtx4070s", name: "RTX 4070 Super", vramGiB: 12 },
  { id: "rtx4090", name: "RTX 4090", vramGiB: 24 },
  { id: "rtx5090", name: "RTX 5090", vramGiB: 32 },
].map((p) => Object.freeze(p)));

export function findPreset(id: string): GpuPreset | undefined {
  return GPU_PRESETS.find((p) => p.id === id);
}
