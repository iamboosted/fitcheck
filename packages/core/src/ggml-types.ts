export interface GgmlTypeInfo {
  readonly name: string;
  /** Elements per block. */
  readonly blockSize: number;
  /** Bytes per block. */
  readonly typeSize: number;
}

/**
 * ggml tensor types (ids from ggml.h). Ids 4, 5, 31-33 and 36-38 were removed
 * upstream and are treated as unknown.
 */
const TABLE: ReadonlyArray<readonly [number, string, number, number]> = [
  [0, "F32", 1, 4],
  [1, "F16", 1, 2],
  [2, "Q4_0", 32, 18],
  [3, "Q4_1", 32, 20],
  [6, "Q5_0", 32, 22],
  [7, "Q5_1", 32, 24],
  [8, "Q8_0", 32, 34],
  [9, "Q8_1", 32, 36],
  [10, "Q2_K", 256, 84],
  [11, "Q3_K", 256, 110],
  [12, "Q4_K", 256, 144],
  [13, "Q5_K", 256, 176],
  [14, "Q6_K", 256, 210],
  [15, "Q8_K", 256, 292],
  [16, "IQ2_XXS", 256, 66],
  [17, "IQ2_XS", 256, 74],
  [18, "IQ3_XXS", 256, 98],
  [19, "IQ1_S", 256, 50],
  [20, "IQ4_NL", 32, 18],
  [21, "IQ3_S", 256, 110],
  [22, "IQ2_S", 256, 82],
  [23, "IQ4_XS", 256, 136],
  [24, "I8", 1, 1],
  [25, "I16", 1, 2],
  [26, "I32", 1, 4],
  [27, "I64", 1, 8],
  [28, "F64", 1, 8],
  [29, "IQ1_M", 256, 56],
  [30, "BF16", 1, 2],
  [34, "TQ1_0", 256, 54],
  [35, "TQ2_0", 256, 66],
  [39, "MXFP4", 32, 17],
];

const BY_ID: ReadonlyMap<number, GgmlTypeInfo> = new Map(
  TABLE.map(([id, name, blockSize, typeSize]) => [id, Object.freeze({ name, blockSize, typeSize })]),
);

export function ggmlType(id: number): GgmlTypeInfo | undefined {
  return BY_ID.get(id);
}

/** llama_ftype values stored in general.file_type. */
const FILE_TYPES: ReadonlyMap<number, string> = new Map([
  [0, "F32"],
  [1, "F16"],
  [2, "Q4_0"],
  [3, "Q4_1"],
  [7, "Q8_0"],
  [8, "Q5_0"],
  [9, "Q5_1"],
  [10, "Q2_K"],
  [11, "Q3_K_S"],
  [12, "Q3_K_M"],
  [13, "Q3_K_L"],
  [14, "Q4_K_S"],
  [15, "Q4_K_M"],
  [16, "Q5_K_S"],
  [17, "Q5_K_M"],
  [18, "Q6_K"],
  [19, "IQ2_XXS"],
  [20, "IQ2_XS"],
  [21, "Q2_K_S"],
  [22, "IQ3_XS"],
  [23, "IQ3_XXS"],
  [24, "IQ1_S"],
  [25, "IQ4_NL"],
  [26, "IQ3_S"],
  [27, "IQ3_M"],
  [28, "IQ2_S"],
  [29, "IQ2_M"],
  [30, "IQ4_XS"],
  [31, "IQ1_M"],
  [32, "BF16"],
  [36, "TQ1_0"],
  [37, "TQ2_0"],
  [38, "MXFP4_MOE"],
]);

export function fileTypeName(ftype: number): string | undefined {
  return FILE_TYPES.get(ftype);
}
