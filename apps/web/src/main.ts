import {
  BOUNDS,
  CACHE_TYPES,
  CTX_MAX,
  CTX_MIN,
  DEFAULT_STATE,
  GPU_PRESETS,
  LIMITS,
  ParseError,
  ValidationError,
  buildCommands,
  findPreset,
  formatBytes,
  formatCount,
  formatCtx,
  groupGgufFiles,
  hfListGgufFiles,
  isCacheType,
  matchQuant,
  maxContext,
  parseHfInput,
  parseShareState,
  plan as computePlan,
  sanitizeText,
  summarizeModel,
  serializeShareState,
  shardInfo,
  shardPaths,
  type CacheType,
  type GgufGroup,
  type GpuCard,
  type HfInput,
  type HfRepoRef,
  type ModelInfo,
  type ModelSource,
  type Plan,
  type ShareState,
} from "@fitcheck/core";
import { byId, h, replace } from "./dom.js";
import { parseAll } from "./worker-client.js";

// ---------------------------------------------------------------------------
// State

interface LoadedModel {
  readonly info: ModelInfo;
  readonly source: ModelSource;
  /** Sanitized label for display. */
  readonly label: string;
  readonly local: boolean;
}

interface Picker {
  readonly ref: HfRepoRef;
  readonly groups: readonly GgufGroup[];
  readonly all: readonly GgufGroup[];
  readonly note: string | null;
}

interface AppState {
  share: ShareState;
  model: LoadedModel | null;
  picker: Picker | null;
  status: { kind: "idle" | "busy" | "error" | "ok"; text: string };
  tab: "bash" | "powershell";
  /** Bumped on each load so stale results are ignored. */
  loadId: number;
}

const state: AppState = {
  share: parseShareState(new URLSearchParams(location.search)),
  model: null,
  picker: null,
  status: { kind: "idle", text: "" },
  tab: "bash",
  loadId: 0,
};

/** Slider stops (8K .. 1M). Any other value can be typed into the box beside it. */
const CTX_STOPS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];

const AI_NEXUS_URL = "https://github.com/iamboosted/ai-nexus";

// ---------------------------------------------------------------------------
// Errors: only our own fixed messages are shown, sanitized and truncated.

function errorText(e: unknown): string {
  if (e instanceof ParseError || e instanceof ValidationError) return sanitizeText(e.message, 200);
  return "Something went wrong while reading the model.";
}

function setStatus(kind: AppState["status"]["kind"], text: string): void {
  state.status = { kind, text };
  const el = byId("status");
  el.className = `status status-${kind}`;
  el.textContent = text;
}

// ---------------------------------------------------------------------------
// Layout

function heading(step: string, text: string, id?: string): HTMLHeadingElement {
  return h("h2", id ? { id } : null, h("span", { class: "step", "aria-hidden": "true" }, step), text);
}


function buildLayout(): void {
  const app = byId("app");
  replace(
    app,
    h(
      "header",
      { class: "top" },
      h(
        "div",
        { class: "brand" },
        h("span", { class: "logo", "aria-hidden": "true" }, h("span", null), h("span", null), h("span", null)),
        h("div", null, h("h1", null, "fitcheck"), h("p", { class: "tagline" }, "Will your model fit? Plan llama.cpp multi-GPU splits from a GGUF header.")),
      ),
      h(
        "div",
        { class: "top-actions" },
        h(
          "a",
          { class: "nexus-link", href: AI_NEXUS_URL, target: "_blank", rel: "noopener noreferrer", title: "AI Nexus: a desktop dashboard for your local models" },
          "AI Nexus \u2197",
        ),
        h("button", { type: "button", id: "theme", class: "ghost small", "aria-label": "Toggle light or dark theme" }, "Light"),
      ),
    ),
    h(
      "main",
      { class: "grid" },
      h(
        "div",
        { class: "col" },
        h(
          "section",
          { class: "card", "aria-labelledby": "in-h" },
          heading("01", "Model", "in-h"),
          h(
            "label",
            { class: "drop", id: "drop", for: "file" },
            h("span", { class: "drop-icon", "aria-hidden": "true" }),
            h("strong", null, "Drop a .gguf file"),
            h("span", null, "or click to browse. Only the header is read, never the weights. Drop every shard of a split model together."),
            h("input", { type: "file", id: "file", accept: ".gguf", multiple: true, class: "visually-hidden" }),
          ),
          h("div", { class: "or" }, h("span", null, "or from Hugging Face")),
          h(
            "div",
            { class: "row" },
            h("label", { for: "hf", class: "visually-hidden" }, "Hugging Face model"),
            h("input", {
              type: "text",
              id: "hf",
              placeholder: "org/repo:Q4_K_M, org/repo, or a huggingface.co URL",
              autocomplete: "off",
              spellcheck: "false",
              autocapitalize: "off",
              maxlength: 2048,
            }),
            h("button", { type: "button", id: "hf-load" }, "Load"),
          ),
          h(
            "details",
            { class: "token" },
            h("summary", null, "Access token (gated repos)"),
            h("label", { for: "token", class: "visually-hidden" }, "Hugging Face access token"),
            h("input", {
              type: "password",
              id: "token",
              autocomplete: "off",
              spellcheck: "false",
              placeholder: "hf_\u2026",
              maxlength: 512,
            }),
            h(
              "p",
              { class: "hint" },
              "Kept in memory only and sent only to huggingface.co. Never stored, never put in the URL. A read-only, fine-grained token is best.",
            ),
          ),
          h("p", { id: "status", class: "status", role: "status", "aria-live": "polite" }),
          h("div", { id: "picker" }),
        ),
        h("section", { class: "card", id: "summary", "aria-label": "Model summary" }),
      ),
      h(
        "div",
        { class: "col" },
        h(
          "section",
          { class: "card", "aria-labelledby": "gpu-h" },
          heading("02", "GPUs", "gpu-h"),
          h("div", { id: "gpus" }),
          h(
            "div",
            { class: "row wrap" },
            h("button", { type: "button", id: "add-gpu", class: "ghost add" }, "+ Add GPU"),
            h(
              "label",
              { class: "inline" },
              "Compute headroom per GPU (GB) ",
              h("input", { type: "number", id: "headroom", min: 0, max: BOUNDS.headroomGbMax, step: 0.25, inputmode: "decimal" }),
            ),
          ),
          h("p", { class: "hint" }, "List cards in llama.cpp device order (run llama-server --list-devices)."),
        ),
        h(
          "section",
          { class: "card", "aria-labelledby": "ctx-h" },
          heading("03", "Context", "ctx-h"),
          h(
            "div",
            { class: "row" },
            h(
              "div",
              { class: "slider" },
              h("label", { for: "ctx-range", class: "visually-hidden" }, "Context size"),
              h("input", { type: "range", id: "ctx-range", min: 0, max: CTX_STOPS.length - 1, step: 1 }),
              h(
                "div",
                { class: "ticks", id: "ctx-ticks" },
                ...CTX_STOPS.map((stop, i) => {
                  const t = h(
                    "button",
                    { type: "button", class: "tick", "aria-label": `Set context to ${formatCtx(stop)} tokens` },
                    formatCtx(stop),
                  );
                  t.style.left = `${(i / (CTX_STOPS.length - 1)) * 100}%`;
                  t.addEventListener("click", () => setCtx(stop));
                  return t;
                }),
              ),
            ),
            h("label", { for: "ctx", class: "visually-hidden" }, "Context size (tokens)"),
            h(
              "label",
              { class: "ctx-box" },
              h("input", { type: "number", id: "ctx", min: CTX_MIN, max: CTX_MAX, step: 256, inputmode: "numeric", class: "num" }),
              h("span", null, "tokens"),
            ),
          ),
          h(
            "div",
            { class: "row wrap" },
            h(
              "label",
              { class: "inline" },
              "KV cache type ",
              h("select", { id: "cache" }, ...CACHE_TYPES.map((t) => h("option", { value: t }, t))),
            ),
            h("span", { id: "maxctx", class: "maxctx" }),
          ),
        ),
        h("section", { class: "card", id: "result", "aria-label": "Plan" }),
        h("section", { class: "card", id: "command", "aria-label": "llama-server command" }),
      ),
    ),
    h(
      "footer",
      null,
      h(
        "a",
        { class: "nexus-card", href: AI_NEXUS_URL, target: "_blank", rel: "noopener noreferrer" },
        h("img", { class: "nexus-mark", src: "./ai-nexus-chevron.svg", alt: "", width: 44, height: 44 }),
        h(
          "span",
          { class: "nexus-text" },
          h("strong", null, "Running models locally? Meet AI Nexus"),
          h("span", null, "A desktop dashboard for your models, your llama.cpp build and your way of working. Windows developer preview."),
        ),
        h("span", { class: "nexus-cta" }, "View on GitHub \u2197"),
      ),
      h(
        "ul",
        { class: "pills" },
        h("li", null, "No uploads"),
        h("li", null, "No backend"),
        h("li", null, "No tracking"),
        h("li", null, "Open source (MIT)"),
      ),
      h("p", null, "Estimates assume llama.cpp's default layer split. Leave yourself some margin."),
    ),
  );
}

// ---------------------------------------------------------------------------
// GPU list

function gpuName(g: GpuCard): string {
  return findPreset(g.preset)?.name ?? `Custom ${g.vramGiB} GB`;
}

function renderGpus(): void {
  const list = byId("gpus");
  const rows = state.share.gpus.map((g, i) => {
    const select = h(
      "select",
      { "aria-label": `GPU ${i + 1} model` },
      ...GPU_PRESETS.map((p) => h("option", { value: p.id, selected: p.id === g.preset }, `${p.name} (${p.vramGiB} GB)`)),
      h("option", { value: "custom", selected: g.preset === "custom" }, "Custom\u2026"),
    );
    const vram = h("input", {
      type: "number",
      min: BOUNDS.gpuGbMin,
      max: BOUNDS.gpuGbMax,
      step: 1,
      value: g.vramGiB,
      inputmode: "decimal",
      class: "num",
      "aria-label": `GPU ${i + 1} VRAM in GB`,
      disabled: g.preset !== "custom",
    });
    const reserved = h("input", {
      type: "number",
      min: 0,
      max: BOUNDS.reservedGbMax,
      step: 0.25,
      value: g.reservedGiB,
      inputmode: "decimal",
      class: "num",
      "aria-label": `GPU ${i + 1} VRAM reserved for desktop or display, in GB`,
    });
    const remove = h(
      "button",
      { type: "button", class: "icon-btn", title: "Remove", "aria-label": `Remove GPU ${i + 1}`, disabled: state.share.gpus.length <= 1 },
      "\u2715",
    );

    select.addEventListener("change", () => {
      const preset = findPreset(select.value);
      const cur = state.share.gpus[i];
      if (!cur) return;
      const next: GpuCard = preset
        ? { preset: preset.id, vramGiB: preset.vramGiB, reservedGiB: Math.min(cur.reservedGiB, preset.vramGiB - 0.5) }
        : { preset: "custom", vramGiB: cur.vramGiB, reservedGiB: cur.reservedGiB };
      updateGpu(i, next, true);
    });
    vram.addEventListener("input", () => {
      const cur = state.share.gpus[i];
      const v = vram.valueAsNumber;
      if (!cur || !Number.isFinite(v) || v < BOUNDS.gpuGbMin || v > BOUNDS.gpuGbMax) return;
      updateGpu(i, { ...cur, vramGiB: v, reservedGiB: Math.min(cur.reservedGiB, v / 2) }, false);
    });
    reserved.addEventListener("input", () => {
      const cur = state.share.gpus[i];
      const v = reserved.valueAsNumber;
      if (!cur || !Number.isFinite(v) || v < 0 || v > BOUNDS.reservedGbMax || v >= cur.vramGiB) return;
      updateGpu(i, { ...cur, reservedGiB: v }, false);
    });
    remove.addEventListener("click", () => {
      if (state.share.gpus.length <= 1) return;
      state.share = { ...state.share, gpus: state.share.gpus.filter((_, j) => j !== i) };
      renderGpus();
      update();
    });

    return h(
      "div",
      { class: "gpu-row" },
      h("span", { class: "gpu-idx" }, `GPU ${i}`),
      select,
      h("label", { class: "inline" }, "VRAM ", vram),
      h("label", { class: "inline" }, "Reserved ", reserved),
      remove,
    );
  });
  replace(list, ...rows);
  (byId("add-gpu") as HTMLButtonElement).disabled = state.share.gpus.length >= BOUNDS.gpuCountMax;
}

function updateGpu(i: number, g: GpuCard, rerender: boolean): void {
  state.share = { ...state.share, gpus: state.share.gpus.map((x, j) => (j === i ? g : x)) };
  if (rerender) renderGpus();
  update();
}

// ---------------------------------------------------------------------------
// Context controls

function nearestStop(ctx: number): number {
  let best = 0;
  for (let i = 0; i < CTX_STOPS.length; i++) {
    const d = Math.abs(Math.log2(CTX_STOPS[i] as number) - Math.log2(ctx));
    if (d < Math.abs(Math.log2(CTX_STOPS[best] as number) - Math.log2(ctx))) best = i;
  }
  return best;
}

function syncContextInputs(): void {
  const range = byId("ctx-range") as HTMLInputElement;
  const idx = nearestStop(state.share.ctx);
  range.value = String(idx);
  range.style.setProperty("--fill", `${(idx / (CTX_STOPS.length - 1)) * 100}%`);
  range.setAttribute("aria-valuetext", `${formatCtx(state.share.ctx)} tokens`);
  const ticks = byId("ctx-ticks").children;
  for (let i = 0; i < ticks.length; i++) {
    ticks[i]?.classList.toggle("active", CTX_STOPS[i] === state.share.ctx);
  }
  (byId("ctx") as HTMLInputElement).value = String(state.share.ctx);
  (byId("cache") as HTMLSelectElement).value = state.share.cache;
  (byId("headroom") as HTMLInputElement).value = String(state.share.headroomGiB);
}

function setCtx(ctx: number): void {
  if (!Number.isInteger(ctx) || ctx < CTX_MIN || ctx > CTX_MAX) return;
  state.share = { ...state.share, ctx };
  syncContextInputs();
  update();
}

// ---------------------------------------------------------------------------
// Rendering: summary, plan, command

function dl(rows: ReadonlyArray<readonly [string, string]>): HTMLDListElement {
  return h("dl", { class: "kv" }, ...rows.flatMap(([k, v]) => [h("dt", null, k), h("dd", null, v)]));
}

function renderSummary(): void {
  const el = byId("summary");
  const m = state.model;
  if (!m) {
    replace(
      el,
      heading("\u2014", "Summary"),
      h("p", { class: "muted" }, "Load a model to see its architecture, size and layer layout."),
    );
    return;
  }
  const i = m.info;
  const rows: Array<[string, string]> = [
    ["Model", m.label],
    ...(i.name ? [["Name", sanitizeText(i.name, 80)] as [string, string]] : []),
    ["Architecture", sanitizeText(i.architecture, 64)],
    ["Parameters", `${formatCount(i.paramCount)}${i.sizeLabel ? ` (${sanitizeText(i.sizeLabel, 24)})` : ""}`],
    ["Quant", i.quant ? sanitizeText(i.quant, 32) : "unknown"],
    ["Weights", formatBytes(i.totalBytes)],
    ["Layers", String(i.nLayers)],
  ];
  if (i.headCount !== null) {
    rows.push(["Attention", `${i.headCount} heads, ${i.headCountKv ?? "?"} KV heads, dim ${i.headDim ?? "?"}`]);
  }
  if (i.expertCount) rows.push(["Experts", `${i.expertCount} (${i.expertUsedCount ?? "?"} active)`]);
  if (i.contextLength) rows.push(["Trained context", formatCtx(i.contextLength)]);
  if (i.shardCount > 1) rows.push(["Shards", String(i.shardCount)]);
  rows.push(["token_embd (CPU)", formatBytes(i.tokenEmbdBytes)]);
  rows.push(["Output layer", `${formatBytes(i.outputBytes)}${i.tiedOutput ? " (tied to embeddings)" : ""}`]);
  if (i.hiddenMetadata.tokens !== null) rows.push(["Vocabulary", `${formatCount(i.hiddenMetadata.tokens)} tokens (not shown)`]);
  if (i.hiddenMetadata.chatTemplateBytes !== null) {
    rows.push(["Chat template", `${formatBytes(i.hiddenMetadata.chatTemplateBytes)} (not shown)`]);
  }
  replace(
    el,
    heading("\u2014", "Summary"),
    dl(rows),
    i.warnings.length > 0
      ? h("ul", { class: "warnings" }, ...i.warnings.slice(0, 10).map((w) => h("li", null, sanitizeText(w, 200))))
      : null,
  );
}

function currentPlan(): { plan: Plan; maxCtx: number | null } | null {
  const m = state.model;
  if (!m) return null;
  const gpus = state.share.gpus.map((g) => ({ name: gpuName(g), vramGiB: g.vramGiB, reservedGiB: g.reservedGiB }));
  const base = { model: m.info, gpus, cacheK: state.share.cache, cacheV: state.share.cache, headroomGiB: state.share.headroomGiB };
  return { plan: computePlan({ ...base, ctx: state.share.ctx }), maxCtx: maxContext(base) };
}

function seg(kind: string, bytes: number, capacity: number, label: string): HTMLElement | null {
  if (bytes <= 0 || capacity <= 0) return null;
  const s = h("span", { class: `seg seg-${kind}`, title: `${label}: ${formatBytes(bytes)}` });
  // CSSOM property writes are allowed under style-src 'self' (no inline style attribute).
  s.style.width = `${Math.min(100, (bytes / capacity) * 100)}%`;
  return s;
}

function renderPlan(result: { plan: Plan; maxCtx: number | null } | null): void {
  const el = byId("result");
  const maxEl = byId("maxctx");
  if (!result || !state.model) {
    const ghost = (w1: number, w2: number) => {
      const bar = h("div", { class: "bar ghost-bar", "aria-hidden": "true" }, h("span", { class: "seg seg-weights" }), h("span", { class: "seg seg-kv" }));
      (bar.children[0] as HTMLElement).style.width = `${w1}%`;
      (bar.children[1] as HTMLElement).style.width = `${w2}%`;
      return bar;
    };
    replace(
      el,
      heading("04", "Fit"),
      h("p", { class: "muted" }, "Load a model to see how weights and KV cache land on each GPU."),
      h("div", { class: "ghost-wrap" }, ghost(46, 14), ghost(38, 12)),
    );
    replace(maxEl);
    return;
  }
  const { plan: p, maxCtx } = result;
  const info = state.model.info;

  replace(
    maxEl,
    maxCtx === null
      ? h("span", { class: "chip bad" }, "Doesn't fully fit even at 512 tokens")
      : h(
          "span",
          { class: "chip" },
          "Max that fits ",
          h("strong", null, maxCtx.toLocaleString("en-US")),
          h("button", { type: "button", class: "ghost small", id: "use-max" }, "Use"),
          info.contextLength && maxCtx > info.contextLength ? h("span", { class: "muted" }, ` (trained ${formatCtx(info.contextLength)})`) : null,
        ),
  );
  const useMax = document.getElementById("use-max");
  if (useMax && maxCtx !== null) {
    const target = info.contextLength ? Math.min(maxCtx, info.contextLength) : maxCtx;
    useMax.addEventListener("click", () => setCtx(Math.max(CTX_MIN, target)));
  }

  const verdict = p.fits
    ? h("p", { class: "verdict good" }, h("span", { class: "dot", "aria-hidden": "true" }), `Fits: all ${p.nLayers} layers + output on GPU`)
    : h(
        "p",
        { class: "verdict bad" },
        h("span", { class: "dot", "aria-hidden": "true" }),
        `Doesn't fully fit. Max -ngl ${p.ngl} of ${p.nLayers + 1}; ${formatBytes(p.cpuSpillBytes)} of weights` +
          ` and ${formatBytes(p.cpuKvBytes)} of KV cache spill to system RAM.`,
      );

  const bars = p.gpus.map((g) => {
    const used = g.reserved + g.headroom + g.weights + g.kv;
    const over = used > g.capacity;
    const layers = g.layers ? `layers ${g.layers[0]}\u2013${g.layers[1] - 1}` : "no layers";
    return h(
      "div",
      { class: `gpu-bar${over ? " over" : ""}` },
      h(
        "div",
        { class: "bar-head" },
        h(
          "div",
          { class: "bar-title" },
          h("span", { class: "gpu-tag" }, `GPU ${g.index}`),
          h("strong", null, sanitizeText(g.name, 40)),
          h("span", { class: "muted" }, `${layers}${g.hasOutput ? " + output" : ""}`),
        ),
        h("span", { class: `bar-total${over ? " bad" : ""}` }, `${formatBytes(Math.min(used, g.capacity * 9))} / ${formatBytes(g.capacity)}`),
      ),
      h(
        "div",
        { class: "bar", role: "img", "aria-label": `${formatBytes(used)} of ${formatBytes(g.capacity)} used` },
        seg("reserved", g.reserved, g.capacity, "Reserved"),
        seg("weights", g.weights, g.capacity, "Weights"),
        seg("kv", g.kv, g.capacity, "KV cache"),
        seg("headroom", g.headroom, g.capacity, "Headroom"),
      ),
      h(
        "div",
        { class: "bar-legend" },
        h("span", { class: "lg lg-weights" }, `weights ${formatBytes(g.weights)}`),
        h("span", { class: "lg lg-kv" }, `KV ${formatBytes(g.kv)}`),
        h("span", { class: "lg lg-headroom" }, `headroom ${formatBytes(g.headroom)}`),
        g.reserved > 0 ? h("span", { class: "lg lg-reserved" }, `reserved ${formatBytes(g.reserved)}`) : null,
        h("span", { class: over ? "bad" : "muted" }, over ? `over by ${formatBytes(-g.free)}` : `free ${formatBytes(g.free)}`),
      ),
    );
  });

  replace(
    el,
    heading("04", "Fit"),
    verdict,
    ...bars,
    dl([
      ["Total KV cache", `${formatBytes(p.totalKvBytes)} at ${formatCtx(state.share.ctx)} (${state.share.cache})`],
      ["System RAM (weights)", `${formatBytes(p.cpuWeightBytes)} (token_embd${p.cpuSpillBytes > 0 ? " + spilled layers" : ""})`],
    ]),
  );
}

function renderCommand(result: { plan: Plan } | null): void {
  const el = byId("command");
  const m = state.model;
  if (!result || !m) {
    replace(
      el,
      heading("05", "llama-server"),
      h(
        "div",
        { class: "terminal" },
        h("div", { class: "term-bar" }, h("span", { class: "term-dots", "aria-hidden": "true" }, h("span", null), h("span", null), h("span", null))),
        h("pre", { class: "cmd placeholder" }, h("code", null, "llama-server -m your-model.gguf -ngl \u2026")),
      ),
    );
    return;
  }
  let cmds: { bash: string; powershell: string } | null = null;
  let err: string | null = null;
  try {
    cmds = buildCommands({
      source: m.source,
      plan: result.plan,
      ctx: state.share.ctx,
      cacheK: state.share.cache,
      cacheV: state.share.cache,
    });
  } catch (e) {
    err = errorText(e);
  }

  const tabs = h(
    "div",
    { class: "tabs", role: "tablist" },
    ...(["bash", "powershell"] as const).map((t) => {
      const b = h(
        "button",
        { type: "button", role: "tab", class: "tab", "aria-selected": state.tab === t ? "true" : "false", "aria-controls": "cmd-text" },
        t === "bash" ? "bash" : "PowerShell",
      );
      b.addEventListener("click", () => {
        state.tab = t;
        renderCommand(result);
      });
      return b;
    }),
  );

  const text = cmds ? (state.tab === "bash" ? cmds.bash : cmds.powershell) : "";
  const copy = h("button", { type: "button", class: "small copy", disabled: !cmds }, "Copy");
  copy.addEventListener("click", () => {
    navigator.clipboard.writeText(text).then(
      () => flash(copy, "Copied"),
      () => flash(copy, "Copy failed"),
    );
  });
  const share = h("button", { type: "button", class: "ghost small" }, "Copy share link");
  share.addEventListener("click", () => {
    navigator.clipboard.writeText(location.href).then(
      () => flash(share, "Link copied"),
      () => flash(share, "Copy failed"),
    );
  });

  const notes: string[] = [];
  if (m.local) notes.push("Loaded from a local file: the share link includes GPUs and context, but not the file.");
  if (result.plan.flashAttention) notes.push("-fa on was added because a quantized V cache requires flash attention.");
  if (result.plan.tensorSplit.length > 1) notes.push("-ts lists layers per GPU in device order (the output layer counts on the last card).");
  if (!result.plan.fits) notes.push("Partial offload: expect much slower generation for layers left on the CPU.");
  if (m.source.kind !== "local" && state.share.hf && state.share.hf.revision !== "main") {
    notes.push("llama-server downloads from the default branch, not the revision in your URL.");
  }

  replace(
    el,
    heading("05", "llama-server"),
    h(
      "div",
      { class: "terminal" },
      h(
        "div",
        { class: "term-bar" },
        h("span", { class: "term-dots", "aria-hidden": "true" }, h("span", null), h("span", null), h("span", null)),
        tabs,
        copy,
      ),
      err
        ? h("p", { class: "bad term-err" }, err)
        : h("pre", { class: `cmd${state.tab === "powershell" ? " ps" : ""}`, id: "cmd-text", tabindex: 0 }, h("code", null, text)),
    ),
    h("div", { class: "row wrap" }, share),
    notes.length > 0 ? h("ul", { class: "notes" }, ...notes.map((n) => h("li", null, n))) : null,
  );
}

function flash(btn: HTMLButtonElement, text: string): void {
  const prev = btn.textContent ?? "";
  btn.textContent = text;
  setTimeout(() => (btn.textContent = prev), 1200);
}

function renderPicker(): void {
  const el = byId("picker");
  const p = state.picker;
  if (!p) {
    replace(el);
    return;
  }
  const items = p.groups.slice(0, 200).map((g) => {
    const b = h(
      "button",
      { type: "button", class: "pick", disabled: !g.complete },
      h("span", { class: "pick-name" }, sanitizeText(g.label, 90)),
      h(
        "span",
        { class: "pick-meta" },
        `${formatBytes(g.totalSize)}${g.paths.length > 1 ? ` \u00B7 ${g.paths.length} shards` : ""}${g.complete ? "" : " \u00B7 incomplete"}`,
      ),
    );
    b.addEventListener("click", () => void loadHfGroup(p.ref, g, p.all));
    return b;
  });
  replace(
    el,
    p.note ? h("p", { class: "hint" }, p.note) : null,
    items.length > 0 ? h("div", { class: "picks" }, ...items) : h("p", { class: "muted" }, "No GGUF files found in this repo."),
  );
}

// ---------------------------------------------------------------------------
// Update cycle

function syncUrl(): void {
  const qs = serializeShareState(state.share).toString();
  history.replaceState(null, "", `${location.pathname}?${qs}`);
}

function update(): void {
  syncUrl();
  const result = currentPlan();
  renderPlan(result);
  renderCommand(result);
}

// ---------------------------------------------------------------------------
// Loading

function token(): string | undefined {
  const v = (byId("token") as HTMLInputElement).value.trim();
  return v.length > 0 ? v : undefined;
}

async function loadLocal(fileList: FileList | readonly File[]): Promise<void> {
  const files = Array.from(fileList).slice(0, LIMITS.maxShards + 1);
  if (files.length === 0) return;
  const id = ++state.loadId;
  state.picker = null;
  renderPicker();
  try {
    if (files.length > LIMITS.maxShards) throw new ValidationError("Too many files.");
    if (!files.every((f) => f.name.toLowerCase().endsWith(".gguf"))) throw new ValidationError("Only .gguf files are supported.");
    if (files.length > 1) {
      const bases = new Set(files.map((f) => {
        const s = shardInfo(f.name);
        return s ? `${s.base}/${s.count}` : null;
      }));
      if (bases.has(null) || bases.size !== 1) {
        throw new ValidationError("Drop one model at a time (or every shard of one model).");
      }
      files.sort((a, b) => (shardInfo(a.name)?.index ?? 0) - (shardInfo(b.name)?.index ?? 0));
    }
    setStatus("busy", `Reading header${files.length > 1 ? `s of ${files.length} shards` : ""}\u2026`);
    const parsed = await parseAll(files.map((file) => ({ kind: "blob" as const, file })));
    if (id !== state.loadId) return;
    const info = summarizeModel(parsed);
    const first = files[0] as File;
    state.model = { info, source: { kind: "local", fileName: first.name }, label: sanitizeText(first.name, 90), local: true };
    state.share = { ...state.share, hf: null };
    const read = parsed.reduce((n, f) => n + f.bytesRead, 0);
    setStatus("ok", `Read ${formatBytes(read)} of header data; the rest of the file was never touched.`);
    renderSummary();
    update();
  } catch (e) {
    if (id !== state.loadId) return;
    setStatus("error", errorText(e));
  }
}

async function loadHfInput(input: HfInput): Promise<void> {
  const id = ++state.loadId;
  state.picker = null;
  renderPicker();
  const ref: HfRepoRef = { org: input.org, repo: input.repo, revision: input.revision };
  try {
    if (input.kind === "file") {
      const paths = shardPaths(input.path);
      await loadHfPaths(id, ref, paths, { kind: "hf-file", org: ref.org, repo: ref.repo, path: paths[0] as string }, { path: paths[0] as string, quant: null });
      return;
    }
    setStatus("busy", "Listing GGUF files\u2026");
    const files = await hfListGgufFiles(ref, { token: token(), signal: AbortSignal.timeout(15_000) });
    if (id !== state.loadId) return;
    const all = groupGgufFiles(files);
    const models = all.filter((g) => !g.isProjector);
    if (input.kind === "quant") {
      const matches = matchQuant(all, input.quant);
      if (matches.length === 1) {
        const g = matches[0] as GgufGroup;
        await loadHfPaths(id, ref, g.paths, { kind: "hf-quant", org: ref.org, repo: ref.repo, quant: input.quant }, { path: null, quant: input.quant });
        return;
      }
      state.picker = {
        ref,
        groups: matches.length > 1 ? matches : models,
        all,
        note: matches.length > 1 ? "Several files match that quant; pick one." : "No file matches that quant; pick one.",
      };
    } else {
      const projectors = all.length - models.length;
      state.picker = { ref, groups: models, all, note: projectors > 0 ? `${projectors} multimodal projector file(s) hidden.` : null };
    }
    setStatus("ok", `${models.length} GGUF model file(s) found.`);
    renderPicker();
  } catch (e) {
    if (id !== state.loadId) return;
    setStatus("error", errorText(e));
  }
}

async function loadHfGroup(ref: HfRepoRef, g: GgufGroup, all: readonly GgufGroup[]): Promise<void> {
  const id = ++state.loadId;
  const matches = g.quant ? matchQuant(all, g.quant) : [];
  const unique = g.quant !== null && matches.length === 1 && matches[0] === g;
  const source: ModelSource = unique
    ? { kind: "hf-quant", org: ref.org, repo: ref.repo, quant: g.quant as string }
    : { kind: "hf-file", org: ref.org, repo: ref.repo, path: g.paths[0] as string };
  try {
    await loadHfPaths(id, ref, g.paths, source, unique ? { path: null, quant: g.quant } : { path: g.paths[0] as string, quant: null });
  } catch (e) {
    if (id !== state.loadId) return;
    setStatus("error", errorText(e));
  }
}

async function loadHfPaths(
  id: number,
  ref: HfRepoRef,
  paths: readonly string[],
  source: ModelSource,
  sel: { path: string | null; quant: string | null },
): Promise<void> {
  if (paths.length > LIMITS.maxShards) throw new ValidationError("Too many shards.");
  setStatus("busy", `Fetching header${paths.length > 1 ? `s of ${paths.length} shards` : ""} with Range requests\u2026`);
  const tok = token();
  const parsed = await parseAll(paths.map((path) => ({ kind: "hf" as const, ref, path, ...(tok ? { token: tok } : {}) })));
  if (id !== state.loadId) return;
  const info = summarizeModel(parsed);
  const label =
    source.kind === "hf-quant"
      ? `${ref.org}/${ref.repo}:${source.quant}`
      : `${ref.org}/${ref.repo}/${(paths[0] as string).split("/").pop() ?? ""}`;
  state.model = { info, source, label: sanitizeText(label, 120), local: false };
  state.share = { ...state.share, hf: { ...ref, quant: sel.quant, path: sel.path } };
  const read = parsed.reduce((n, f) => n + f.bytesRead, 0);
  setStatus("ok", `Fetched ${formatBytes(read)} of header data.`);
  renderSummary();
  update();
}

// ---------------------------------------------------------------------------
// Wiring

function wire(): void {
  const fileInput = byId("file") as HTMLInputElement;
  fileInput.addEventListener("change", () => {
    if (fileInput.files) void loadLocal(fileInput.files);
    fileInput.value = "";
  });

  const drop = byId("drop");
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("dragging");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("dragging");
    if (e.dataTransfer?.files) void loadLocal(e.dataTransfer.files);
  });
  // Dropping outside the zone must not navigate to the file.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  const hfInput = byId("hf") as HTMLInputElement;
  const go = () => {
    try {
      void loadHfInput(parseHfInput(hfInput.value));
    } catch (e) {
      setStatus("error", errorText(e));
    }
  };
  byId("hf-load").addEventListener("click", go);
  hfInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") go();
  });

  const tokenInput = byId("token") as HTMLInputElement;
  const clearToken = () => {
    tokenInput.value = "";
  };
  window.addEventListener("pagehide", clearToken);
  window.addEventListener("beforeunload", clearToken);

  byId("add-gpu").addEventListener("click", () => {
    if (state.share.gpus.length >= BOUNDS.gpuCountMax) return;
    const last = state.share.gpus[state.share.gpus.length - 1] ?? DEFAULT_STATE.gpus[0];
    if (!last) return;
    state.share = { ...state.share, gpus: [...state.share.gpus, { ...last, reservedGiB: 0 }] };
    renderGpus();
    update();
  });

  const headroom = byId("headroom") as HTMLInputElement;
  headroom.addEventListener("input", () => {
    const v = headroom.valueAsNumber;
    if (!Number.isFinite(v) || v < 0 || v > BOUNDS.headroomGbMax) return;
    state.share = { ...state.share, headroomGiB: v };
    update();
  });

  const range = byId("ctx-range") as HTMLInputElement;
  range.addEventListener("input", () => {
    const v = CTX_STOPS[Number(range.value)];
    if (v !== undefined) setCtx(v);
  });
  const ctxInput = byId("ctx") as HTMLInputElement;
  ctxInput.addEventListener("change", () => {
    const v = Math.round(ctxInput.valueAsNumber);
    if (Number.isFinite(v)) setCtx(Math.min(CTX_MAX, Math.max(CTX_MIN, v)));
    else syncContextInputs();
  });
  const cache = byId("cache") as HTMLSelectElement;
  cache.addEventListener("change", () => {
    const v: unknown = cache.value;
    if (!isCacheType(v)) return;
    state.share = { ...state.share, cache: v as CacheType };
    update();
  });

  byId("theme").addEventListener("click", () => {
    const root = document.documentElement;
    const next = root.dataset["theme"] === "light" ? "dark" : "light";
    root.dataset["theme"] = next;
    byId("theme").textContent = next === "light" ? "Dark" : "Light";
  });
}

function init(): void {
  buildLayout();
  wire();
  renderGpus();
  syncContextInputs();
  renderSummary();
  update();

  const hf = state.share.hf;
  if (hf) {
    const input: HfInput = hf.path
      ? { kind: "file", org: hf.org, repo: hf.repo, revision: hf.revision, path: hf.path }
      : hf.quant
        ? { kind: "quant", org: hf.org, repo: hf.repo, revision: hf.revision, quant: hf.quant }
        : { kind: "repo", org: hf.org, repo: hf.repo, revision: hf.revision };
    (byId("hf") as HTMLInputElement).value =
      input.kind === "quant" ? `${hf.org}/${hf.repo}:${input.quant}` : input.kind === "file" ? `https://huggingface.co/${hf.org}/${hf.repo}/blob/${hf.revision}/${input.path}` : `${hf.org}/${hf.repo}`;
    void loadHfInput(input);
  }
}

init();
