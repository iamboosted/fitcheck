/**
 * Minimal DOM builder. Text always goes through text nodes; attributes are
 * allowlisted (no event handler attributes, no style, no srcdoc).
 */
export type Child = Node | string | number | null | undefined | false;

const ALLOWED_ATTRS = new Set([
  "id", "class", "type", "value", "min", "max", "step", "placeholder", "for", "name", "role",
  "title", "inputmode", "accept", "autocomplete", "spellcheck", "autocapitalize", "tabindex",
  "aria-label", "aria-live", "aria-selected", "aria-controls", "aria-pressed", "aria-hidden",
  "aria-describedby", "aria-labelledby", "aria-valuenow", "aria-valuemin", "aria-valuemax",
  "href", "rel", "target", "src", "alt", "width", "height", "data-tab", "data-kind", "list", "maxlength", "size",
]);

const BOOLEAN_ATTRS = new Set(["multiple", "disabled", "hidden", "checked", "selected", "readonly"]);

export type Attrs = Record<string, string | number | boolean | null | undefined>;

function setAttr(el: Element, name: string, value: string | number | boolean | null | undefined): void {
  if (value === null || value === undefined || value === false) return;
  if (BOOLEAN_ATTRS.has(name)) {
    if (value === true) el.setAttribute(name, "");
    return;
  }
  if (!ALLOWED_ATTRS.has(name)) throw new Error(`attribute not allowed: ${name}`);
  const v = String(value);
  if (name === "href" && !/^(?:https:\/\/|#|\.\/)/.test(v)) throw new Error("href not allowed");
  // Images are same-origin files only (CSP img-src 'self').
  if (name === "src" && !/^\.\/[A-Za-z0-9._-]+$/.test(v)) throw new Error("src not allowed");
  el.setAttribute(name, v);
}

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Attrs | null = null,
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) setAttr(el, k, v);
  // New-tab links never get a handle back to this page or leak a referrer.
  if (el.hasAttribute("target")) el.setAttribute("rel", "noopener noreferrer");
  append(el, children);
  return el;
}

export function append(el: Node, children: readonly Child[]): void {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    el.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
  }
}

export function replace(el: Element, ...children: Child[]): void {
  el.replaceChildren();
  append(el, children);
}

export function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
}
