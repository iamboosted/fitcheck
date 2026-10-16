import tseslint from "typescript-eslint";

/** Sinks that must never appear: they parse strings as HTML or code. */
const htmlSinks = ["innerHTML", "outerHTML", "insertAdjacentHTML", "srcdoc"].map((property) => ({
  property,
  message: "HTML string sinks are banned. Build DOM nodes and use textContent.",
}));

const securityRules = {
  "no-eval": "error",
  "no-implied-eval": "error",
  "no-new-func": "error",
  "no-script-url": "error",
  "no-restricted-properties": [
    "error",
    ...htmlSinks,
    { object: "document", property: "write", message: "document.write is banned." },
    { object: "document", property: "writeln", message: "document.writeln is banned." },
    { object: "window", property: "eval", message: "eval is banned." },
    { object: "globalThis", property: "eval", message: "eval is banned." },
    { object: "Range", property: "createContextualFragment", message: "HTML parsing is banned." },
  ],
  "no-restricted-syntax": [
    "error",
    {
      selector: "MemberExpression[property.name=/^(innerHTML|outerHTML|insertAdjacentHTML|createContextualFragment|srcdoc)$/]",
      message: "HTML string sinks are banned.",
    },
    {
      selector: "MemberExpression[property.value=/^(innerHTML|outerHTML|insertAdjacentHTML|createContextualFragment|srcdoc)$/]",
      message: "HTML string sinks are banned (computed access).",
    },
    { selector: "CallExpression[callee.name='eval']", message: "eval is banned." },
    { selector: "NewExpression[callee.name='Function']", message: "new Function is banned." },
    { selector: "CallExpression[callee.name='Function']", message: "Function() is banned." },
    { selector: "CallExpression[callee.property.name='setAttribute'][arguments.0.value=/^on/i]", message: "Inline event handler attributes are banned." },
    { selector: "CallExpression[callee.property.name='setAttribute'][arguments.0.value='style']", message: "Inline style attributes are banned (CSP)." },
    { selector: "CallExpression[callee.property.name='parseFromString']", message: "DOMParser is banned." },
    { selector: "CallExpression[callee.name=/^(setTimeout|setInterval)$/][arguments.0.type='Literal']", message: "String timers are banned." },
  ],
  "no-restricted-globals": [
    "error",
    { name: "localStorage", message: "No persistent storage (tokens must never be stored)." },
    { name: "sessionStorage", message: "No persistent storage (tokens must never be stored)." },
    { name: "indexedDB", message: "No persistent storage (tokens must never be stored)." },
    { name: "eval", message: "eval is banned." },
  ],
};

export default [
  { ignores: ["**/dist/**", "**/node_modules/**", "coverage/**"] },
  {
    files: ["**/*.ts", "**/*.js"],
    languageOptions: {
      parser: tseslint.parser,
      ecmaVersion: 2022,
      sourceType: "module",
    },
    plugins: { "@typescript-eslint": tseslint.plugin },
    rules: {
      ...securityRules,
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_", caughtErrors: "none" }],
    },
  },
  {
    // The config itself names the banned sinks in strings; tests deliberately construct hostile input.
    files: ["eslint.config.js"],
    rules: { "no-restricted-syntax": "off", "no-restricted-properties": "off" },
  },
];
