import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/**", "playwright-report/**", "test-results/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  { files: ["scripts/**", "*.config.*"], languageOptions: { globals: globals.node } },
  { files: ["src/**", "pages/**"], languageOptions: { globals: globals.browser } },
);
