import { defineConfig } from "vite";
import solid from "@solidjs/vite-plugin";

// Project Pages are served from a sub-path, so the deploy workflow sets
// SITE_BASE=/deterministic-ai/. Local dev and `vite preview` keep "/".
const siteBase =
  (
    globalThis as {
      process?: { env?: Record<string, string | undefined> };
    }
  ).process?.env?.SITE_BASE ?? "/";

export default defineConfig({
  base: siteBase,
  plugins: [solid()],
  server: {
    port: 3000,
  },
  build: {
    target: "esnext",
    // Keep the WASM package as a separate asset instead of inlining it.
    assetsInlineLimit: 0,
  },
});
