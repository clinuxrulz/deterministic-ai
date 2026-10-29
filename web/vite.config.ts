import { defineConfig } from "vite";
import solid from "@solidjs/vite-plugin";

export default defineConfig({
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
