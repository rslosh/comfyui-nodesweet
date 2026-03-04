import { defineConfig } from "vite"
import vue from "@vitejs/plugin-vue"
import cssInjectedByJsPlugin from "vite-plugin-css-injected-by-js"
import { resolve } from "path"

export default defineConfig({
  plugins: [vue(), cssInjectedByJsPlugin()],
  build: {
    lib: {
      entry: resolve(__dirname, "./src/main.ts"),
      formats: ["es"],
      fileName: "main",
    },
    rollupOptions: {
      external: [
        /\.\.\/\.\.\/scripts\/.*/,
      ],
      output: {
        dir: resolve(__dirname, "../js"),
        entryFileNames: "main.js",
        chunkFileNames: "assets/[name]-[hash].js",
      },
    },
  },
})
