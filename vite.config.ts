import { defineConfig, type Plugin } from "vite";
import solid from "vite-plugin-solid";
import unocss from "unocss/vite";
import { fileURLToPath } from "url";
import { join } from "path";
import { buildDocs } from "./docs-build";

const VID = "virtual:docs";
const RESOLVED = "\0" + VID;

/** Exposes rendered docs (frontmatter + HTML + TOC) as a virtual module. */
function docsPlugin(): Plugin {
  return {
    name: "modgrad-docs",
    resolveId(id) {
      if (id === VID) return RESOLVED;
    },
    load(id) {
      if (id !== RESOLVED) return;
      const { pages, sections } = buildDocs();
      return `export const pages = ${JSON.stringify(pages)};\nexport const sections = ${JSON.stringify(sections)};`;
    },
    configureServer(server) {
      const dir = fileURLToPath(new URL("./src/docs", import.meta.url));
      server.watcher.add(dir);
      server.watcher.on("change", (file) => {
        if (file.startsWith(dir) && file.endsWith(".md")) {
          const mod = server.moduleGraph.getModuleById(RESOLVED);
          if (mod) server.moduleGraph.invalidateModule(mod);
          server.ws.send({ type: "full-reload" });
        }
      });
    },
  };
}

export default defineConfig({
  publicDir: "public",
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  plugins: [unocss(), solid(), docsPlugin()],
});
