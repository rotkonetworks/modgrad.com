#!/usr/bin/env node
// Tiny static file server that sends the cross-origin-isolation headers
// (COOP: same-origin + COEP: require-corp) the browser requires before it
// will hand out SharedArrayBuffer — which wasm-bindgen-rayon needs for the
// multithreaded engine. For LOCAL VERIFICATION ONLY; production headers are
// handled separately in nginx (NOT touched by this task).
//
//   node scripts/serve-coop-coep.mjs [port] [rootDir]
//   node scripts/serve-coop-coep.mjs 8788 dist
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { join, normalize, extname } from "node:path";

const port = Number(process.argv[2] || 8788);
const root = normalize(join(process.cwd(), process.argv[3] || "dist"));

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".txt": "text/plain; charset=utf-8",
  ".map": "application/json; charset=utf-8",
};

const server = createServer(async (req, res) => {
  // The two headers that enable crossOriginIsolated → SharedArrayBuffer.
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  // so cross-origin-isolated docs can still load these same-origin assets.
  res.setHeader("Cross-Origin-Resource-Policy", "same-origin");

  try {
    const url = decodeURIComponent((req.url || "/").split("?")[0]);
    let path = normalize(join(root, url));
    if (!path.startsWith(root)) {
      res.writeHead(403).end("forbidden");
      return;
    }
    let s;
    try {
      s = await stat(path);
    } catch {
      s = null;
    }
    if (s && s.isDirectory()) path = join(path, "index.html");
    let body;
    try {
      body = await readFile(path);
    } catch {
      // SPA fallback: serve index.html for unknown routes.
      try {
        body = await readFile(join(root, "index.html"));
        path = join(root, "index.html");
      } catch {
        res.writeHead(404).end("not found");
        return;
      }
    }
    res.setHeader("Content-Type", MIME[extname(path)] || "application/octet-stream");
    res.writeHead(200).end(body);
  } catch (e) {
    res.writeHead(500).end(String(e));
  }
});

server.listen(port, () => {
  console.log(`COOP/COEP static server: http://localhost:${port}  (root: ${root})`);
  console.log("crossOriginIsolated will be true → SharedArrayBuffer enabled.");
});
