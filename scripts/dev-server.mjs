// Zero-dependency static file server for the web/ explainer.
// ES modules must be served over HTTP (not file://), so this is the dev entry.
//   npm run dev        -> serves web/ on http://localhost:5173
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { watch } from "node:fs";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("../web", import.meta.url)));
const PORT = Number(process.env.PORT) || 5173;
const clients = new Set();
let refreshTimer;
watch(ROOT, { recursive: true }, (_event, filename) => {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => {
    for (const client of clients)
      client.write(
        `data: ${filename?.endsWith(".css") ? "css" : "reload"}\n\n`,
      );
  }, 150);
});
const liveReload = `<script>
const updates = new EventSource('/__updates');
updates.onmessage = ({data}) => {
  if (data === 'css') document.querySelectorAll('link[rel="stylesheet"]').forEach(link => {
    if (new URL(link.href).origin === location.origin) { const url = new URL(link.href); url.searchParams.set('v', Date.now()); link.href = url; }
  });
  else location.reload();
};
</script>`;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
};

const server = createServer(async (req, res) => {
  try {
    let urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
    if (urlPath === "/__updates") {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });
      res.write(": connected\n\n");
      clients.add(res);
      req.on("close", () => clients.delete(res));
      return;
    }
    if (urlPath === "/delta" || urlPath.startsWith("/delta/")) {
      res
        .writeHead(302, {
          Location: "http://localhost:5174/without-a-gradient/delta/",
        })
        .end();
      return;
    }
    if (urlPath === "/") urlPath = "/index.html";
    // Prevent path traversal: resolve and ensure it stays under ROOT.
    const filePath = normalize(join(ROOT, urlPath));
    if (!filePath.startsWith(ROOT + sep)) {
      res.writeHead(403).end("Forbidden");
      return;
    }
    const info = await stat(filePath).catch(() => null);
    const target =
      info && info.isDirectory() ? join(filePath, "index.html") : filePath;
    let body = await readFile(target);
    if (extname(target) === ".html")
      body = Buffer.from(
        body.toString("utf8").replace("</body>", liveReload + "</body>"),
      );
    res.writeHead(200, {
      "Content-Type": MIME[extname(target)] || "application/octet-stream",
      "Cache-Control": "no-cache",
    });
    res.end(body);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain" }).end("404 Not Found");
  }
});

server.listen(PORT, () => {
  console.log(`\n  metaheuristics-lab ; dev server`);
  console.log(`  ➜  http://localhost:${PORT}\n`);
  console.log(`  serving ${ROOT}`);
  console.log(`  press Ctrl+C to stop\n`);
});
