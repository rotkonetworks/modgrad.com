// Proxy test: identity through a stub modgrad-server whoami, metering into a
// FileLedger, BYO settings by account id. `node test-proxy.mjs`. Zero deps.
import { createServer } from "node:http";
import { rmSync, mkdirSync, writeFileSync } from "node:fs";
import { FileLedger, USD } from "./ledger.mjs";

let pass = 0, fail = 0;
const eq = (name, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  console.log(`${ok ? "✓" : "✗"} ${name}${ok ? "" : `  got ${JSON.stringify(got)} want ${JSON.stringify(want)}`}`);
  ok ? pass++ : fail++;
};
const listen = (srv) => new Promise((r) => srv.listen(0, "127.0.0.1", () => r(`http://127.0.0.1:${srv.address().port}`)));

// ── stub modgrad-server: GET /api/v1/auth/whoami ──
const TOKENS = { "obsp_alice": { account: "acct_alice", via: "token", project: "lab" }, "obsp_bob": { account: "acct_bob", via: "token", project: "bob-lab" } };
const seen = [];
const whoamiSrv = createServer((req, res) => {
  seen.push({ url: req.url, auth: req.headers.authorization, cookie: req.headers.cookie });
  const t = (req.headers.authorization || "").replace(/^Bearer\s+/i, "");
  const who = req.url === "/api/v1/auth/whoami" && TOKENS[t];
  res.writeHead(who ? 200 : 401, { "Content-Type": "application/json" });
  res.end(JSON.stringify(who || { error: "unknown or expired token", code: "bad_token" }));
});

// ── stub upstream (OpenAI-compatible) ──
let upstreamKey = null;
const upstream = createServer((req, res) => {
  upstreamKey = req.headers.authorization;
  let b = "";
  req.on("data", (c) => (b += c));
  req.on("end", () => {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ id: "x", choices: [{ message: { content: "hi" } }], usage: { prompt_tokens: 1000, completion_tokens: 500 } }));
  });
});

const upstreamURL = await listen(upstream);
process.env.OPENAI_BASE_URL = upstreamURL; // read when proxy.mjs loads
process.env.OPENAI_API_KEY = "sk-hosted";
const { createProxy } = await import("./proxy.mjs");

const whoamiURL = await listen(whoamiSrv);
mkdirSync("./data", { recursive: true });
const LEDGER = "./data/ledger.proxytest.json", SETTINGS = "./data/settings.proxytest.json";
rmSync(LEDGER, { force: true });
writeFileSync(SETTINGS, JSON.stringify({ acct_bob: { tier: "byo", byoKey: "sk-bob" } }));
const ledger = new FileLedger(LEDGER);
await ledger.credit("acct_alice", USD(1));

const proxy = createProxy({ ledger, whoamiURL, settingsPath: SETTINGS });
const P = await listen(proxy);
const get = async (path, auth, extra = {}) => {
  const r = await fetch(P + path, { headers: { ...(auth ? { authorization: auth } : {}), ...extra } });
  return [r.status, await r.json()];
};
const chat = async (auth) => {
  const r = await fetch(P + "/v1/chat/completions", { method: "POST", headers: { authorization: auth, "content-type": "application/json" }, body: JSON.stringify({ model: "gpt-4o", messages: [] }) });
  return [r.status, await r.json()];
};

// ── identity ──
let [code, body] = await get("/v1/balance", "Bearer obsp_alice");
eq("balance via whoami", [code, body.account, body.tier, body.balance_uusd], [200, "acct_alice", "hosted", 1_000_000]);
eq("whoami got the bearer", seen.at(-1).auth, "Bearer obsp_alice");
[code, body] = await get("/v1/balance", "Bearer obsp_nope");
eq("unknown token → 401 with the server's code", [code, body.error.type, body.error.code], [401, "unauthorized", "bad_token"]);
const before = seen.length;
[code] = await get("/v1/balance", null, { cookie: "mg_session=" + "a".repeat(43) });
eq("no bearer → 401, whoami not asked, cookie never forwarded", [code, seen.length], [401, before]);
eq("health needs no account", (await get("/api/health"))[0], 200);

// ── metering ──
[code, body] = await chat("Bearer obsp_alice");
eq("hosted chat proxied with our key", [code, upstreamKey], [200, "Bearer sk-hosted"]);
await new Promise((r) => setTimeout(r, 50)); // debit settles after the response
eq("hosted debit: 7500µ × 1.2", await ledger.balance("acct_alice"), 1_000_000 - 9000);
[code] = await chat("Bearer obsp_bob");
eq("byo chat uses the account's key", [code, upstreamKey], [200, "Bearer sk-bob"]);
eq("byo settings by account id; nothing debited", [(await get("/v1/balance", "Bearer obsp_bob"))[1].tier, (await ledger.history("acct_bob")).length], ["byo", 0]);

// ── account service down → 503, not 401 ──
await new Promise((r) => whoamiSrv.close(r));
[code, body] = await get("/v1/balance", "Bearer obsp_alice");
eq("whoami unreachable → 503", [code, body.error.type], [503, "unavailable"]);

proxy.close();
upstream.close();
rmSync(LEDGER, { force: true });
rmSync(SETTINGS, { force: true });
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
