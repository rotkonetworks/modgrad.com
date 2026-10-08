// modgrad LLM proxy — an OpenAI-compatible gateway that meters every request
// through the Ledger. Thin HTTP shell; all money logic lives in ledger.mjs.
//
// Flow (mirrors the OpenCode "Zen" lifecycle, minus the ceremony):
//   auth (key → account) → balance precheck → forward to upstream (BYO or
//   hosted key) → stream response back while capturing usage → settle → debit.
//
// Hybrid tiers: an account with its own `byoKey` pays 0 (we just proxy); a
// hosted account is charged provider-cost × (1 + MARGIN). See settle().
import { createServer } from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { FileLedger, settle, fmtUSD } from "./ledger.mjs";

const PORT = Number(process.env.PORT || 3002);
const MARGIN = Number(process.env.MARGIN || 0.2);
const MIN_BALANCE = Number(process.env.MIN_BALANCE_UUSD || 1); // hosted needs >0

const ledger = new FileLedger();

// ── Accounts: apiKey → { userId, tier, byoKey? }. File-backed, swappable. ──
// Identity, not money — deliberately separate from the Ledger. Format:
//   { "mg-lIVE-abc": { "userId": "u1", "tier": "hosted" },
//     "mg-byok-xyz": { "userId": "u2", "tier": "byo", "byoKey": "sk-..." } }
const ACCOUNTS_PATH = process.env.ACCOUNTS_PATH || "./data/accounts.json";
function resolveAccount(apiKey) {
  if (!apiKey || !existsSync(ACCOUNTS_PATH)) return null;
  try {
    const table = JSON.parse(readFileSync(ACCOUNTS_PATH, "utf8"));
    return table[apiKey] || null;
  } catch {
    return null;
  }
}

// ── Upstream routing: model → { baseURL, key }. One OpenAI-compatible hop. ──
// Hosted requests use OUR provider key; BYO requests use the account's key.
const PROVIDERS = [
  { match: /^(gpt-|o\d|text-|chatgpt)/, baseURL: process.env.OPENAI_BASE_URL || "https://api.openai.com/v1", key: () => process.env.OPENAI_API_KEY },
  { match: /^claude-/, baseURL: process.env.ANTHROPIC_BASE_URL || "https://api.anthropic.com/v1", key: () => process.env.ANTHROPIC_API_KEY },
];
function routeFor(model) {
  return PROVIDERS.find((p) => p.match.test(model || "")) || PROVIDERS[0];
}

function json(res, code, obj) {
  const s = JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(s) });
  res.end(s);
}

// Pull the OpenAI-style usage object out of a (possibly streamed) response body.
// Non-stream: it's `body.usage`. Stream: the final data chunk carries it when
// the client requested stream_options.include_usage (we force that on below).
function extractUsage(text, isStream) {
  if (!isStream) {
    try { return JSON.parse(text).usage || null; } catch { return null; }
  }
  let usage = null;
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (!s.startsWith("data:")) continue;
    const payload = s.slice(5).trim();
    if (payload === "[DONE]") continue;
    try {
      const u = JSON.parse(payload).usage;
      if (u) usage = u;
    } catch { /* partial line, ignore */ }
  }
  return usage;
}

async function handleChat(req, res, account, reqBody) {
  const isStream = !!reqBody.stream;
  const model = reqBody.model;
  const byoKey = account.byoKey || null;

  // Hosted accounts must have a positive balance before we front any cost.
  if (!byoKey) {
    const bal = await ledger.balance(account.userId);
    if (bal < MIN_BALANCE) return json(res, 402, { error: { message: "insufficient balance — top up to continue", type: "insufficient_balance", balance_uusd: bal } });
  }

  const route = routeFor(model);
  const upstreamKey = byoKey || route.key();
  if (!upstreamKey) return json(res, 502, { error: { message: "no upstream key configured for this model" } });

  // Force usage reporting on streams so we can meter them.
  const payload = isStream
    ? { ...reqBody, stream_options: { ...(reqBody.stream_options || {}), include_usage: true } }
    : reqBody;

  let upstream;
  try {
    upstream = await fetch(`${route.baseURL}/chat/completions`, {
      method: "POST",
      headers: { Authorization: `Bearer ${upstreamKey}`, "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    return json(res, 502, { error: { message: `upstream unreachable: ${e.message}` } });
  }

  // Mirror upstream status + content-type; stream the body through verbatim
  // while teeing a copy to capture usage once the response completes.
  res.writeHead(upstream.status, { "Content-Type": upstream.headers.get("content-type") || "application/json" });
  const reader = upstream.body?.getReader();
  const decoder = new TextDecoder();
  let captured = "";
  if (reader) {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
      captured += decoder.decode(value, { stream: true });
    }
  }
  res.end();

  // Meter — only on a successful upstream call.
  if (upstream.status >= 200 && upstream.status < 300) {
    const usage = extractUsage(captured, isStream);
    if (usage) {
      const { providerµUSD, userµUSD } = settle({ byoKey: !!byoKey, model, usage, margin: MARGIN });
      const meta = { model, usage: { in: usage.prompt_tokens, out: usage.completion_tokens }, providerµUSD };
      if (userµUSD > 0) {
        try {
          await ledger.debit(account.userId, userµUSD, meta);
        } catch (e) {
          // Response already sent; log the shortfall for reconciliation.
          console.error(`debit failed for ${account.userId}: ${e.message}`);
        }
      }
    } else {
      console.warn(`no usage captured for ${account.userId} model=${model} stream=${isStream} — not metered`);
    }
  }
}

const server = createServer((req, res) => {
  const url = (req.url || "").split("?")[0];

  if (req.method === "GET" && url === "/api/health") return json(res, 200, { ok: true });

  // Auth: `Authorization: Bearer <modgrad key>` → account.
  const bearer = (req.headers.authorization || "").replace(/^Bearer\s+/i, "").trim();
  const account = resolveAccount(bearer);
  if (!account) return json(res, 401, { error: { message: "invalid or missing API key" } });

  if (req.method === "GET" && url === "/v1/balance") {
    return ledger.balance(account.userId).then((bal) =>
      json(res, 200, { userId: account.userId, tier: account.tier, balance_uusd: bal, balance: fmtUSD(bal) })
    );
  }

  if (req.method === "POST" && url === "/v1/chat/completions") {
    let body = "";
    let tooBig = false;
    req.on("data", (c) => { body += c; if (body.length > 2_000_000) { tooBig = true; req.destroy(); } });
    req.on("end", async () => {
      if (tooBig) return;
      let data;
      try { data = JSON.parse(body); } catch { return json(res, 400, { error: { message: "invalid JSON body" } }); }
      if (!data.model) return json(res, 400, { error: { message: "`model` is required" } });
      try {
        await handleChat(req, res, account, data);
      } catch (e) {
        console.error("chat handler error:", e);
        if (!res.headersSent) json(res, 500, { error: { message: "internal error" } });
        else res.end();
      }
    });
    return;
  }

  return json(res, 404, { error: { message: "not found" } });
});

server.listen(PORT, () => console.log(`modgrad LLM proxy on :${PORT} (margin ${MARGIN}, ledger metering on)`));
