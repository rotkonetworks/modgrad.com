// modgrad contact endpoint — receives the website form and sends it over JMAP.
// Sends from MAIL_FROM (noreply@rotko.net) to MAIL_TO (modgrad@rotko.net) with
// the inquirer set as Reply-To, so the team can reply directly. Zero deps.
import { createServer } from "node:http";

const PORT = Number(process.env.PORT || 3001);
const SESSION_URL = process.env.JMAP_SESSION_URL || "";
const TOKEN = process.env.JMAP_TOKEN || "";
const MAIL_FROM = process.env.MAIL_FROM || "noreply@rotko.net";
const MAIL_TO = process.env.MAIL_TO || "modgrad@rotko.net";

const USING = [
  "urn:ietf:params:jmap:core",
  "urn:ietf:params:jmap:mail",
  "urn:ietf:params:jmap:submission",
];

// ── tiny in-memory rate limit: 5 requests / 10 min / ip ──────────────────────
const WINDOW = 10 * 60 * 1000;
const LIMIT = 5;
const hits = new Map();
function rateLimited(ip) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter((t) => now - t < WINDOW);
  arr.push(now);
  hits.set(ip, arr);
  return arr.length > LIMIT;
}

const emailOk = (e) => typeof e === "string" && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e);

async function jmap(url, body) {
  const res = await fetch(url, {
    method: body ? "POST" : "GET",
    headers: {
      Authorization: `Bearer ${TOKEN}`,
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`JMAP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  return res.json();
}

async function send({ name, email, message }) {
  if (!SESSION_URL || !TOKEN) throw new Error("JMAP not configured");

  const session = await jmap(SESSION_URL);
  const accountId = session.primaryAccounts["urn:ietf:params:jmap:mail"];
  const apiUrl = session.apiUrl;

  const meta = await jmap(apiUrl, {
    using: USING,
    methodCalls: [
      ["Identity/get", { accountId }, "i"],
      ["Mailbox/get", { accountId, properties: ["role", "name"] }, "m"],
    ],
  });
  const identities = meta.methodResponses[0][1].list || [];
  const mailboxes = meta.methodResponses[1][1].list || [];
  const identity =
    identities.find((x) => x.email?.toLowerCase() === MAIL_FROM.toLowerCase()) || identities[0];
  if (!identity) throw new Error("no JMAP identity available");
  const drafts =
    mailboxes.find((x) => x.role === "drafts") ||
    mailboxes.find((x) => x.role === "sent") ||
    mailboxes[0];
  const sent = mailboxes.find((x) => x.role === "sent");

  const text =
    `New modgrad inquiry\n\n` +
    `From: ${name || "(no name)"} <${email}>\n` +
    `------------------------------------------------------------\n\n` +
    `${message}\n`;

  const create = {
    mailboxIds: { [drafts.id]: true },
    keywords: { $draft: true, $seen: true },
    from: [{ email: MAIL_FROM, name: "modgrad" }],
    to: [{ email: MAIL_TO }],
    replyTo: [{ email, name: name || null }],
    subject: `modgrad inquiry — ${email}`,
    bodyValues: { b: { value: text } },
    textBody: [{ partId: "b", type: "text/plain" }],
  };

  const onSuccess = sent
    ? {
        onSuccessUpdateEmail: {
          "#sub": { mailboxIds: { [sent.id]: true }, "keywords/$draft": null },
        },
      }
    : {};

  const out = await jmap(apiUrl, {
    using: USING,
    methodCalls: [
      ["Email/set", { accountId, create: { draft: create } }, "e"],
      [
        "EmailSubmission/set",
        {
          accountId,
          create: {
            sub: {
              identityId: identity.id,
              emailId: "#draft",
              envelope: { mailFrom: { email: MAIL_FROM }, rcptTo: [{ email: MAIL_TO }] },
            },
          },
          ...onSuccess,
        },
        "s",
      ],
    ],
  });

  const eSet = out.methodResponses.find((m) => m[0] === "Email/set")?.[1];
  const sSet = out.methodResponses.find((m) => m[0] === "EmailSubmission/set")?.[1];
  if (eSet?.notCreated?.draft) throw new Error("email not created: " + JSON.stringify(eSet.notCreated.draft));
  if (sSet?.notCreated?.sub) throw new Error("submission failed: " + JSON.stringify(sSet.notCreated.sub));
}

function json(res, code, obj) {
  const s = JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(s) });
  res.end(s);
}

const server = createServer((req, res) => {
  const url = (req.url || "").split("?")[0];

  if (req.method === "GET" && url === "/api/health") return json(res, 200, { ok: true });
  if (req.method !== "POST" || url !== "/api/contact") return json(res, 404, { error: "not found" });

  const ip = (req.headers["x-forwarded-for"]?.split(",")[0] || req.socket.remoteAddress || "?").trim();
  if (rateLimited(ip)) return json(res, 429, { error: "Too many requests — try again later." });

  let body = "";
  let tooBig = false;
  req.on("data", (c) => {
    body += c;
    if (body.length > 20_000) {
      tooBig = true;
      req.destroy();
    }
  });
  req.on("end", async () => {
    if (tooBig) return;
    let data;
    try {
      data = JSON.parse(body);
    } catch {
      return json(res, 400, { error: "Bad request." });
    }
    // honeypot — pretend success, send nothing
    if (data.company) return json(res, 200, { ok: true });

    const name = String(data.name || "").slice(0, 200);
    const email = String(data.email || "").slice(0, 320);
    const message = String(data.message || "").trim().slice(0, 5000);
    if (!emailOk(email)) return json(res, 400, { error: "A valid email is required." });
    if (message.length < 3) return json(res, 400, { error: "Message is too short." });

    try {
      await send({ name, email, message });
      json(res, 200, { ok: true });
    } catch (err) {
      console.error("contact send failed:", err);
      json(res, 502, { error: "Could not send right now. Please try again or use GitHub." });
    }
  });
});

server.listen(PORT, () => console.log(`modgrad contact endpoint on :${PORT}`));
