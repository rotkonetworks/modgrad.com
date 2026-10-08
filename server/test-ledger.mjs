// Smoke test for the Ledger money logic. `node test-ledger.mjs`. Zero deps.
import { rmSync } from "node:fs";
import { FileLedger, InsufficientBalance, USD, providerCost, settle } from "./ledger.mjs";

let pass = 0, fail = 0;
const eq = (name, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  console.log(`${ok ? "✓" : "✗"} ${name}${ok ? "" : `  got ${JSON.stringify(got)} want ${JSON.stringify(want)}`}`);
  ok ? pass++ : fail++;
};
const throws = async (name, fn, Type) => {
  try { await fn(); console.log(`✗ ${name}  (no throw)`); fail++; }
  catch (e) { const ok = e instanceof Type; console.log(`${ok ? "✓" : "✗"} ${name}`); ok ? pass++ : fail++; }
};

const PATH = "./data/ledger.test.json";
rmSync(PATH, { force: true });

// ── pure pricing ──
// gpt-4o: 1000 in × $2.50/Mtok + 500 out × $10/Mtok = 2500 + 5000 = 7500 µUSD
eq("providerCost gpt-4o", providerCost("gpt-4o", { prompt_tokens: 1000, completion_tokens: 500 }), 7500);
// unknown model → DEFAULT_PRICE (5/15): 1000×5 + 100×15 = 6500
eq("providerCost unknown", providerCost("mystery", { prompt_tokens: 1000, completion_tokens: 100 }), 6500);

// ── settle: hosted marks up, BYO is free ──
eq("settle hosted +20%", settle({ byoKey: false, model: "gpt-4o", usage: { prompt_tokens: 1000, completion_tokens: 500 }, margin: 0.2 }),
  { providerµUSD: 7500, userµUSD: 9000 });
eq("settle byo is free", settle({ byoKey: true, model: "gpt-4o", usage: { prompt_tokens: 1000, completion_tokens: 500 }, margin: 0.2 }),
  { providerµUSD: 7500, userµUSD: 0 });

// ── ledger credit/debit/balance ──
const L = new FileLedger(PATH);
eq("start balance 0", await L.balance("u1"), 0);
eq("credit $5", await L.credit("u1", USD(5)), 5_000_000);
eq("debit 9000µ", await L.debit("u1", 9000), 4_991_000);
await throws("debit over balance throws", () => L.debit("u1", USD(1000)), InsufficientBalance);
await throws("negative credit rejected", () => L.credit("u1", -1), Error);

// ── persistence round-trip: reopen the file, balance survives ──
const L2 = new FileLedger(PATH);
eq("balance persisted", await L2.balance("u1"), 4_991_000);
eq("history recorded", (await L2.history("u1")).map((e) => e.type), ["credit", "debit"]);

// ── concurrent debits are serialised (no lost updates) ──
const L3 = new FileLedger("./data/ledger.test2.json");
rmSync("./data/ledger.test2.json", { force: true });
const L3b = new FileLedger("./data/ledger.test2.json");
await L3b.credit("u2", 1000);
await Promise.all([L3b.debit("u2", 300), L3b.debit("u2", 300), L3b.debit("u2", 300)]);
eq("serialised debits", await L3b.balance("u2"), 100);

rmSync(PATH, { force: true });
rmSync("./data/ledger.test2.json", { force: true });
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
