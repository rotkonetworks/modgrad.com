// Ledger — the ONE seam between the app and however account balances are kept.
//
// Everything upstream (the proxy, the metering) talks ONLY to this interface:
//
//   balance(userId)             -> integer µUSD
//   credit(userId, µusd, meta)  -> new balance            (top-ups)
//   debit(userId, µusd, meta)   -> new balance            (usage; throws if short)
//   history(userId, limit?)     -> [event, ...]           (append-only journal)
//
// v1 below (`FileLedger`) is a zero-dep, file-backed implementation. To go
// on-chain / decentralised later, write a second object exposing the SAME four
// methods and swap the constructor in the proxy — nothing upstream changes.
// That swap is the entire reason this file exists. Keep the interface tiny.

import { readFileSync, writeFileSync, renameSync, mkdirSync, existsSync } from "node:fs";
import { dirname } from "node:path";

// ── Money is integer µUSD ($1 = 1_000_000). Never floats — no drift on money. ──
export const USD = (dollars) => Math.round(dollars * 1_000_000);
export const fmtUSD = (micro) => `$${(micro / 1_000_000).toFixed(6)}`;

export class InsufficientBalance extends Error {
  constructor(have, need) {
    super(`insufficient balance: have ${fmtUSD(have)}, need ${fmtUSD(need)}`);
    this.name = "InsufficientBalance";
    this.have = have;
    this.need = need;
  }
}

// ── Pricing: provider list price in USD per 1M tokens, per model. ──
// These are what WE pay upstream. Update to real numbers per provider.
// The user-facing price is provider cost × (1 + MARGIN) — see `settle`.
export const PRICES = {
  "gpt-4o": { in: 2.5, out: 10.0 },
  "gpt-4o-mini": { in: 0.15, out: 0.6 },
  "o4-mini": { in: 1.1, out: 4.4 },
  "claude-sonnet-4-6": { in: 3.0, out: 15.0 },
  "claude-opus-4-8": { in: 15.0, out: 75.0 },
  "claude-haiku-4-5": { in: 1.0, out: 5.0 },
};
// Fallback price when a model isn't in the table — deliberately not free.
export const DEFAULT_PRICE = { in: 5.0, out: 15.0 };

// Pure: what the request cost US upstream, in integer µUSD.
export function providerCost(model, usage) {
  const p = PRICES[model] || DEFAULT_PRICE;
  const inTok = usage?.inputTokens ?? usage?.prompt_tokens ?? 0;
  const outTok = usage?.outputTokens ?? usage?.completion_tokens ?? 0;
  // price is per 1e6 tokens, in USD → µUSD is ×1e6, so the /1e6 and ×1e6 cancel:
  // cost_µUSD = tokens × price_per_Mtok.
  return Math.round(inTok * p.in + outTok * p.out);
}

// Pure: given the request context, how much to debit the USER (µUSD).
// This is where the "hybrid tiers" earn model lives:
//   - byoKey  → user brought their own provider key → we debit 0 (no resale).
//   - hosted  → we fronted the provider cost → charge cost × (1 + margin).
// `margin` is the platform take (e.g. 0.2 = 20%). Returns both figures so the
// journal can record what we paid vs what we charged.
export function settle({ byoKey, model, usage, margin = 0.2 }) {
  const provider = providerCost(model, usage);
  const user = byoKey ? 0 : Math.round(provider * (1 + margin));
  return { providerµUSD: provider, userµUSD: user };
}

// ── FileLedger: zero-dep, atomic, single-process. Swap for a DB or chain later. ──
export class FileLedger {
  #path;
  #state;
  #lock = Promise.resolve(); // serialises read-modify-write so money is exact

  constructor(path = process.env.LEDGER_PATH || "./data/ledger.json") {
    this.#path = path;
    this.#state = { balances: {}, events: [] };
    if (existsSync(path)) {
      try {
        this.#state = JSON.parse(readFileSync(path, "utf8"));
      } catch (e) {
        throw new Error(`ledger file ${path} is corrupt: ${e.message}`);
      }
    }
  }

  #save() {
    mkdirSync(dirname(this.#path), { recursive: true });
    const tmp = `${this.#path}.tmp`;
    writeFileSync(tmp, JSON.stringify(this.#state));
    renameSync(tmp, this.#path); // atomic on POSIX — never a half-written ledger
  }

  // Run `fn` under the lock. All mutations go through here.
  #tx(fn) {
    const run = this.#lock.then(fn, fn);
    // keep the chain alive even if fn throws, but don't swallow the error
    this.#lock = run.then(() => {}, () => {});
    return run;
  }

  async balance(userId) {
    return this.#state.balances[userId] ?? 0;
  }

  async credit(userId, amount, meta = {}) {
    assertAmount(amount);
    return this.#tx(() => {
      const next = (this.#state.balances[userId] ?? 0) + amount;
      this.#state.balances[userId] = next;
      this.#record(userId, "credit", amount, next, meta);
      this.#save();
      return next;
    });
  }

  async debit(userId, amount, meta = {}) {
    assertAmount(amount);
    return this.#tx(() => {
      const have = this.#state.balances[userId] ?? 0;
      if (have < amount) throw new InsufficientBalance(have, amount);
      const next = have - amount;
      this.#state.balances[userId] = next;
      this.#record(userId, "debit", amount, next, meta);
      this.#save();
      return next;
    });
  }

  async history(userId, limit = 100) {
    return this.#state.events
      .filter((e) => e.userId === userId)
      .slice(-limit);
  }

  #record(userId, type, amount, balanceAfter, meta) {
    this.#state.events.push({ ts: Date.now(), userId, type, amount, balanceAfter, meta });
  }
}

function assertAmount(amount) {
  if (!Number.isInteger(amount) || amount <= 0) {
    throw new Error(`amount must be a positive integer µUSD, got ${amount}`);
  }
}
