# modgrad contact endpoint

A tiny zero-dependency Node service that receives the website contact form and
sends it over **JMAP**. The email goes from `noreply@rotko.net` to
`modgrad@rotko.net`, with the inquirer set as **Reply-To** so the team can reply
directly. Nginx proxies `POST /api/contact` to this service.

## Environment

Stalwart JMAP uses **Basic auth** with the account's own credentials.

| var | required | default | notes |
|-----|----------|---------|-------|
| `JMAP_URL` | **yes** | — | mail server base, e.g. `https://mail.rotko.net` (session = `…/.well-known/jmap`) |
| `JMAP_USER` | **yes** | — | sending account, `noreply@rotko.net` |
| `JMAP_PASS` | **yes** | — | that account's password |
| `MAIL_FROM` | no | `JMAP_USER` | envelope + From address |
| `MAIL_TO` | no | `modgrad@rotko.net` | destination inbox |
| `PORT` | no | `3001` | listen port (internal to the podman network) |

`JMAP_URL`, `JMAP_USER`, `JMAP_PASS` are **secrets** — set them in the repo's
GitHub Actions secrets; the deploy workflow passes them to the container. Never
commit them.

## Run locally

```bash
JMAP_URL=https://mail.rotko.net \
JMAP_USER=noreply@rotko.net \
JMAP_PASS=xxxx \
node contact.mjs
# POST http://localhost:3001/api/contact  {"email":"you@x.com","message":"hi"}
```

## Behavior

- validates the email and message; rejects oversized bodies (>20 KB)
- in-memory rate limit: 5 requests / 10 min / IP
- a hidden `company` honeypot field silently drops bots
- `GET /api/health` → `{ ok: true }`

## Deploy

Built and run by `.github/workflows/deploy.yaml` as the `modgrad-api` container on
the `modgrad-net` podman network, alongside the nginx static server which proxies
`/api/` to it.

# LLM proxy (`proxy.mjs`, not deployed yet)

An OpenAI-compatible gateway (`POST /v1/chat/completions`, `GET /v1/balance`)
that meters every request through the ledger (`ledger.mjs`: `balance`,
`credit`, `debit`, `history`, integer µUSD).

Callers are **modgrad accounts**. A request carries
`Authorization: Bearer <modgrad project token>` (`obsp_…`, minted on the
account page at `/app/account`); the proxy asks modgrad-server
`GET /api/v1/auth/whoami` over the internal podman network, and the answer's
`account` (`acct_…`) keys the ledger. A project token stands for its project's
owner; run tokens and anonymous (unclaimed) projects are refused. Cookies are
never forwarded, so a cross-site page cannot spend a signed-in user's balance.
whoami unreachable → 503.

| var | default | notes |
|-----|---------|-------|
| `MODGRAD_INTERNAL_URL` | `http://modgrad-app:8790` | modgrad-server, no `/app` prefix (internal network) |
| `SETTINGS_PATH` | `./data/settings.json` | per-account billing: `{ "acct_…": { "tier": "byo", "byoKey": "sk-…" } }`; no entry = hosted |
| `LEDGER_PATH` | `./data/ledger.json` | the FileLedger journal |
| `MARGIN` | `0.2` | hosted price = provider cost × (1 + MARGIN) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | — | hosted upstream keys (secrets) |
| `PORT` | `3002` | |

`npm test` runs the ledger test and `test-proxy.mjs` (a stub whoami and a stub
upstream; no network).
