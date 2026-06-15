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
