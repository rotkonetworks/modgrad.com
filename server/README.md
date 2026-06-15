# modgrad contact endpoint

A tiny zero-dependency Node service that receives the website contact form and
sends it over **JMAP**. The email goes from `noreply@rotko.net` to
`modgrad@rotko.net`, with the inquirer set as **Reply-To** so the team can reply
directly. Nginx proxies `POST /api/contact` to this service.

## Environment

| var | required | default | notes |
|-----|----------|---------|-------|
| `JMAP_SESSION_URL` | **yes** | — | JMAP session resource, e.g. `https://<mail-host>/.well-known/jmap` |
| `JMAP_TOKEN` | **yes** | — | bearer token for the sending account (has a `noreply@rotko.net` identity) |
| `MAIL_FROM` | no | `noreply@rotko.net` | envelope + From address |
| `MAIL_TO` | no | `modgrad@rotko.net` | destination inbox |
| `PORT` | no | `3001` | listen port (internal to the podman network) |

`JMAP_SESSION_URL` and `JMAP_TOKEN` are **secrets** — set them in the repo's
GitHub Actions secrets; the deploy workflow passes them to the container. Never
commit them.

## Run locally

```bash
JMAP_SESSION_URL=https://mail.example/.well-known/jmap \
JMAP_TOKEN=xxxx \
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
