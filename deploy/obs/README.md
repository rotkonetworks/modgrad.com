# modgrad-obs container

One image: `modgrad-server` (ingest + reads), the `modgrad` and `modgrad-wandb` CLIs, and
the dashboard (`obs/web`, built in a node stage, with its analysis and ingest
wasm bundled). The server stores and serves data only;
graphs, fits and forecasts are computed in the viewer's browser.

    podman build -f deploy/obs/Containerfile -t modgrad-obs .
    podman run -p 8790:8790 -v obs-data:/data modgrad-obs
    # open http://localhost:8790/

## Under a path (modgrad.com/app)

modgrad.com runs this image as `modgrad-app` behind the site's nginx, at
`/app/` (`.github/workflows/deploy.yaml`, `deploy/nginx/modgrad.conf`):

    podman run … modgrad-app --public-url https://modgrad.com/app --mail jmap --ssh 0.0.0.0:2222 --mcp

nginx strips `/app` from the raw request URI (so `%2F` in run ids survives)
and does not buffer (SSE). With a path in `--public-url`, every link the
server prints keeps it (`/app/login#email=…`, `/app/#run=…`), the session
cookie gets `Path=/app`, and the browser origin checked for CSRF, zafu and
passkeys is still `https://modgrad.com` (passkey rpId `modgrad.com`).
Clients take the prefixed URL as the server: `MODGRAD_URL=https://modgrad.com/app`.

Build args: `JOBS` (cargo parallelism, default 6). Runs as uid 10001; data
lives in the `/data` volume (`/data/runs/<run>/`; boards saved by the MCP
`make_board` tool in `/data/boards/`, opened as `/#board=<name>`). Image ≈ 97 MB. `--dev` (appended to `podman run … modgrad-obs`) accepts writes without a token.

## SSH door (optional)

`python train.py 2>&1 | ssh -p 2222 obs.example.org` streams a training log
into a new run with nothing installed: the output is echoed back, metric
lines become rows, the dashboard URL is printed first. The client's SSH key
is its account (one anonymous project per key; no signup, no password).

    podman run -p 8790:8790 -p 2222:2222 -v obs-data:/data modgrad-obs \
        --ssh 0.0.0.0:2222 --public-url https://obs.example.org

- **`--public-url`** is the base of the links SSH prints (and, when set, of
  links in HTTP responses, instead of the request's Host header). Without it
  SSH links point at the listen address.
- Port 22 needs root; the image runs as uid 10001, so it listens on 2222.
  Map public 22 to it (`-p 22:2222`) if the host's own sshd is elsewhere;
  then users type plain `ssh obs.example.org`.
- The host key is `/data/ssh_host_ed25519`, made on first start, so it
  lives in the volume and survives image rebuilds. Back it up with the data.
- Limits: `--ssh-idle <secs>` (close a silent connection, default 3600),
  `--ssh-max-conns` (all connections, default 256), `--ssh-max-per-key`
  (concurrent sessions per key, default 8). Lines over 64 KiB are echoed
  but only their tail is parsed.
- Only fixed verbs run (`run`, `token`, `runs`, `tail`, `help`); no shell,
  no command execution, no forwarding.

## HTTP/3 (optional)

`modgrad run` and `modgrad push` can send rows over HTTP/3 (QUIC). The server
needs a UDP port and a certificate for the public hostname (QUIC always uses
TLS). The TCP port stays plain HTTP behind your TLS proxy:

    podman run -p 8790:8790 -p 8790:8790/udp -v obs-data:/data -v /etc/obs-tls:/tls:ro modgrad-obs \
        --h3 0.0.0.0:8790 --tls-cert /tls/fullchain.pem --tls-key /tls/privkey.pem

- **Open the UDP port** in the firewall. The key must be readable by uid 10001.
- TCP responses then carry `Alt-Svc: h3=":8790"`. Clients look for HTTP/3 on
  that UDP port of the hostname they reached over TCP, so the port must be
  reachable at the public name. If a port mapping changes the number (e.g.
  public UDP 443, which passes more firewalls), add `--h3-advertise 443`.
  The TLS proxy must pass the `Alt-Svc` header through, which most proxies do.
- **Fallback:** if UDP is blocked, the client gives up on QUIC after 1.5 s and
  keeps using TCP for the rest of the run. Nothing is lost: rows are spooled
  before they are sent, whatever the transport. Browsers, curl and the Python
  client always use TCP.
- `--h3-self-signed` (instead of the cert flags) is for local testing only.
  It writes the certificate to `/data/h3-self-signed.pem`; clients pass it
  with `--tls-ca`.
