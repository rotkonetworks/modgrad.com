# modgrad-obs container

One image: `modgrad-server` (ingest + reads), the `modgrad` and `modgrad-wandb` CLIs, and
the dashboard (`obs/web`, built in a node stage, with its analysis and ingest
wasm bundled). The server stores and serves data only;
graphs, fits and forecasts are computed in the viewer's browser.

    podman build -f deploy/obs/Containerfile -t modgrad-obs .
    podman run -p 8790:8790 -v obs-data:/data modgrad-obs
    # open http://localhost:8790/

Build args: `JOBS` (cargo parallelism, default 6). Runs as uid 10001; data
lives in the `/data` volume (`/data/runs/<run>/`; boards saved by the MCP
`make_board` tool in `/data/boards/`, opened as `/#board=<name>`). Image ≈ 97 MB. `--dev` (appended to `podman run … modgrad-obs`) accepts writes without a token.

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
