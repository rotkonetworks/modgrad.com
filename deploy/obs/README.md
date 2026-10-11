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
