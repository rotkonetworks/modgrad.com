# modgrad-obs container

One image: `obs-server` (ingest + reads), the `obs` and `obs-wandb` CLIs, and
the dashboard with its analysis wasm. The server stores and serves data only;
graphs, fits and forecasts are computed in the viewer's browser.

    podman build -f deploy/obs/Containerfile -t modgrad-obs .
    podman run -p 8790:8790 -v obs-data:/data modgrad-obs
    # open http://localhost:8790/

Build args: `JOBS` (cargo parallelism, default 6). Runs as uid 10001; data
lives in the `/data` volume (`/data/runs/<run>/`). Image ≈ 94 MB.
