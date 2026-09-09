# Local Qwen image generation

Runtime: MFLUX 0.19.1 on Apple Silicon. Model: `mflux-community/qwen-image-2512-mflux-q4`, a quantized Qwen-Image-2512 checkpoint. The repository download is approximately 27.6 GB including supporting components.

From the project directory:

```bash
bash scripts/qwen-image-local.sh 'A red cabin beside a lake at sunrise' /private/tmp/cabin.png
```

The script uses 768×768 resolution, 30 steps, seed 42, and low-memory mode. With no arguments it generates a cabin scene at `/private/tmp/qwen-image.png`. The first run downloads the weights to the Hugging Face cache; later runs reuse them. Each invocation loads the model and exits after writing the image, releasing its memory. This is an image-generation command, not a persistent HTTP server or a Claude Code chat model.

Source: [MFLUX Qwen documentation](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/qwen/README.md).

## Persistent local API

Install the server dependencies into the MFLUX tool environment:

```bash
uv tool install mflux==0.19.1 --with fastapi --with uvicorn
bash scripts/qwen-image-server.sh
```

Open http://127.0.0.1:8001/docs for interactive API documentation. Startup loads the model once; `/health` returns 503 until ready, then 200. The server has one MLX worker thread and accepts up to eight waiting requests. It binds only to loopback and has no authentication; do not expose it to other hosts as configured.

```bash
curl http://127.0.0.1:8001/jobs \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A red cabin beside a lake","seed":42}'

# Substitute the returned id:
curl http://127.0.0.1:8001/jobs/JOB_ID
curl http://127.0.0.1:8001/jobs/JOB_ID/image -o /private/tmp/result.png
```

Defaults: 768×768, 30 steps, guidance 3.5, seed 42. Width/height must be multiples of 16 between 256 and 1024; steps are limited to 1–50. Only the 768×768 default has been exercised at full step count. Each response includes job timing and peak MLX memory after completion. `/health` includes memory sampled between jobs; it is not a live GPU memory gauge.

The model and text encoder stay loaded between jobs. Tiled VAE decoding and a 1 GB temporary MLX cache limit reduce working memory; prompt embeddings are cleared after each request. This intentionally avoids the CLI callback that unloads components.

Images are saved in `local_image/data/`; logs are in `local_image/server.log`. Images remain until manually deleted. Job metadata is in memory (latest 100 completed/failed jobs) and resets on restart; queued/running jobs are not resumed after a crash. No automatic startup is installed. Stop the foreground server with Ctrl+C, or signal the PID in `local_image/server.pid` after checking it belongs to this server.

Validated on this Mac, 2026-09-09: a 768×768, 30-step request completed in 88.29 seconds including encoding/decoding/save, peaking at 35.48 GB MLX memory. A different-prompt 256×256, two-step smoke request queued behind it and completed using the same model. Health remained responsive during generation; oversized requests returned HTTP 422; the PNG download succeeded. These timings are a single-run observation, not a benchmark. Readiness reported approximately 27.6 GB model memory before generation.
