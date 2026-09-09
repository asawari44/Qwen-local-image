# Qwen Image — local Mac studio

A simple browser UI and FastAPI server for **Qwen-Image-2512**, running on Apple Silicon through MFLUX/MLX. Enter a prompt, generate an image, preview it, and download the PNG. Everything runs locally after the model download.

## Start

Requirements: Apple Silicon Mac, enough memory (tested on a 64 GB Mac), and `uv`.

```bash
cd ~/github/qwen-image
uv tool install mflux==0.19.1 --with fastapi==0.141.1 --with uvicorn==0.52.4
bash scripts/qwen-image-server.sh
```

Open **http://127.0.0.1:8001/**. The page shows when the model is ready. The server remains in the foreground; stop it with Ctrl+C. No automatic startup is installed. Do not run multiple server instances: each loads its own model.

The model is `mflux-community/qwen-image-2512-mflux-q4`. Its approximately 27.6 GB download is cached under `~/.cache/huggingface/hub/`. The existing download is reused. No model weights are stored in this project.

[Example image](./local_image/data/trump.png)

## Usage

Enter a prompt, choose a square image size, steps, and seed, then click **Generate image**. The default is 768×768 at 30 steps. The UI displays the result and provides a PNG download. Requests run one at a time; up to eight requests may wait in the server queue. Reloading the page in the same tab resumes tracking the last job while the server remains running.

Images: `local_image/data/`. Logs: `local_image/server.log`. PID: `local_image/server.pid`. Images remain on disk until manually removed. Job status is in memory and resets when the server restarts; completed image files survive. The last 100 completed/failed jobs are retained in memory.

## API

Interactive API documentation: http://127.0.0.1:8001/docs

```bash
curl http://127.0.0.1:8001/jobs \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A cabin beside a lake at sunrise","seed":42}'

# Replace JOB_ID with the response id:
curl http://127.0.0.1:8001/jobs/JOB_ID
curl http://127.0.0.1:8001/jobs/JOB_ID/image -o image.png
curl http://127.0.0.1:8001/health
```

Health returns 503 during loading and 200 when ready. Invalid inputs return 422; a full queue returns 429. The server listens only on loopback with no authentication. Keep it local as configured.

## How it works

One FastAPI process serves the UI and jobs API. A dedicated worker thread loads the Qwen model once, then generates sequentially with MLX. Tiled VAE decoding and a 1 GB temporary cache limit control working memory; prompt embeddings are cleared after each job. The language-model `mlx_lm.server` is not used.

On the tested Mac, a 768×768 image at 30 steps took **88.29 seconds**, with **35.48 GB peak MLX memory**. A second, different-prompt request succeeded without reloading the model. The loaded model alone uses about 27.6 GB. These are individual observations, not performance guarantees; the 1024×1024 option has not been benchmarked.

## Project files

- `local_image/server.py`: API, bounded queue, persistent model worker.
- `local_image/index.html`: self-contained UI, no frontend build or external assets.
- `scripts/qwen-image-server.sh`: start the server and log output.
- `scripts/qwen-image-local.sh`: generate one image from the terminal and exit.
- `requirements.txt`: dependency versions used for this setup.
- `DETAILS.md`: implementation and validation notes.

Based on [MFLUX](https://github.com/mflux-community/mflux) and [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512).

## Live progress and logs

The UI displays actual completed denoising steps, plus encoding, decoding, and saving phases. The expandable logs panel refreshes every two seconds and shows the latest 120 lines of `local_image/server.log`, including MFLUX progress and server errors. Scroll up to inspect older entries without automatic scrolling; return to the bottom to follow new output. HTTP access logging is disabled to keep polling requests out of the panel. The read-only `/logs` endpoint returns a bounded tail of that fixed log file.
