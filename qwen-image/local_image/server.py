"""Single-process, loopback-only Qwen image API with one MLX worker thread."""
import asyncio
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent / "data"
MODEL = "mflux-community/qwen-image-2512-mflux-q4"
jobs = {}
pending = queue.Queue(maxsize=8)
lock = threading.Lock()
stop = threading.Event()
state = {"status": "loading", "model": MODEL}
log = logging.getLogger("uvicorn.error")


class Request(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=4000)
    width: int = Field(default=768, ge=256, le=1024, multiple_of=16)
    height: int = Field(default=768, ge=256, le=1024, multiple_of=16)
    steps: int = Field(default=30, ge=1, le=50)
    seed: int = Field(default=42, ge=0, le=2147483647)

    @field_validator("prompt")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Prompt must not be blank")
        return value


class Progress:
    job_id = None

    def call_before_loop(self, **kwargs):
        with lock:
            jobs[self.job_id].update(phase="denoising", step=0)
        log.info("Job %s: prompt encoded; starting denoising", self.job_id)

    def call_in_loop(self, t, latents, **kwargs):
        import mlx.core as mx
        mx.eval(latents)
        with lock:
            jobs[self.job_id].update(step=int(t)+1, active_memory_gb=round(mx.get_active_memory()/1e9, 2))

    def call_after_loop(self, **kwargs):
        with lock:
            jobs[self.job_id].update(phase="decoding")
        log.info("Job %s: denoising complete; decoding image", self.job_id)


def worker():
    try:
        import mlx.core as mx
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
        from mflux.models.common.vae.tiling_config import TilingConfig
        log.info("Loading Qwen image model from cache")
        mx.set_cache_limit(1_000_000_000)
        model = QwenImage(model_path=MODEL)
        progress = Progress()
        model.callbacks.register(progress)
        model.tiling_config = TilingConfig(vae_decode_tile_size=512)
        mx.eval(model.parameters())
        with lock:
            state.update(status="ready", active_memory_gb=mx.get_active_memory()/1e9)
        log.info("Qwen model ready")
    except Exception:
        log.exception("Model startup failed")
        with lock:
            state.update(status="failed", error="Model startup failed; see server.log")
        return
    while not stop.is_set():
        try:
            job_id, request = pending.get(timeout=0.5)
        except queue.Empty:
            continue
        started = time.time()
        with lock:
            jobs[job_id].update(status="running", started_at=started, phase="encoding", step=0)
        progress.job_id = job_id
        log.info("Job %s: encoding prompt (%sx%s, %s steps)", job_id, request.width, request.height, request.steps)
        try:
            mx.reset_peak_memory()
            result = model.generate_image(
                prompt=request.prompt, negative_prompt=request.negative_prompt,
                width=request.width, height=request.height,
                num_inference_steps=request.steps, seed=request.seed, guidance=3.5,
            )
            with lock:
                jobs[job_id].update(phase="saving")
            result.save(path=str(ROOT / f"{job_id}.png"))
            del result
            with lock:
                jobs[job_id].update(status="completed", phase="completed", image_url=f"/jobs/{job_id}/image")
        except Exception:
            log.exception("Generation failed for %s", job_id)
            with lock:
                jobs[job_id].update(status="failed", error="Generation failed; see server.log")
        finally:
            model.prompt_cache.clear()
            mx.clear_cache()
            with lock:
                jobs[job_id].update(finished_at=time.time(), elapsed_seconds=round(time.time()-started, 2),
                                   peak_mlx_memory_gb=round(mx.get_peak_memory()/1e9, 2))
                state.update(active_memory_gb=round(mx.get_active_memory()/1e9, 2))
                # Bound metadata retention; image files remain on disk.
                completed = [k for k,v in jobs.items() if v["status"] in ("completed", "failed")]
                for old in completed[:-100]:
                    del jobs[old]
            log.info("Job %s: %s in %.2fs; peak MLX %.2f GB", job_id, jobs[job_id]["status"], time.time()-started, mx.get_peak_memory()/1e9)
            pending.task_done()


@asynccontextmanager
async def lifespan(app):
    ROOT.mkdir(parents=True, exist_ok=True)
    thread = threading.Thread(target=worker, daemon=True, name="mlx-worker")
    thread.start()
    yield
    stop.set()
    await asyncio.to_thread(thread.join, 5)


app = FastAPI(title="Local Qwen Image", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def ui():
    return FileResponse(Path(__file__).with_name("index.html"), media_type="text/html")


@app.get("/health")
async def health():
    with lock:
        result = dict(state, queued=pending.qsize(), running=sum(j["status"] == "running" for j in jobs.values()))
    return JSONResponse(result, status_code=200 if result["status"] == "ready" else 503)


@app.post("/jobs", status_code=202)
async def submit(request: Request):
    with lock:
        if state["status"] != "ready":
            raise HTTPException(503, "Model is not ready")
        job_id = str(uuid4())
        job = dict(id=job_id, status="queued", phase="queued", step=0, submitted_at=time.time(), parameters=request.model_dump())
        jobs[job_id] = job
        try:
            pending.put_nowait((job_id, request))
        except queue.Full:
            del jobs[job_id]
            raise HTTPException(429, "Queue full; try again later")
        return dict(job)


@app.get("/jobs/{job_id}")
async def status(job_id: UUID):
    with lock:
        if str(job_id) not in jobs:
            raise HTTPException(404, "Unknown job; job history resets on restart")
        return dict(jobs[str(job_id)])


@app.get("/jobs/{job_id}/image")
async def image(job_id: UUID):
    path = ROOT / f"{job_id}.png"
    if not path.is_file():
        raise HTTPException(404, "Image not available yet")
    return FileResponse(path, media_type="image/png")


@app.get("/logs")
async def logs():
    # Fixed path and bounded tail: never accept a user-supplied filename.
    path = Path(__file__).with_name("server.log")
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell()-32768))
            tail = f.read().decode("utf-8", errors="replace")
        import re
        tail = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", tail)
        return {"lines": tail.replace("\r", "\n").splitlines()[-120:]}
    except FileNotFoundError:
        return {"lines": ["Log file is not available yet."]}
