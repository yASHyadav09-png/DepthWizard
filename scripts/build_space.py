"""Build the Hugging Face Space bundle for DepthWizard (Vercel + HF ZeroGPU deployment).

Produces `space_dist/`, a self-contained copy of the parts of this repo the Space needs:

    space_dist/
        app.py              new Gradio-SDK entrypoint: mounts the UNCHANGED FastAPI app
                             (routes/services/config, copied verbatim below) onto Gradio's
                             own ASGI app, so the REST API contract is byte-for-byte the
                             same as the local backend. Only the actual model-inference
                             call is wrapped with @spaces.GPU (via monkeypatch, so the
                             copied service code itself is never touched).
        requirements.txt    Space-specific deps (adds `spaces`, drops the +cu128 wheel
                             index since HF's ZeroGPU image supplies its own CUDA build)
        README.md           required HF Space metadata header (sdk, hardware, etc.)
        depthwizard/        verbatim copy of the top-level `depthwizard/` package
        backend/dwapp/      copy of `backend/app/`, with internal `app.` imports rewritten
                             to `dwapp.` (the package is renamed only to avoid colliding
                             with the required top-level `app.py` entrypoint filename).
                             `main.py` is NOT copied -- its FastAPI wiring is re-created
                             directly in the new `app.py` instead.

This script only ever WRITES into space_dist/ (deleted and rebuilt each run); nothing in
backend/ or depthwizard/ is modified.

Usage:  python scripts/build_space.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_APP = ROOT / "backend" / "app"
DEPTHWIZARD = ROOT / "depthwizard"
OUT = ROOT / "space_dist"

# Anchored to import statements only -- never rewrites a variable/attribute access like
# `app.mount(...)` (verified: no such usage exists outside main.py, which we don't copy).
IMPORT_RE = re.compile(r"^(\s*)(from|import)\s+app\b", re.MULTILINE)


def rewrite_imports(text: str) -> str:
    def sub(m: re.Match) -> str:
        return f"{m.group(1)}{m.group(2)} dwapp" if m.group(2) == "import" else f"{m.group(1)}from dwapp"
    # `from app.x import y` / `from app import y` -> `from dwapp...`
    text = re.sub(r"^(\s*)from\s+app\b", r"\1from dwapp", text, flags=re.MULTILINE)
    # bare `import app` / `import app.x` (none currently exist, kept for safety)
    text = re.sub(r"^(\s*)import\s+app\b", r"\1import dwapp", text, flags=re.MULTILINE)
    return text


def copy_py_tree(src: Path, dst: Path, rewrite: bool) -> int:
    n = 0
    for f in src.rglob("*.py"):
        if "__pycache__" in f.parts:
            continue
        rel = f.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        text = f.read_text(encoding="utf-8")
        if rewrite:
            text = rewrite_imports(text)
        out.write_text(text, encoding="utf-8")
        n += 1
    return n


SPACE_README = """---
title: DepthWizard
emoji: \U0001F5FA️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: mit
---

# DepthWizard API (private backend for the Vercel frontend)

Single-view height estimation (SIH 2026, PS 26175 / ISRO). This Space hosts the FastAPI
backend (mounted inside the required Gradio app) behind a ZeroGPU-accelerated inference
call. It is called by the DepthWizard frontend; visiting this page directly shows a
minimal status UI, not the full app.

REST API: see `/api/health`, `/api/process`, `/api/results/{job_id}`,
`/api/results/{job_id}/validate`, `/api/jobs` (identical contract to the local backend,
see ../docs/pipeline.md in the main repo).
"""

SPACE_REQUIREMENTS = """# Space-specific: HF's ZeroGPU image supplies its own CUDA-matched torch build, so we
# request a plain (non +cu128) torch/torchvision here, plus `spaces` for the GPU decorator.
spaces
torch
torchvision
gradio==6.17.3  # pinned to match the local test (see scripts/build_space.py)
huggingface_hub
transformers>=4.45,<5
h5py
numpy
scipy
matplotlib
pyyaml
pandas
scikit-learn
rasterio
pyproj
pystac-client
planetary-computer
python-multipart
pydantic>=2.7
pillow
"""

APP_PY = '''"""DepthWizard HF Space entrypoint (Vercel + ZeroGPU deployment).

Mounts the unchanged FastAPI app (dwapp/, a renamed copy of backend/app/) onto the
FastAPI/ASGI app Gradio creates, so the REST API contract is identical to running the
local backend directly (same routes, same request/response shapes, same error format).
A minimal gr.Blocks UI is required by the Gradio Space SDK and doubles as a human-visible
status page; the real client is the DepthWizard React frontend calling the REST API.

Model weights: by default downloaded from a private HF model repo at startup
(DW_MODEL_REPO). For local testing without any Hub upload, set DW_RUN_DIR to a local
run directory instead (see README "Local test" section) and leave DW_MODEL_REPO unset.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# `dwapp` lives at ./backend/dwapp (mirrors backend/app/ in the main repo, so its internal
# BASE_DIR/REPO_ROOT path math -- e.g. OUTPUT_DIR default, RUN_DIR default -- resolves the
# same way); Space root (this file's directory) is not on sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
logger = logging.getLogger("depthwizard.space")

# ---- resolve model weights BEFORE importing dwapp.config (RUN_DIR is read at import time)
if not os.environ.get("DW_RUN_DIR") and os.environ.get("DW_MODEL_REPO"):
    from huggingface_hub import snapshot_download
    repo_id = os.environ["DW_MODEL_REPO"]
    logger.info("Downloading model weights from the private repo %s ...", repo_id)
    run_dir = snapshot_download(repo_id=repo_id, repo_type="model")
    os.environ["DW_RUN_DIR"] = run_dir
    logger.info("Model weights ready at %s", run_dir)
elif os.environ.get("DW_RUN_DIR"):
    logger.info("Using local run dir %s (DW_MODEL_REPO not used).", os.environ["DW_RUN_DIR"])
else:
    raise RuntimeError("Set DW_MODEL_REPO (Hub download) or DW_RUN_DIR (local test) before starting.")

# outputs must be writable on the Space's ephemeral disk
os.environ.setdefault("DW_OUTPUT_DIR", "/tmp/dw_outputs")
# Gradio itself reserves "/static/{path:path}" for its own UI assets (verified against
# gradio==6.17.3: registered before anything we add, so it silently shadows ours there) --
# use a prefix Gradio doesn't claim.
os.environ.setdefault("DW_STATIC_URL_PREFIX", "/dw-static")
# same-origin browsing (this page) + the Vercel frontend, overridable at Space startup
os.environ.setdefault(
    "DW_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,"
    + os.environ.get("DW_EXTRA_CORS_ORIGINS", ""),
)

import gradio as gr  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.middleware import Middleware  # noqa: E402

from dwapp.api.routes import router  # noqa: E402
from dwapp.config import settings  # noqa: E402
from dwapp.services.height_estimator import HeightEstimator, get_height_estimator  # noqa: E402
from dwapp.utils.errors import DepthWizardError  # noqa: E402

# ---- ZeroGPU: wrap ONLY the actual inference call, without touching the copied service
# code. Effect-free outside a real ZeroGPU Space (HF docs), so this is safe to import and
# run locally / on CPU-only hosts unchanged.
#
# IMPORTANT (found by deploying and reading the worker traceback -- PicklingError: cannot
# pickle '_thread.lock' object): @spaces.GPU does not just switch CUDA context in-process --
# it sends the CALL ARGUMENTS across a queue to a separate worker process (which re-imports
# this module and therefore already has its own copy of any MODULE-LEVEL object, matching
# HF's own "load models at module level" guidance). Wrapping an *instance method*
# (HeightEstimator.predict) means the implicit `self` becomes part of those pickled
# arguments -- and HeightEstimator/NDSMPredictor hold a threading.Lock (for the unrelated,
# already-serialized-by-ZeroGPU concern of concurrent local requests), which cannot be
# pickled at all. Fix: decorate a plain module-level function whose only argument is the
# picklable rgb array; it reaches the already-loaded model via a module-level reference
# instead of a bound `self`, exactly like HF's own documented pattern.
try:
    import spaces

    _estimator = get_height_estimator()
    _estimator.load()  # eager (not just PRELOAD_MODEL-conditional): must exist at module
                       # level before spaces.GPU wraps anything that references it

    @spaces.GPU(duration=60)
    def _zerogpu_predict(rgb):
        return _estimator._predictor.predict(rgb)  # bypasses HeightEstimator's own lock;
                                                    # ZeroGPU serializes GPU calls itself

    def _predict_via_zerogpu(self, rgb):
        t0 = time.perf_counter()
        h = _zerogpu_predict(rgb)
        return h, (time.perf_counter() - t0) * 1000.0

    HeightEstimator.predict = _predict_via_zerogpu
    logger.info("ZeroGPU decorator attached (module-level function, not an instance method).")
except ImportError:
    logger.warning("`spaces` package not available; running without ZeroGPU (fine for local tests).")


def _status_text() -> str:
    info = get_height_estimator().info()
    return (
        f"DepthWizard backend is running.\\n\\n"
        f"Model: {info['model_label']}\\n"
        f"Device: {info['device_label']}\\n"
        f"Loaded: {info['loaded']}\\n\\n"
        "This Space is a backend API for the DepthWizard web app, not an interactive "
        "demo page. See /api/health and /docs on this Space's URL for the REST API."
    )


with gr.Blocks(title="DepthWizard API") as demo:
    gr.Markdown("## DepthWizard\\nBackend API for the DepthWizard frontend (SIH 2026, PS 26175).")
    status = gr.Textbox(label="Status", value=_status_text, every=None)
    gr.Button("Refresh status").click(_status_text, outputs=status)


async def depthwizard_error_handler(_, exc: DepthWizardError) -> JSONResponse:
    if exc.status_code >= 500:
        logger.exception("Pipeline failure: %s", exc.message)
    else:
        logger.warning("%s: %s", exc.code, exc.message)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "detail": exc.message})


async def validation_error_handler(_, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "invalid_request",
            "detail": "Expected a multipart form with an 'image' file field (and optional numeric 'gsd_m').",
            "errors": exc.errors(),
        },
    )


# IMPORTANT (verified against gradio==6.17.3, see scripts/build_space.py comment history):
# `Blocks.launch()` builds a FRESH FastAPI app internally (`self.app = App.create_app(...)`)
# and only assigns it to `demo.app` at that point -- anything mounted on `demo.app` BEFORE
# `launch()` runs is attached to a throwaway object and silently never served. Middleware and
# exception handlers additionally can't be registered AFTER launch() either (Starlette freezes
# its middleware stack once the app has served its first request) -- both must go through
# `app_kwargs` instead. Routes (`include_router`/`mount`) CAN be added after launch(), since
# Starlette's router is checked live per-request. Verified with a minimal reproduction before
# relying on it here; if a future gradio version changes this, the local test in this file's
# README ("Local test") will catch it immediately (every /api/* call would 404).
demo.queue().launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    prevent_thread_lock=True,
    ssr_mode=False,
    app_kwargs={
        "middleware": [
            Middleware(
                CORSMiddleware,
                allow_origins=[o.strip() for o in settings.CORS_ORIGINS if o.strip()],
                allow_credentials=False,
                allow_methods=["*"],
                allow_headers=["*"],
            ),
        ],
        "exception_handlers": {
            DepthWizardError: depthwizard_error_handler,
            RequestValidationError: validation_error_handler,
        },
    },
)

fastapi_app = demo.app  # now the REAL, already-serving app (see note above)
Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
fastapi_app.mount(settings.STATIC_URL_PREFIX, StaticFiles(directory=str(settings.OUTPUT_DIR)), name="static")
fastapi_app.include_router(router)

if settings.PRELOAD_MODEL:
    try:
        get_height_estimator().load()
    except DepthWizardError as exc:
        logger.error("Model preload failed (will retry on first request): %s", exc.message)

if __name__ == "__main__":
    demo.block_thread()  # launch() already started serving (prevent_thread_lock=True); just wait
'''


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    n_dw = copy_py_tree(DEPTHWIZARD, OUT / "depthwizard", rewrite=False)
    n_app = 0
    for f in BACKEND_APP.rglob("*.py"):
        if "__pycache__" in f.parts or f.name == "main.py":
            continue
        rel = f.relative_to(BACKEND_APP)
        out = OUT / "backend" / "dwapp" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rewrite_imports(f.read_text(encoding="utf-8")), encoding="utf-8")
        n_app += 1

    (OUT / "app.py").write_text(APP_PY, encoding="utf-8")
    (OUT / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"  depthwizard/: {n_dw} files (verbatim)")
    print(f"  backend/dwapp/: {n_app} files (app. -> dwapp. imports rewritten, main.py excluded)")
    print("  app.py, requirements.txt, README.md")


if __name__ == "__main__":
    main()
