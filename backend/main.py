from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.agent import DeepfakeInvestigator
from backend.detection_engine import ALLOWED_EXTENSIONS, DetectionEngine

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deepfake_platform")

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", BASE_DIR / "backend" / "uploads"))
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE_BYTES", 200 * 1024 * 1024))  # 200MB
KEEP_UPLOADS = os.environ.get("KEEP_UPLOADS", "false").lower() == "true"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#  App setup
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="DeepFake Detection Platform API",
    description="Multi-model ensemble deepfake detection with an AI investigator agent.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets (frontend/assets) if present
assets_dir = FRONTEND_DIR / "assets"
if assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
else:
    logger.warning("Assets directory '%s' not found — skipping static mount.", assets_dir)

# --------------------------------------------------------------------------- #
#  Engine & agent (singletons, loaded once at startup)
# --------------------------------------------------------------------------- #
engine = DetectionEngine()
agent = DeepfakeInvestigator()


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _safe_suffix(filename: str) -> str:
    """Return a lower-cased, validated file extension (with leading dot)."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or '(none)'}'. Allowed: {allowed}",
        )
    return suffix


async def _save_upload(file: UploadFile) -> tuple[Path, int]:
    """
    Stream an UploadFile to disk under a random name (preserving extension),
    enforcing a maximum size limit. Returns (path, size_bytes).

    Raises HTTPException(413) if the file exceeds MAX_UPLOAD_SIZE.
    """
    suffix = _safe_suffix(file.filename)
    dest_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    size = 0
    chunk_size = 1024 * 1024  # 1MB

    try:
        with open(dest_path, "wb") as out_file:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    out_file.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large. Maximum allowed size is "
                            f"{MAX_UPLOAD_SIZE // (1024 * 1024)}MB."
                        ),
                    )
                out_file.write(chunk)
    except HTTPException:
        raise
    except Exception:
        dest_path.unlink(missing_ok=True)
        logger.exception("Failed to save uploaded file '%s'", file.filename)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    if size == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return dest_path, size


# --------------------------------------------------------------------------- #
#  API routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health_check():
    """Health/status endpoint — also reports whether real models are loaded."""
    return {
        "status": "ok",
        "message": "Deepfake Detection Platform API is running",
        "engine_mode": engine.mode,
        "device": engine.device,
        "agent_mode": "gemini" if agent._gemini_available else "rule-based",
        "max_upload_mb": MAX_UPLOAD_SIZE // (1024 * 1024),
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
    }


@app.post("/api/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    """
    Detect whether an uploaded image or video shows signs of manipulation.

    - Validates file extension and size before saving.
    - Saves the file to a temporary location.
    - Runs the detection engine (real models if available, otherwise a
      deterministic simulation).
    - Cleans up the saved file afterwards unless KEEP_UPLOADS=true.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    saved_path: Path | None = None
    try:
        saved_path, size_bytes = await _save_upload(file)
        logger.info(
            "Received '%s' (%.2f MB) — running detection (mode=%s)",
            file.filename,
            size_bytes / (1024 * 1024),
            engine.mode,
        )

        result = engine.detect(str(saved_path), file.filename)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        # Bad/unsupported input from the engine (e.g. unreadable video)
        logger.warning("Detection rejected for '%s': %s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during detection for '%s'", file.filename)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while analyzing the file.",
        )
    finally:
        if saved_path and saved_path.exists() and not KEEP_UPLOADS:
            try:
                saved_path.unlink()
            except OSError:
                logger.warning("Could not delete temporary file '%s'", saved_path)


@app.post("/api/chat")
async def chat_with_agent(request: Request):
    """Conversational endpoint for the AI Investigator agent."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")

    message = (data.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="'message' field is required.")

    context = data.get("context")

    try:
        response = agent.chat(message, context)
        return {"response": response}
    except Exception:
        logger.exception("Agent chat failed for message: %r", message)
        raise HTTPException(
            status_code=500, detail="The AI agent encountered an error. Please try again."
        )


# --------------------------------------------------------------------------- #
#  Static frontend serving (catch-all, must be last)
# --------------------------------------------------------------------------- #
@app.get("/{path:path}")
async def serve_static(path: str):
    """Serve the frontend SPA, falling back to index.html for unknown paths."""
    index_path = FRONTEND_DIR / "index.html"

    if not path:
        if not index_path.is_file():
            raise HTTPException(status_code=500, detail="Frontend not found.")
        return FileResponse(index_path)

    candidate = (FRONTEND_DIR / path).resolve()

    # Prevent path traversal outside the frontend directory
    if FRONTEND_DIR.resolve() not in candidate.parents and candidate != FRONTEND_DIR.resolve():
        raise HTTPException(status_code=404, detail="Not found")

    if candidate.is_file():
        return FileResponse(candidate)

    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="Frontend not found.")
    return FileResponse(index_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
