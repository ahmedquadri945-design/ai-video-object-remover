import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import (
    DATA_DIR,
    UPLOADS_DIR,
    PROCESSED_DIR,
    TEMP_DIR,
    FFMPEG_PATH,
    FFPROBE_PATH,
    HOST,
    PORT,
    CORS_ORIGINS,
)
from routes.video_routes import router as video_router
from routes.websocket_routes import router as ws_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("video-object-remover-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    logger.info("Initializing Video Object Remover Backend...")
    for folder in [DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, TEMP_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    if FFMPEG_PATH:
        logger.info(f"FFmpeg binary detected at: {FFMPEG_PATH}")
    else:
        logger.warning(
            "FFmpeg was not detected on system PATH. OpenCV fallback will be used for video encoding (without audio)."
        )

    yield

    # Shutdown tasks
    logger.info("Shutting down Video Object Remover Backend...")


app = FastAPI(
    title="AI Video Object Remover API",
    description="Backend service for uploading videos, interactive keyframe mask selection, spatio-temporal tracking, inpainting, and audio remuxing.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API and WebSocket routes
app.include_router(video_router)
app.include_router(ws_router)

# Mount static media directory for direct access to uploaded/processed media
app.mount("/media", StaticFiles(directory=str(DATA_DIR)), name="media")

# Serve Frontend Web Application (http://127.0.0.1:8000)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if (FRONTEND_DIR / "dist").exists():
    FRONTEND_DIR = FRONTEND_DIR / "dist"

INDEX_HTML = FRONTEND_DIR / "index.html"


@app.get("/")
async def serve_index():
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    return {
        "service": "AI Video Object Remover API",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0",
    }


# Mount frontend static assets (styles.css, app.js, fonts, etc.)
if FRONTEND_DIR.exists():
    logger.info(f"Serving static frontend files from: {FRONTEND_DIR}")
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend_assets")


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "ffmpeg_available": bool(FFMPEG_PATH),
        "ffmpeg_path": FFMPEG_PATH,
        "uploads_dir": str(UPLOADS_DIR),
        "processed_dir": str(PROCESSED_DIR),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
