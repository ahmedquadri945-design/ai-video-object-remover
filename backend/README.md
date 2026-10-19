# AI Video Object Remover - Backend Service

High-performance FastAPI backend service providing video uploading, preview extraction, interactive mask segmentation, spatio-temporal tracking, video inpainting, and audio remuxing.

## Features

- **Video Upload & Metadata Extraction**: Extracts resolution, frame count, FPS, aspect ratio, duration, and keyframe preview.
- **Keyframe Preview Generation**: First-frame capture for the frontend canvas to draw masks or click points.
- **Spatio-temporal Inpainting**: Optical-flow object tracking and Telea/Navier-Stokes background synthesis.
- **Audio Remuxing**: Extracts original audio tracks with FFmpeg and remuxes into clean H.264 MP4 videos.
- **Asynchronous Job Queue**: Tracks job progress (0% to 100%) and stages (`uploading`, `tracking`, `inpainting`, `remuxing`, `completed`, `failed`).
- **Real-time WebSocket Streaming**: Live progress broadcasts to frontend clients via `/ws/{job_id}`.
- **Direct Streaming & Download**: Serves processed MP4 files for direct web playback and browser downloads.

## Directory Structure

```
backend/
├── config.py                 # Paths, environment configs, FFmpeg detection
├── main.py                   # FastAPI app entry point with CORS and static mounts
├── requirements.txt          # Python dependencies
├── routes/
│   ├── __init__.py
│   ├── video_routes.py       # REST API endpoints (/upload, /process, /status, /download)
│   └── websocket_routes.py   # WebSocket endpoint (/ws/{job_id})
└── services/
    ├── __init__.py
    ├── inpaint_service.py    # Mask parsing, tracking & inpainting pipeline
    ├── job_manager.py        # Asynchronous job tracking & WebSocket dispatching
    └── video_service.py      # OpenCV & FFmpeg utilities
```

## Quick Start

### 1. Install Dependencies
```bash
cd backend
python -m pip install -r requirements.txt
```

### 2. Start the Server
```bash
python main.py
```
Or with Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. API Documentation
Open your browser to:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
