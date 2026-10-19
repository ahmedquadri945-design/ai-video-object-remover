# AI Video Object Remover - Web Application

A full-stack, state-of-the-art Web Application for **AI Video Object Removal** (rebuilding the functionality of Filmora AI Object Remover / Runway Inpainting).

---

## 🌟 Overview & Features

- **Interactive Web Studio**:
  - Drag-and-drop video upload supporting MP4, MOV, MKV, AVI, and WebM.
  - Interactive HTML5 Canvas overlay synchronized with video pixel dimensions.
  - **Smart Brush**: Paint directly over unwanted objects, people, watermarks, or text.
  - **SAM 2 Point Prompting**: Click positive (`+`) points on foreground objects and negative (`-`) points on background regions.
  - Timeline scrubber with frame-by-frame navigation, timecodes, and keyframe markers.
- **AI Tracking & Segmentation (Meta SAM 2)**:
  - Propagates object masks across the entire video timeline from a single keyframe prompt.
  - Built-in optical flow fallback for immediate out-of-the-box operation without GPU/weights.
- **Flow-Guided Temporal Inpainting (ProPainter)**:
  - Recurrent optical flow completion + dual-domain temporal feature synthesis.
  - Eliminates flickering across frames.
  - Automatic morphological mask dilation (`cv2.dilate` with ellipse structuring element) to eliminate edge halos.
  - Chunked sub-video inference (50–80 frame windows) with FP16 support to prevent consumer GPU OOM.
- **Audio Preservation & FFmpeg Integration**:
  - Automatically isolates the original audio track and re-muxes it with the cleaned video stream.
- **Before / After Comparison Slider**:
  - Interactive split-screen comparison slider allowing users to scrub between original and erased video in real time.
- **Real-Time WebSocket Feed**:
  - Live progress percentage and console logs streamed to the frontend in real time.

---

## 📁 Project Architecture

```
video-object-remover/
├── run.py                          # One-click Python launcher
├── start.bat                       # Windows one-click batch launcher
├── README.md                       # Comprehensive guide & architecture
├── frontend/                       # Interactive Web Studio UI
│   ├── index.html                  # Dark-mode Studio workspace
│   ├── styles.css                  # Modern UI theme with glassmorphism & neon accents
│   └── app.js                      # Canvas brush, video synchronization & API client
├── backend/                        # FastAPI REST & WebSocket Server
│   ├── main.py                     # App entrypoint & static mount
│   ├── config.py                   # Storage paths, FFmpeg detection & settings
│   ├── requirements.txt            # Python server dependencies
│   ├── routes/
│   │   ├── video_routes.py         # Upload, keyframe, process, download endpoints
│   │   └── websocket_routes.py     # Live progress streaming over WS
│   └── services/
│       ├── video_service.py        # FFmpeg frame extraction & audio re-muxing
│       ├── job_manager.py          # Asynchronous job queue & status manager
│       └── inpaint_service.py      # Background processing pipeline
└── ai_engine/                      # Computer Vision & Inpainting Engine
    ├── __init__.py                 # Clean package exports
    ├── mask_processor.py           # Stroke rasterization, dilation, contour cleaning
    ├── tracker_sam2.py             # Meta SAM 2 video predictor & optical flow fallback
    ├── inpainter_propainter.py     # ProPainter flow-guided inpainting & fallback
    ├── pipeline.py                 # End-to-end pipeline orchestrator
    └── README.md                   # Model weights setup & GPU tuning guide
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- FFmpeg installed and added to your system `PATH` (optional but recommended for high-speed encoding).

### 2. Install Dependencies
```bash
cd video-object-remover
pip install -r backend/requirements.txt
```

### 3. Launch the Application
Run the launcher:
```bash
python run.py
```
Or on Windows, double-click **`start.bat`**.

The web application will open automatically in your browser at:
👉 **`http://127.0.0.1:8000`**

---

## 🧠 Optional: Download Pretrained Weights for Deep Learning Mode

The application works **immediately out-of-the-box** using the built-in optical flow tracking and temporal inpainter fallback.

To enable full GPU-accelerated **Meta SAM 2** and **ProPainter**:

1. **Meta SAM 2**:
   ```bash
   pip install git+https://github.com/facebookresearch/segment-anything-2.git
   mkdir -p ai_engine/weights
   # Download SAM 2 Large checkpoint:
   curl -L -o ai_engine/weights/sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
   ```

2. **ProPainter**:
   ```bash
   # Download ProPainter weights into ai_engine/weights/:
   # Download ProPainter.pth, recurrent_flow_completion.pth, and raft-things.pth
   # from https://github.com/sczhou/ProPainter/releases
   ```
   *See [`ai_engine/README.md`](ai_engine/README.md) for full instructions.*
