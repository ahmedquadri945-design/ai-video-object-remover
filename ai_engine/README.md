# AI Vision & Inpainting Engine

High-performance, flow-guided video object removal engine powered by **SAM 2 (Segment Anything Model 2)** and **ProPainter (Dual-Domain Flow-Guided Video Inpainting)**, featuring zero-setup optical flow fallbacks for CPU/local environments.

---

## 🚀 Engine Capabilities

1. **`MaskProcessor`**:
   - Decodes Base64 canvas masks (PNG/Alpha) and vector brush stroke coordinates.
   - Elliptical morphological dilation (`cv2.dilate` with `cv2.MORPH_ELLIPSE`) to completely eliminate edge boundary halos and color bleed.
   - Mask smoothing and multi-frame temporal voting stabilization to eliminate mask edge jitter.

2. **`SAM2Tracker`**:
   - Segment Anything Model 2 video predictor wrapper.
   - Interactive prompt options: positive/negative foreground points, bounding boxes, or initial brush masks.
   - Bidirectional temporal mask propagation across all video frames.
   - **Zero-Setup Optical Flow Fallback**: Uses Farneback dense optical flow and GrabCut when SAM 2 weights or dependencies are not loaded.

3. **`ProPainterInpainter`**:
   - Temporal flow-guided video inpainting using recurrent flow completion and dual-domain attention.
   - **Chunked Sub-Video Inference**: Splits video into chunks (50–80 frames) with temporal linear cross-fade blending to prevent GPU VRAM spikes and out-of-memory errors on consumer GPUs (RTX 3060/4060/4070/4080).
   - **FP16 Mixed Precision**: Accelerated inference with 50% memory savings.
   - **Temporal Flow Local Fallback**: Seamless multi-frame bidirectional optical flow pixel borrowing with Navier-Stokes / Fast-Marching hole filling for CPU/local environments.

4. **`VideoObjectRemovalPipeline`**:
   - Master pipeline coordinating ingestion, segmentation, dilation, inpainting, and real-time progress callbacks for desktop and web interfaces.

---

## 📦 Directory Structure

```text
ai_engine/
├── __init__.py                # Package exports
├── mask_processor.py          # Mask decoding, dilation, and temporal smoothing
├── tracker_sam2.py            # SAM 2 tracking & optical flow fallback
├── inpainter_propainter.py    # ProPainter wrapper & temporal flow fallback
├── pipeline.py                # End-to-end master pipeline coordinator
└── README.md                  # Documentation and weight setup guide
```

---

## 🛠️ Environment Setup

### 1. PyTorch & Core Dependencies

Install PyTorch with CUDA support (or CPU-only if no GPU is present):

```bash
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core computer vision & image libraries:
pip install opencv-python pillow numpy
```

### 2. Segment Anything 2 (SAM 2) Installation

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

### 3. ProPainter Installation

```bash
git clone https://github.com/sczhou/ProPainter.git
cd ProPainter
pip install -r requirements.txt
```

---

## 📥 Model Weights Download Links

Store checkpoints in a designated directory (e.g. `weights/` or `checkpoints/`):

### 1. SAM 2 Checkpoints (Meta AI)

Choose the model checkpoint suited to your GPU memory:

| Model | Checkpoint File | Size | Download Link |
| :--- | :--- | :--- | :--- |
| **SAM 2 Hiera Large** (Recommended) | `sam2_hiera_large.pt` | ~898 MB | [Download](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt) |
| **SAM 2 Hiera Base+** | `sam2_hiera_base_plus.pt` | ~324 MB | [Download](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt) |
| **SAM 2 Hiera Small** | `sam2_hiera_small.pt` | ~184 MB | [Download](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt) |
| **SAM 2 Hiera Tiny** | `sam2_hiera_tiny.pt` | ~156 MB | [Download](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt) |

**Quick Download (PowerShell / Windows):**
```powershell
New-Item -ItemType Directory -Force -Path "./weights/sam2"
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt" -OutFile "./weights/sam2/sam2_hiera_large.pt"
```

### 2. ProPainter Checkpoints

ProPainter requires three pretrained models stored together in a weights folder (e.g., `weights/propainter/`):

| Model Name | Filename | Download URL |
| :--- | :--- | :--- |
| **ProPainter Inpainter** | `ProPainter.pth` | [HuggingFace / Release](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth) |
| **Recurrent Flow Completion** | `recurrent_flow_completion.pth` | [HuggingFace / Release](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth) |
| **RAFT Optical Flow** | `raft-things.pth` | [HuggingFace / Release](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth) |

**Quick Download (PowerShell / Windows):**
```powershell
New-Item -ItemType Directory -Force -Path "./weights/propainter"
Invoke-WebRequest -Uri "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth" -OutFile "./weights/propainter/ProPainter.pth"
Invoke-WebRequest -Uri "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth" -OutFile "./weights/propainter/recurrent_flow_completion.pth"
Invoke-WebRequest -Uri "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth" -OutFile "./weights/propainter/raft-things.pth"
```

---

## 💡 Quick Start Usage

### 1. End-to-End Object Removal from File

```python
from ai_engine import VideoObjectRemovalPipeline

# Initialize pipeline (works out of the box with automatic fallback if weights are not yet present)
pipeline = VideoObjectRemovalPipeline(
    sam2_checkpoint="weights/sam2/sam2_hiera_large.pt",
    propainter_weights_dir="weights/propainter",
    dilation_radius=6,   # Expands mask boundary to remove edge halo bleed
    chunk_size=60,       # Keeps VRAM under ~6-8 GB
    use_fp16=True,       # GPU mixed precision
)

# Callback to track progress in real-time
def on_progress(stage: str, step: int, total: int, message: str):
    percent = (step / total * 100) if total > 0 else 0
    print(f"[{stage.upper()}] {percent:5.1f}% - {message}")

# Process video using point prompt on keyframe 0:
result = pipeline.process_video_file(
    input_video_path="input_video.mp4",
    output_video_path="output_removed.mp4",
    keyframe_idx=0,
    points=[(320, 240)],   # Foreground point (x, y)
    labels=[1],            # 1=Foreground, 0=Background
    dilation_radius=8,
    progress_callback=on_progress,
)

print(f"Processed in {result['stats']['total_time_seconds']:.2f} seconds.")
print(f"Tracker: {result['stats']['tracker_mode']}")
print(f"Inpainter: {result['stats']['inpainter_mode']}")
```

### 2. Using Base64 Canvas Mask from Web Frontend

```python
from ai_engine import VideoObjectRemovalPipeline

pipeline = VideoObjectRemovalPipeline()

# base64_mask from HTML5 canvas: 'data:image/png;base64,iVBORw0KGgo...'
result = pipeline.remove_object(
    frames=frame_list,
    keyframe_idx=15,
    initial_mask=base64_mask,
    dilation_radius=6,
    progress_callback=lambda stage, step, total, msg: print(f"{stage}: {msg}"),
)

inpainted_frames = result["inpainted_frames"]
```

### 3. Individual Component Usage

#### Mask Processing & Edge Dilation
```python
from ai_engine import MaskProcessor
import cv2

processor = MaskProcessor(default_dilation_radius=6)

# Dilate with elliptical structuring element to avoid rectangular boundary artifacts
dilated_mask = processor.dilate_mask(binary_mask, radius=8)

# Smooth mask contours
smoothed_mask = processor.smooth_mask(dilated_mask, blur_kernel=5)
```

#### Interactive SAM 2 Tracking
```python
from ai_engine import SAM2Tracker

tracker = SAM2Tracker(checkpoint_path="weights/sam2/sam2_hiera_large.pt")

# Track object forward and backward
masks = tracker.track_video(
    frames=video_frames,
    keyframe_idx=0,
    points=[(250, 180), (280, 210)],
    labels=[1, 1],
)
```

---

## ⚡ Fallback & Zero-Weight Mode

If you run the pipeline without downloading any weights or on a CPU machine:
- **`SAM2Tracker`** automatically switches to `OpticalFlowTrackerFallback`, using bidirectional Farneback dense optical flow and GrabCut.
- **`ProPainterInpainter`** automatically switches to `TemporalFlowInpainterFallback`, using multi-frame bidirectional optical flow pixel borrowing with Navier-Stokes hole filling.
- The pipeline functions smoothly without throwing exceptions, enabling immediate testing, development, and unit test execution!
