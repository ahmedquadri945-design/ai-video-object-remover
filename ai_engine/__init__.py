"""
AI Vision & Video Inpainting Engine for Video Object Remover.

Provides high-performance modules for:
- Mask processing, polygon/brush stroke rasterization, base64 decoding, morphological dilation, and temporal smoothing.
- SAM 2 (Segment Anything Model 2) video object tracking with optical flow fallback.
- Flow-guided temporal video inpainting (ProPainter with chunked VRAM control and fast temporal optical flow fallback).
- End-to-end VideoObjectRemovalPipeline.
"""

from .mask_processor import MaskProcessor
from .tracker_sam2 import SAM2Tracker, OpticalFlowTrackerFallback
from .inpainter_propainter import ProPainterInpainter, TemporalFlowInpainterFallback
from .pipeline import VideoObjectRemovalPipeline

__all__ = [
    "MaskProcessor",
    "SAM2Tracker",
    "OpticalFlowTrackerFallback",
    "ProPainterInpainter",
    "TemporalFlowInpainterFallback",
    "VideoObjectRemovalPipeline",
]

__version__ = "1.0.0"
