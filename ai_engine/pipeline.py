"""
Master Video Object Removal Pipeline.

Coordinates:
1. Video Ingestion: Read video frames or image sequences.
2. Temporal Object Tracking & Segmentation: SAM 2 (Segment Anything Model 2) or Optical Flow.
3. Mask Post-Processing: Elliptical morphological dilation (halo removal), smoothing, temporal stabilization.
4. Temporal Video Inpainting: ProPainter with chunked sub-video processing or temporal flow fallback.
5. Export & Progress Reporting: Real-time callbacks for web/desktop UIs and MP4 video generation.
"""

from typing import List, Optional, Tuple, Union, Dict, Any, Callable
import os
import time
import logging
import numpy as np
import cv2

from .mask_processor import MaskProcessor
from .tracker_sam2 import SAM2Tracker
from .inpainter_propainter import ProPainterInpainter

logger = logging.getLogger(__name__)


class VideoObjectRemovalPipeline:
    """
    End-to-end AI pipeline for removing objects from videos seamlessly.
    """

    def __init__(
        self,
        sam2_checkpoint: Optional[str] = None,
        sam2_model_cfg: str = "sam2_hiera_l.yaml",
        propainter_weights_dir: Optional[str] = None,
        device: Optional[str] = None,
        dilation_radius: int = 6,
        temporal_mask_smoothing: bool = True,
        chunk_size: int = 60,
        use_fp16: bool = True,
        force_fallback: bool = False,
    ) -> None:
        """
        Initialize the Video Object Removal Pipeline.

        Args:
            sam2_checkpoint: Path to SAM 2 PyTorch checkpoint (.pt).
            sam2_model_cfg: SAM 2 architecture configuration file.
            propainter_weights_dir: Path to directory containing ProPainter weights.
            device: 'cuda', 'cpu', or None (auto-detects CUDA).
            dilation_radius: Radius for morphological dilation to eliminate boundary color bleeding.
            temporal_mask_smoothing: Whether to apply temporal moving-average smoothing to masks.
            chunk_size: Sub-video chunk length to manage GPU VRAM.
            use_fp16: Enable FP16 half-precision inpainting on GPU.
            force_fallback: Force local fallback modes for tracker and inpainter.
        """
        self.device = device
        self.dilation_radius = dilation_radius
        self.temporal_mask_smoothing = temporal_mask_smoothing

        # Initialize sub-components
        self.mask_processor = MaskProcessor(
            default_dilation_radius=dilation_radius,
            default_blur_kernel=3,
            temporal_window_size=3,
        )

        self.tracker = SAM2Tracker(
            checkpoint_path=sam2_checkpoint,
            model_cfg=sam2_model_cfg,
            device=device,
            force_fallback=force_fallback,
        )

        self.inpainter = ProPainterInpainter(
            weights_dir=propainter_weights_dir,
            chunk_size=chunk_size,
            chunk_overlap=5,
            use_fp16=use_fp16,
            device=device,
            force_fallback=force_fallback,
        )

    # =========================================================================
    # High-Level Execution Methods
    # =========================================================================

    def remove_object(
        self,
        frames: List[np.ndarray],
        keyframe_idx: int = 0,
        initial_mask: Optional[Union[np.ndarray, str]] = None,
        points: Optional[List[Tuple[float, float]]] = None,
        labels: Optional[List[int]] = None,
        box: Optional[Tuple[float, float, float, float]] = None,
        dilation_radius: Optional[int] = None,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full object removal pipeline on a list of in-memory video frames.

        Args:
            frames: List of BGR/RGB image arrays (H, W, 3) uint8.
            keyframe_idx: Frame index where interaction prompt was specified.
            initial_mask: Binary mask numpy array (H, W) or base64 canvas mask string.
            points: List of (x, y) prompt coordinates.
            labels: 1 for positive (foreground), 0 for negative (background).
            box: (x1, y1, x2, y2) bounding box prompt.
            dilation_radius: Custom dilation radius in pixels (overrides default).
            progress_callback: Callback `(stage, current_step, total_steps, message)`.

        Returns:
            Dict containing:
                - 'inpainted_frames': List[np.ndarray] of result video frames.
                - 'masks': List[np.ndarray] of dilated masks used for inpainting.
                - 'raw_masks': List[np.ndarray] of tracked masks before dilation.
                - 'stats': Dict with execution timings and metadata.
        """
        start_time = time.time()
        num_frames = len(frames)
        if num_frames == 0:
            raise ValueError("Input frames list is empty.")

        h, w = frames[0].shape[:2]

        def _notify(stage: str, step: int, total: int, msg: str):
            if progress_callback:
                progress_callback(stage, step, total, msg)

        # ---------------------------------------------------------------------
        # 1. Prompt Preparation
        # ---------------------------------------------------------------------
        _notify("preprocess", 0, 1, "Parsing interaction prompts...")
        seed_mask: Optional[np.ndarray] = None
        if isinstance(initial_mask, str):
            seed_mask = self.mask_processor.base64_to_mask(initial_mask, target_shape=(h, w))
        elif isinstance(initial_mask, np.ndarray):
            seed_mask = (initial_mask > 127).astype(np.uint8) * 255

        # ---------------------------------------------------------------------
        # 2. Tracking / Mask Propagation across frames
        # ---------------------------------------------------------------------
        _notify("tracking", 0, num_frames, "Tracking object across video frames...")

        def _track_cb(step: int, total: int, msg: str):
            _notify("tracking", step, total, msg)

        raw_masks = self.tracker.track_video(
            frames=frames,
            keyframe_idx=keyframe_idx,
            initial_mask=seed_mask,
            points=points,
            labels=labels,
            box=box,
            progress_callback=_track_cb,
        )

        tracking_time = time.time() - start_time

        # ---------------------------------------------------------------------
        # 3. Mask Post-Processing (Dilation & Temporal Smoothing)
        # ---------------------------------------------------------------------
        _notify("mask_processing", 0, num_frames, "Post-processing & dilating masks...")
        d_radius = self.dilation_radius if dilation_radius is None else dilation_radius

        processed_masks: List[np.ndarray] = []
        for i, m in enumerate(raw_masks):
            # Dilate with elliptical kernel to absorb edge halos & color bleed
            dilated = self.mask_processor.dilate_mask(m, radius=d_radius)
            # Smooth contour
            smoothed = self.mask_processor.smooth_mask(dilated, blur_kernel=3)
            processed_masks.append(smoothed)

        # Temporal stabilization
        if self.temporal_mask_smoothing:
            processed_masks = self.mask_processor.temporal_stabilize_masks(
                processed_masks, window_size=3
            )

        mask_time = time.time() - start_time - tracking_time

        # ---------------------------------------------------------------------
        # 4. Temporal Flow Video Inpainting
        # ---------------------------------------------------------------------
        _notify("inpainting", 0, num_frames, "Inpainting video frames...")

        def _inpaint_cb(step: int, total: int, msg: str):
            _notify("inpainting", step, total, msg)

        inpainted_frames = self.inpainter.inpaint_video(
            frames=frames,
            masks=processed_masks,
            progress_callback=_inpaint_cb,
        )

        total_time = time.time() - start_time
        _notify("completed", num_frames, num_frames, f"Finished in {total_time:.2f}s")

        return {
            "inpainted_frames": inpainted_frames,
            "masks": processed_masks,
            "raw_masks": raw_masks,
            "stats": {
                "num_frames": num_frames,
                "resolution": (w, h),
                "total_time_seconds": total_time,
                "tracking_time_seconds": tracking_time,
                "mask_time_seconds": mask_time,
                "inpainting_time_seconds": total_time - tracking_time - mask_time,
                "tracker_mode": "SAM 2" if self.tracker.is_sam2_ready else "Optical Flow Fallback",
                "inpainter_mode": "ProPainter Neural" if self.inpainter.is_neural_ready else "Temporal Flow Fallback",
            },
        }

    def process_video_file(
        self,
        input_video_path: str,
        output_video_path: str,
        keyframe_idx: int = 0,
        initial_mask: Optional[Union[np.ndarray, str]] = None,
        points: Optional[List[Tuple[float, float]]] = None,
        labels: Optional[List[int]] = None,
        box: Optional[Tuple[float, float, float, float]] = None,
        dilation_radius: Optional[int] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a video file from disk and save the inpainted video to disk.

        Args:
            input_video_path: Path to source video (.mp4, .mov, .avi, etc.).
            output_video_path: Target path to output video (.mp4).
            keyframe_idx: Frame index for initial interaction prompt.
            initial_mask: Binary mask or base64 string.
            points: Points prompt list.
            labels: Labels list (1 or 0).
            box: Bounding box prompt tuple.
            dilation_radius: Custom dilation radius.
            max_frames: Optional frame limit for testing or short previews.
            progress_callback: Callback `(stage, step, total, message)`.

        Returns:
            Dict containing pipeline result metadata.
        """
        if not os.path.isfile(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        # 1. Ingest frames
        if progress_callback:
            progress_callback("ingest", 0, 1, "Reading input video...")

        frames, fps = self.extract_frames_from_video(input_video_path, max_frames=max_frames)

        # 2. Run object removal
        result = self.remove_object(
            frames=frames,
            keyframe_idx=keyframe_idx,
            initial_mask=initial_mask,
            points=points,
            labels=labels,
            box=box,
            dilation_radius=dilation_radius,
            progress_callback=progress_callback,
        )

        # 3. Save output video
        if progress_callback:
            progress_callback("export", 0, 1, "Encoding output video...")

        self.save_frames_to_video(
            frames=result["inpainted_frames"],
            output_path=output_video_path,
            fps=fps,
        )

        if progress_callback:
            progress_callback("export", 1, 1, f"Video saved to {output_video_path}")

        result["output_video_path"] = output_video_path
        result["fps"] = fps
        return result

    # =========================================================================
    # Video I/O Helpers
    # =========================================================================

    @staticmethod
    def extract_frames_from_video(
        video_path: str, max_frames: Optional[int] = None
    ) -> Tuple[List[np.ndarray], float]:
        """
        Extract all frames from a video file into memory.

        Args:
            video_path: Video file path.
            max_frames: Optional limit on number of frames to load.

        Returns:
            Tuple[List[np.ndarray], float]: List of BGR frames and video FPS.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if max_frames and len(frames) >= max_frames:
                    break
        finally:
            cap.release()

        return frames, fps

    @staticmethod
    def save_frames_to_video(
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 30.0,
        fourcc: str = "mp4v",
    ) -> None:
        """
        Write a sequence of video frames to an MP4 video file.

        Args:
            frames: List of BGR frames (H, W, 3).
            output_path: Destination video file path.
            fps: Video frames per second.
            fourcc: FourCC codec string ('mp4v', 'avc1', etc.).
        """
        if not frames:
            raise ValueError("No frames to write.")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        h, w = frames[0].shape[:2]

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (w, h),
        )

        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
