"""
SAM 2 Video Object Tracking and Temporal Propagation Module.

Provides:
- SAM2Tracker: Integration wrapper for Meta's Segment Anything Model 2 (SAM 2) video predictor.
- OpticalFlowTrackerFallback: Robust CPU/GPU fallback tracker using dense Farneback optical flow
  and GrabCut/morphological propagation when SAM 2 weights or dependencies are not available.
"""

from typing import List, Optional, Tuple, Union, Dict, Any, Callable
import os
import glob
import logging
import numpy as np
import cv2

# Configure module logger
logger = logging.getLogger(__name__)


class SAM2Tracker:
    """
    Video object segmentation and propagation wrapper using SAM 2 (Segment Anything Model 2).
    Automatically falls back to OpticalFlowTrackerFallback if SAM 2 model weights or the
    `sam2` package are missing.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_cfg: str = "sam2_hiera_l.yaml",
        device: Optional[str] = None,
        force_fallback: bool = False,
    ) -> None:
        """
        Initialize the SAM 2 tracker.

        Args:
            checkpoint_path: Path to the SAM 2 checkpoint file (e.g. sam2_hiera_large.pt).
            model_cfg: SAM 2 configuration file name or path.
            device: 'cuda', 'cpu', or None (auto-detects CUDA).
            force_fallback: If True, bypasses SAM 2 and uses OpticalFlowTrackerFallback.
        """
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.force_fallback = force_fallback

        # Setup device
        try:
            import torch
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
        except ImportError:
            self.device = "cpu"

        self.predictor = None
        self.is_sam2_ready = False
        self.fallback_tracker = OpticalFlowTrackerFallback()

        if not self.force_fallback:
            self._init_sam2()

    def _init_sam2(self) -> None:
        """Attempt to import and initialize SAM 2 Video Predictor."""
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            logger.info(
                f"SAM 2 checkpoint not found at '{self.checkpoint_path}'. "
                "Pipeline will use OpticalFlowTrackerFallback."
            )
            return

        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor

            logger.info(f"Loading SAM 2 predictor ({self.model_cfg}) on {self.device}...")
            self.predictor = build_sam2_video_predictor(
                self.model_cfg, self.checkpoint_path, device=self.device
            )
            self.is_sam2_ready = True
            logger.info("SAM 2 Video Predictor loaded successfully.")
        except ImportError as err:
            logger.warning(
                f"`sam2` package not installed or import error: {err}. "
                "Using OpticalFlowTrackerFallback."
            )
        except Exception as ex:
            logger.warning(
                f"Failed to load SAM 2 model: {ex}. Using OpticalFlowTrackerFallback."
            )

    # =========================================================================
    # Tracking / Propagation API
    # =========================================================================

    def track_video(
        self,
        frames: List[np.ndarray],
        keyframe_idx: int = 0,
        initial_mask: Optional[np.ndarray] = None,
        points: Optional[List[Tuple[float, float]]] = None,
        labels: Optional[List[int]] = None,
        box: Optional[Tuple[float, float, float, float]] = None,
        obj_id: int = 1,
        temp_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[np.ndarray]:
        """
        Track object across all video frames starting from a keyframe prompt.

        Args:
            frames: List of BGR/RGB video frames as numpy uint8 arrays (H, W, 3).
            keyframe_idx: Index of the frame where user provided interaction prompt (0-indexed).
            initial_mask: Optional binary mask (H, W) on keyframe (0 or 255).
            points: Optional list of (x, y) coordinates on keyframe.
            labels: Optional list of point labels (1=positive foreground, 0=negative background).
            box: Optional bounding box (x1, y1, x2, y2) on keyframe.
            obj_id: Integer identifier for the object being tracked.
            temp_dir: Temporary directory used to write JPEG frames for SAM 2 video predictor.
            progress_callback: Optional callback `(current_step, total_steps, message)`.

        Returns:
            List[np.ndarray]: List of binary masks (H, W) uint8 (values 0 or 255) for all frames.
        """
        num_frames = len(frames)
        if num_frames == 0:
            return []

        # If SAM 2 is ready and weights are present, execute SAM 2 video propagation
        if self.is_sam2_ready and self.predictor is not None:
            try:
                return self._propagate_sam2(
                    frames=frames,
                    keyframe_idx=keyframe_idx,
                    initial_mask=initial_mask,
                    points=points,
                    labels=labels,
                    box=box,
                    obj_id=obj_id,
                    temp_dir=temp_dir,
                    progress_callback=progress_callback,
                )
            except Exception as ex:
                logger.error(
                    f"SAM 2 propagation failed ({ex}). Falling back to OpticalFlowTrackerFallback."
                )

        # Fallback tracking
        return self.fallback_tracker.propagate(
            frames=frames,
            keyframe_idx=keyframe_idx,
            initial_mask=initial_mask,
            points=points,
            labels=labels,
            box=box,
            progress_callback=progress_callback,
        )

    def _propagate_sam2(
        self,
        frames: List[np.ndarray],
        keyframe_idx: int,
        initial_mask: Optional[np.ndarray],
        points: Optional[List[Tuple[float, float]]],
        labels: Optional[List[int]],
        box: Optional[Tuple[float, float, float, float]],
        obj_id: int,
        temp_dir: Optional[str],
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> List[np.ndarray]:
        """Execute propagation using SAM 2 Video Predictor."""
        import torch
        import tempfile

        num_frames = len(frames)
        h, w = frames[0].shape[:2]

        # SAM 2 requires video frames saved in a folder as JPEGs or mp4
        managed_temp_dir = False
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
            managed_temp_dir = True

        try:
            if progress_callback:
                progress_callback(0, num_frames, "Exporting frames for SAM 2...")

            for i, frame in enumerate(frames):
                frame_path = os.path.join(temp_dir, f"{i:05d}.jpg")
                if not os.path.exists(frame_path):
                    cv2.imwrite(frame_path, frame)

            if progress_callback:
                progress_callback(0, num_frames, "Initializing SAM 2 state...")

            inference_state = self.predictor.init_state(video_path=temp_dir)

            # Apply initial prompts
            if initial_mask is not None:
                mask_bool = (initial_mask > 127).astype(np.float32)
                mask_tensor = torch.from_numpy(mask_bool).to(self.device)
                self.predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=keyframe_idx,
                    obj_id=obj_id,
                    mask=mask_tensor,
                )
            elif points is not None and len(points) > 0:
                pts_np = np.array(points, dtype=np.float32)
                lbl_np = (
                    np.array(labels, dtype=np.int32)
                    if labels is not None
                    else np.ones(len(points), dtype=np.int32)
                )
                box_np = np.array(box, dtype=np.float32) if box is not None else None
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=keyframe_idx,
                    obj_id=obj_id,
                    points=pts_np,
                    labels=lbl_np,
                    box=box_np,
                )
            elif box is not None:
                box_np = np.array(box, dtype=np.float32)
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=keyframe_idx,
                    obj_id=obj_id,
                    box=box_np,
                )
            else:
                raise ValueError("No prompt (mask, points, or box) provided for keyframe.")

            # Propagate forward and backward
            masks_dict: Dict[int, np.ndarray] = {}

            # Propagate throughout video
            step_count = 0
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                inference_state
            ):
                step_count += 1
                if progress_callback and step_count % 5 == 0:
                    progress_callback(
                        step_count, num_frames, f"SAM 2 propagating: frame {out_frame_idx}/{num_frames}"
                    )

                # Find object mask
                for idx, oid in enumerate(out_obj_ids):
                    if oid == obj_id:
                        mask_logit = out_mask_logits[idx]
                        binary_mask = (mask_logit > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
                        masks_dict[out_frame_idx] = binary_mask

            # Compile into ordered list
            result_masks = []
            for i in range(num_frames):
                if i in masks_dict:
                    result_masks.append(masks_dict[i])
                else:
                    result_masks.append(np.zeros((h, w), dtype=np.uint8))

            return result_masks

        finally:
            if managed_temp_dir and os.path.exists(temp_dir):
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)


class OpticalFlowTrackerFallback:
    """
    Robust optical flow based mask tracker.
    Used when SAM 2 weights or dependencies are unavailable.
    Propagates binary masks bidirectional (forward and backward) from keyframe
    using Farneback dense optical flow and morphology.
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ) -> None:
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma

    def propagate(
        self,
        frames: List[np.ndarray],
        keyframe_idx: int = 0,
        initial_mask: Optional[np.ndarray] = None,
        points: Optional[List[Tuple[float, float]]] = None,
        labels: Optional[List[int]] = None,
        box: Optional[Tuple[float, float, float, float]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[np.ndarray]:
        """
        Propagate mask across all frames using optical flow.
        """
        num_frames = len(frames)
        if num_frames == 0:
            return []

        h, w = frames[0].shape[:2]

        # 1. Establish the keyframe mask
        if initial_mask is not None:
            base_mask = (initial_mask > 127).astype(np.uint8) * 255
        else:
            base_mask = self._generate_seed_mask(
                frame=frames[keyframe_idx], points=points, labels=labels, box=box
            )

        masks: List[Optional[np.ndarray]] = [None] * num_frames
        masks[keyframe_idx] = base_mask

        # Precompute grayscale frames for optical flow
        gray_frames = [
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            for f in frames
        ]

        total_steps = num_frames - 1
        current_step = 0

        # 2. Propagate forward: keyframe_idx -> num_frames - 1
        for t in range(keyframe_idx, num_frames - 1):
            curr_mask = masks[t]
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[t],
                gray_frames[t + 1],
                None,
                pyr_scale=self.pyr_scale,
                levels=self.levels,
                winsize=self.winsize,
                iterations=self.iterations,
                poly_n=self.poly_n,
                poly_sigma=self.poly_sigma,
                flags=0,
            )
            next_mask = self._warp_mask_forward(curr_mask, flow)
            masks[t + 1] = next_mask
            current_step += 1
            if progress_callback:
                progress_callback(
                    current_step, total_steps, f"Flow tracking (forward): frame {t + 1}/{num_frames}"
                )

        # 3. Propagate backward: keyframe_idx -> 0
        for t in range(keyframe_idx, 0, -1):
            curr_mask = masks[t]
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[t],
                gray_frames[t - 1],
                None,
                pyr_scale=self.pyr_scale,
                levels=self.levels,
                winsize=self.winsize,
                iterations=self.iterations,
                poly_n=self.poly_n,
                poly_sigma=self.poly_sigma,
                flags=0,
            )
            prev_mask = self._warp_mask_forward(curr_mask, flow)
            masks[t - 1] = prev_mask
            current_step += 1
            if progress_callback:
                progress_callback(
                    current_step, total_steps, f"Flow tracking (backward): frame {t - 1}/{num_frames}"
                )

        return [m if m is not None else np.zeros((h, w), dtype=np.uint8) for m in masks]

    def _warp_mask_forward(self, mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warp mask from frame t to t+1 using optical flow.
        """
        h, w = mask.shape[:2]
        # Generate inverse mapping coordinate grid:
        # remap requires coordinates in current image mapping to source image
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x - flow_x).astype(np.float32)
        map_y = (grid_y - flow_y).astype(np.float32)

        warped = cv2.remap(
            mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Low threshold to preserve object mask coverage across frames
        _, warped_binary = cv2.threshold(warped, 30, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        warped_binary = cv2.morphologyEx(warped_binary, cv2.MORPH_CLOSE, kernel)

        # Safety: if flow tracking shrank or lost the mask, preserve the original mask shape
        if np.count_nonzero(warped_binary) < (0.2 * np.count_nonzero(mask)):
            warped_binary = cv2.bitwise_or(warped_binary, mask)

        return warped_binary

    def _generate_seed_mask(
        self,
        frame: np.ndarray,
        points: Optional[List[Tuple[float, float]]],
        labels: Optional[List[int]],
        box: Optional[Tuple[float, float, float, float]],
    ) -> np.ndarray:
        """
        Generate initial seed mask when only points or bounding boxes are provided.
        Uses GrabCut algorithm seeded with user points / box.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            grab_mask = np.zeros((h, w), np.uint8)

            try:
                cv2.grabCut(
                    frame,
                    grab_mask,
                    rect,
                    bgd_model,
                    fgd_model,
                    iterCount=3,
                    mode=cv2.GC_INIT_WITH_RECT,
                )
                mask = np.where(
                    (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 255, 0
                ).astype(np.uint8)
                return mask
            except Exception:
                # If GrabCut fails, return bounding box mask
                mask[y1 : y2 + 1, x1 : x2 + 1] = 255
                return mask

        if points is not None and len(points) > 0:
            # Positive points flood/circle fill
            for idx, pt in enumerate(points):
                lbl = labels[idx] if labels is not None and idx < len(labels) else 1
                px, py = int(pt[0]), int(pt[1])
                if 0 <= px < w and 0 <= py < h:
                    if lbl == 1:
                        cv2.circle(mask, (px, py), radius=20, color=255, thickness=-1)
                    else:
                        cv2.circle(mask, (px, py), radius=20, color=0, thickness=-1)
            return mask

        return mask
