"""
ProPainter Video Inpainting and Temporal Flow Completion Module.

Provides:
- ProPainterInpainter: Video inpainting engine wrapper supporting ProPainter with chunked sub-video
  inference (to prevent GPU VRAM out-of-memory errors), FP16 mixed precision, and boundary overlap blending.
- TemporalFlowInpainterFallback: High-performance CPU/GPU fallback using bidirectional optical flow
  temporal pixel warping and Navier-Stokes/Telea inpainting for environments without full ProPainter weights.
"""

from typing import List, Optional, Tuple, Union, Dict, Any, Callable
import os
import sys
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ProPainterInpainter:
    """
    Video inpainting wrapper for ProPainter (Flow-guided Video Inpainting with Recurrent
    Flow Completion and Dual-Domain Attention).

    Supports:
    - Chunked sub-video processing (e.g. 50-80 frames per chunk) with overlap blending.
    - FP16 half precision for reduced VRAM footprint.
    - Automatic fallback to TemporalFlowInpainterFallback when neural weights are absent.
    """

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        chunk_size: int = 60,
        chunk_overlap: int = 5,
        use_fp16: bool = True,
        device: Optional[str] = None,
        force_fallback: bool = False,
    ) -> None:
        """
        Initialize ProPainterInpainter.

        Args:
            weights_dir: Directory containing ProPainter.pth, recurrent_flow_completion.pth, raft-things.pth.
            chunk_size: Number of frames per chunk during inference to cap VRAM consumption.
            chunk_overlap: Overlap frames between adjacent chunks for smooth linear temporal cross-fading.
            use_fp16: Whether to run inference in FP16 mixed precision on GPU.
            device: 'cuda', 'cpu', or None (auto-detect).
            force_fallback: Force local fallback inpainting without loading neural models.
        """
        self.weights_dir = weights_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_fp16 = use_fp16
        self.force_fallback = force_fallback

        try:
            import torch
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
        except ImportError:
            self.device = "cpu"

        self.model = None
        self.flow_model = None
        self.is_neural_ready = False
        self.fallback_inpainter = TemporalFlowInpainterFallback(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        if not self.force_fallback:
            self._init_propainter()

    def _init_propainter(self) -> None:
        """Check checkpoint existence and attempt initializing ProPainter neural pipeline."""
        if not self.weights_dir or not os.path.isdir(self.weights_dir):
            logger.info(
                f"ProPainter weights directory not found at '{self.weights_dir}'. "
                "Pipeline will use high-quality TemporalFlowInpainterFallback."
            )
            return

        model_path = os.path.join(self.weights_dir, "ProPainter.pth")
        flow_path = os.path.join(self.weights_dir, "recurrent_flow_completion.pth")
        raft_path = os.path.join(self.weights_dir, "raft-things.pth")

        if not (os.path.exists(model_path) and os.path.exists(flow_path) and os.path.exists(raft_path)):
            logger.info(
                "One or more ProPainter checkpoints are missing in weights directory. "
                "Pipeline will use TemporalFlowInpainterFallback."
            )
            return

        try:
            import torch
            # Check for ProPainter package / model definitions
            # If repo is added to sys.path or installed as package
            try:
                from model.modules.flow_comp_raft import RAFT_Bi
                from model.recurrent_flow_completion import RecurrentFlowCompleteNet
                from model.propainter import Inpainter
            except ImportError:
                # Attempt to find ProPainter repository in parent paths or sys.path
                logger.info("ProPainter modules not in Python path. Using fallback inpainter.")
                return

            logger.info(f"Loading ProPainter models on {self.device} (FP16={self.use_fp16})...")

            # Load Flow completion model
            self.flow_model = RecurrentFlowCompleteNet(flow_path)
            self.flow_model.to(self.device)
            self.flow_model.eval()

            # Load Main ProPainter inpainting model
            self.model = Inpainter(model_path)
            self.model.to(self.device)
            self.model.eval()

            if self.use_fp16 and self.device == "cuda":
                self.flow_model = self.flow_model.half()
                self.model = self.model.half()

            self.is_neural_ready = True
            logger.info("ProPainter loaded successfully.")
        except Exception as ex:
            logger.warning(
                f"Failed to initialize ProPainter models ({ex}). Using TemporalFlowInpainterFallback."
            )

    # =========================================================================
    # Inpainting API
    # =========================================================================

    def inpaint_video(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[np.ndarray]:
        """
        Inpaint video sequence with masks.

        Args:
            frames: List of BGR/RGB video frames as numpy uint8 arrays (H, W, 3).
            masks: List of binary masks (H, W) uint8 (255=masked area to remove, 0=keep).
            progress_callback: Optional callback `(current_step, total_steps, message)`.

        Returns:
            List[np.ndarray]: Inpainted video frames.
        """
        assert len(frames) == len(masks), "Frames and masks count must match."
        num_frames = len(frames)
        if num_frames == 0:
            return []

        # If neural model is ready, execute chunked ProPainter neural inpainting
        if self.is_neural_ready and self.model is not None:
            try:
                return self._inpaint_neural_chunked(frames, masks, progress_callback)
            except Exception as ex:
                logger.error(
                    f"ProPainter neural inpainting failed ({ex}). "
                    "Switching to TemporalFlowInpainterFallback."
                )

        # Execute temporal optical flow fallback inpainting
        return self.fallback_inpainter.inpaint_video(frames, masks, progress_callback)

    def _inpaint_neural_chunked(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> List[np.ndarray]:
        """
        Execute ProPainter neural inference in overlapping chunks with temporal linear blending.
        """
        import torch

        num_frames = len(frames)
        chunk_size = self.chunk_size
        overlap = self.chunk_overlap
        stride = chunk_size - overlap

        inpainted_frames: List[np.ndarray] = [f.copy() for f in frames]
        blend_weights = np.zeros(num_frames, dtype=np.float32)

        chunks_indices = []
        start = 0
        while start < num_frames:
            end = min(start + chunk_size, num_frames)
            chunks_indices.append((start, end))
            if end == num_frames:
                break
            start += stride

        total_chunks = len(chunks_indices)

        for chunk_idx, (c_start, c_end) in enumerate(chunks_indices):
            chunk_len = c_end - c_start
            if progress_callback:
                progress_callback(
                    chunk_idx,
                    total_chunks,
                    f"Neural Inpainting chunk {chunk_idx + 1}/{total_chunks} (frames {c_start}-{c_end})",
                )

            sub_frames = frames[c_start:c_end]
            sub_masks = masks[c_start:c_end]

            # Prepare tensors: frames normalized to [-1, 1], masks to [0, 1]
            # Convert list of BGR to RGB tensor (T, C, H, W)
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sub_frames]
            t_frames = torch.from_numpy(np.stack(rgb_frames)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            t_masks = torch.from_numpy(np.stack(sub_masks)).unsqueeze(1).float() / 255.0

            t_frames = t_frames.to(self.device)
            t_masks = t_masks.to(self.device)

            if self.use_fp16 and self.device == "cuda":
                t_frames = t_frames.half()
                t_masks = t_masks.half()

            with torch.no_grad():
                # Neural inference
                with torch.cuda.amp.autocast(enabled=(self.use_fp16 and self.device == "cuda")):
                    # Completed flows
                    flows = self.flow_model(t_frames, t_masks)
                    # Completed frames
                    out_frames_tensor = self.model(t_frames, flows, t_masks)

            out_np = (
                ((out_frames_tensor.permute(0, 2, 3, 1).cpu().float().numpy() + 1.0) * 127.5)
                .clip(0, 255)
                .astype(np.uint8)
            )

            # Convert back to BGR and blend into result sequence
            for idx in range(chunk_len):
                frame_global_idx = c_start + idx
                inpainted_bgr = cv2.cvtColor(out_np[idx], cv2.COLOR_RGB2BGR)

                # Linear overlap weight
                w = 1.0
                if idx < overlap and c_start > 0:
                    w = float(idx + 1) / float(overlap + 1)
                elif idx >= chunk_len - overlap and c_end < num_frames:
                    w = float(chunk_len - idx) / float(overlap + 1)

                if blend_weights[frame_global_idx] == 0:
                    inpainted_frames[frame_global_idx] = inpainted_bgr
                    blend_weights[frame_global_idx] = w
                else:
                    curr = inpainted_frames[frame_global_idx].astype(np.float32)
                    new_val = inpainted_bgr.astype(np.float32)
                    total_w = blend_weights[frame_global_idx] + w
                    blended = (curr * blend_weights[frame_global_idx] + new_val * w) / total_w
                    inpainted_frames[frame_global_idx] = blended.astype(np.uint8)
                    blend_weights[frame_global_idx] = total_w

            # Clear CUDA cache between chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if progress_callback:
            progress_callback(total_chunks, total_chunks, "Neural Inpainting completed.")

        return inpainted_frames


class TemporalFlowInpainterFallback:
    """
    High-quality temporal video inpainting fallback.
    Utilizes bidirectional optical flow frame-to-frame pixel borrowing across a temporal window
    combined with Navier-Stokes / Fast Marching edge inpainting and temporal cross-blending.
    Runs reliably on any CPU or GPU without requiring multi-gigabyte neural weights.
    """

    def __init__(
        self,
        temporal_radius: int = 4,
        inpaint_method: int = cv2.INPAINT_TELEA,
        inpaint_radius: int = 5,
        chunk_size: int = 60,
        chunk_overlap: int = 5,
    ) -> None:
        """
        Initialize temporal flow fallback inpainter.

        Args:
            temporal_radius: Number of neighbor frames before and after to source background pixels from.
            inpaint_method: cv2.INPAINT_TELEA or cv2.INPAINT_NS (Navier-Stokes).
            inpaint_radius: Pixel radius for spatial inpainting.
            chunk_size: Processing batch size.
            chunk_overlap: Overlap size between chunks.
        """
        self.temporal_radius = temporal_radius
        self.inpaint_method = inpaint_method
        self.inpaint_radius = inpaint_radius
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def inpaint_video(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[np.ndarray]:
        """
        Inpaint video sequence with temporal optical flow warping and boundary blending.
        """
        num_frames = len(frames)
        if num_frames == 0:
            return []

        h, w = frames[0].shape[:2]
        result_frames: List[np.ndarray] = []

        # Precompute grayscale representations for optical flow calculation
        gray_frames = [
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            for f in frames
        ]

        # Process each frame
        for t in range(num_frames):
            if progress_callback and (t % 3 == 0 or t == num_frames - 1):
                progress_callback(
                    t + 1,
                    num_frames,
                    f"Temporal Flow Inpainting: frame {t + 1}/{num_frames}",
                )

            curr_frame = frames[t].copy()
            curr_mask = masks[t]

            # If no mask on this frame, keep original
            if np.count_nonzero(curr_mask) == 0:
                result_frames.append(curr_frame)
                continue

            # Temporal pixel accumulation buffer
            accum_color = np.zeros((h, w, 3), dtype=np.float32)
            accum_weight = np.zeros((h, w), dtype=np.float32)
            mask_bool = curr_mask > 0

            # Scan temporal neighborhood
            neighbor_indices = []
            for offset in range(1, self.temporal_radius + 1):
                if t - offset >= 0:
                    neighbor_indices.append((t - offset, offset))
                if t + offset < num_frames:
                    neighbor_indices.append((t + offset, offset))

            # Sort by temporal distance
            neighbor_indices.sort(key=lambda x: x[1])

            # Grid for remap
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

            for n_idx, dist in neighbor_indices:
                # Compute optical flow from current frame t to neighbor n_idx
                flow = cv2.calcOpticalFlowFarneback(
                    gray_frames[t],
                    gray_frames[n_idx],
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )

                map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
                map_y = (grid_y + flow[:, :, 1]).astype(np.float32)

                # Warp neighbor frame and neighbor mask to current perspective
                warped_neighbor = cv2.remap(
                    frames[n_idx],
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                warped_neighbor_mask = cv2.remap(
                    masks[n_idx],
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255,
                )

                # Valid pixels are those inside current mask WHERE neighbor pixel is NOT masked
                valid_sample = mask_bool & (warped_neighbor_mask == 0)

                if np.any(valid_sample):
                    # Weight decays with temporal distance
                    weight = 1.0 / (dist + 0.5)
                    accum_color[valid_sample] += (
                        warped_neighbor[valid_sample].astype(np.float32) * weight
                    )
                    accum_weight[valid_sample] += weight

            # Blend accumulated temporal pixels into current frame
            filled_by_flow = accum_weight > 0
            if np.any(filled_by_flow):
                curr_frame[filled_by_flow] = (
                    accum_color[filled_by_flow] / accum_weight[filled_by_flow, None]
                ).astype(np.uint8)

            # Unfilled remaining mask pixels (not visible in any neighbor frame)
            remaining_mask = np.zeros((h, w), dtype=np.uint8)
            remaining_mask[mask_bool & (~filled_by_flow)] = 255

            if np.count_nonzero(remaining_mask) > 0:
                # Fill remaining occluded holes using Fast Marching / Navier-Stokes
                curr_frame = cv2.inpaint(
                    curr_frame,
                    remaining_mask,
                    inpaintRadius=self.inpaint_radius,
                    flags=self.inpaint_method,
                )

            # Smooth composite boundaries to eliminate any seam edges
            seam_boundary = cv2.Canny(curr_mask, 100, 200)
            if np.any(seam_boundary > 0):
                dilated_seam = cv2.dilate(
                    seam_boundary,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                )
                curr_frame = cv2.inpaint(
                    curr_frame, dilated_seam, inpaintRadius=3, flags=cv2.INPAINT_TELEA
                )

            result_frames.append(curr_frame)

        return result_frames
