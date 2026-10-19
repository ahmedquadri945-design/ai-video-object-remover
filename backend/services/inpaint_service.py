import os
import shutil
import base64
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from config import UPLOADS_DIR, PROCESSED_DIR, TEMP_DIR
from services.job_manager import job_manager, JobStatus
from services.video_service import VideoService

try:
    from ai_engine.pipeline import VideoObjectRemovalPipeline
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from ai_engine.pipeline import VideoObjectRemovalPipeline

logger = logging.getLogger(__name__)


class InpaintPipeline:
    @staticmethod
    def decode_mask_from_data(prompt_data: Any, target_width: int, target_height: int) -> np.ndarray:
        """Decode mask from base64 string, list of points, or binary data into a 2D uint8 mask (0 or 255)."""
        mask = np.zeros((target_height, target_width), dtype=np.uint8)

        if not prompt_data:
            return mask

        # Case 1: Base64 encoded image string (data:image/png;base64,...)
        if isinstance(prompt_data, str) and ("base64," in prompt_data or len(prompt_data) > 100):
            try:
                base64_str = prompt_data.split("base64,")[-1]
                img_bytes = base64.b64decode(base64_str)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                decoded = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

                if decoded is not None:
                    # If RGBA, extract alpha channel or positive pixels
                    if len(decoded.shape) == 3 and decoded.shape[2] == 4:
                        alpha = decoded[:, :, 3]
                        raw_mask = (alpha > 10).astype(np.uint8) * 255
                    elif len(decoded.shape) == 3:
                        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
                        raw_mask = (gray > 10).astype(np.uint8) * 255
                    else:
                        raw_mask = (decoded > 10).astype(np.uint8) * 255

                    # Resize to frame dimensions
                    if raw_mask.shape[:2] != (target_height, target_width):
                        raw_mask = cv2.resize(raw_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
                    return raw_mask
            except Exception as e:
                logger.error(f"Failed to decode base64 mask: {e}")

        # Case 2: List of points or coordinates e.g. [{"x": 120, "y": 300}, ...]
        if isinstance(prompt_data, list):
            points = []
            for pt in prompt_data:
                if isinstance(pt, dict) and "x" in pt and "y" in pt:
                    points.append((int(pt["x"]), int(pt["y"])))
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    points.append((int(pt[0]), int(pt[1])))

            if points:
                # If multiple points, draw connected polygon or circular regions around each point
                if len(points) >= 3:
                    pts_np = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [pts_np], 255)
                for pt in points:
                    cv2.circle(mask, pt, radius=25, color=255, thickness=-1)

        # Case 3: Bounding box dict e.g. {"x": 10, "y": 20, "width": 100, "height": 100}
        if isinstance(prompt_data, dict):
            if "x" in prompt_data and "y" in prompt_data and "width" in prompt_data and "height" in prompt_data:
                x = int(prompt_data["x"])
                y = int(prompt_data["y"])
                w = int(prompt_data["width"])
                h = int(prompt_data["height"])
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            elif "points" in prompt_data:
                return InpaintPipeline.decode_mask_from_data(prompt_data["points"], target_width, target_height)
            elif "mask" in prompt_data:
                return InpaintPipeline.decode_mask_from_data(prompt_data["mask"], target_width, target_height)

        return mask

    @classmethod
    async def process_job(cls, job_id: str, video_path: Path):
        """Asynchronous execution pipeline for video object tracking and inpainting."""
        job = job_manager.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found.")
            return

        job_dir = TEMP_DIR / job_id
        extracted_frames_dir = job_dir / "extracted_frames"
        inpainted_frames_dir = job_dir / "inpainted_frames"
        audio_file = job_dir / "audio.aac"
        final_video_name = f"{job_id}_clean.mp4"
        final_video_path = PROCESSED_DIR / final_video_name

        try:
            # 1. Initialization
            await job_manager.update_job(
                job_id,
                status=JobStatus.TRACKING,
                progress=5.0,
                stage="Analyzing video stream and extracting audio track...",
                log_msg="Starting video analysis and preparation.",
            )

            job_dir.mkdir(parents=True, exist_ok=True)
            extracted_frames_dir.mkdir(parents=True, exist_ok=True)
            inpainted_frames_dir.mkdir(parents=True, exist_ok=True)

            # Get video specs
            info = await asyncio.to_thread(VideoService.get_video_info, video_path)
            width, height, fps, total_frames = (
                info["width"],
                info["height"],
                info["fps"],
                info["frame_count"],
            )

            # Extract audio track
            has_audio = await asyncio.to_thread(VideoService.extract_audio, video_path, audio_file)
            if has_audio:
                await job_manager.update_job(
                    job_id, log_msg="Extracted original audio track for synchronization."
                )

            # 2. Extract frames
            await job_manager.update_job(
                job_id,
                progress=12.0,
                stage=f"Extracting {total_frames} frames from video...",
                log_msg=f"Video dimensions: {width}x{height} @ {fps} fps ({total_frames} frames).",
            )

            frame_paths = await asyncio.to_thread(
                VideoService.extract_frames,
                video_path,
                extracted_frames_dir,
            )

            total_frames = len(frame_paths)
            if total_frames == 0:
                raise RuntimeError("No frames could be extracted from video.")

            # 3. Create initial mask from prompt
            await job_manager.update_job(
                job_id,
                progress=20.0,
                stage="Generating initial object mask...",
                log_msg=f"Processing {job.prompt_type} prompt data for object segmentation.",
            )

            initial_mask = cls.decode_mask_from_data(job.prompt_data, width, height)

            # Check keyframes in settings if initial mask is empty
            if cv2.countNonZero(initial_mask) == 0 and "keyframes" in job.settings:
                for kf in job.settings["keyframes"]:
                    if kf.get("mask_data_url"):
                        initial_mask = cls.decode_mask_from_data(kf["mask_data_url"], width, height)
                        if cv2.countNonZero(initial_mask) > 0:
                            break

            # Fallback if mask is somehow blank
            if cv2.countNonZero(initial_mask) == 0:
                logger.warning("Empty mask received. Defaulting to center region for demonstration.")
                cx, cy = width // 2, height // 2
                cv2.circle(initial_mask, (cx, cy), radius=max(35, min(width, height) // 8), color=255, thickness=-1)

            keyframe_idx = int(job.settings.get("keyframe_idx", 0))
            if keyframe_idx < 0 or keyframe_idx >= total_frames:
                keyframe_idx = 0

            # Dilation: default 14px for clean seamless boundary absorption
            dilate_size = int(job.settings.get("dilate_mask", 14))

            # 4. Load all frames for temporal processing
            frames = [cv2.imread(p) for p in frame_paths]

            # 5. Execute Video Object Removal Pipeline
            await job_manager.update_job(
                job_id,
                status=JobStatus.TRACKING,
                progress=25.0,
                stage="Tracking object and propagating mask across frames...",
                log_msg="Propagating spatio-temporal mask across video sequence.",
            )

            loop = asyncio.get_running_loop()

            def pipeline_progress_cb(stage: str, step: int, total: int, msg: str):
                try:
                    if stage == "tracking":
                        track_prog = 25.0 + (step / max(1, total)) * 15.0
                        asyncio.run_coroutine_threadsafe(
                            job_manager.update_job(
                                job_id,
                                progress=track_prog,
                                stage=f"Tracking object: frame {step}/{total}...",
                                log_msg=msg,
                            ),
                            loop,
                        )
                    elif stage == "mask_processing":
                        asyncio.run_coroutine_threadsafe(
                            job_manager.update_job(
                                job_id,
                                progress=40.0,
                                stage="Dilating and smoothing mask boundaries...",
                                log_msg=msg,
                            ),
                            loop,
                        )
                    elif stage == "inpainting":
                        inpaint_prog = 45.0 + (step / max(1, total)) * 42.0
                        asyncio.run_coroutine_threadsafe(
                            job_manager.update_job(
                                job_id,
                                status=JobStatus.INPAINTING,
                                progress=inpaint_prog,
                                stage=f"Temporal Inpainting: frame {step}/{total}...",
                                log_msg=msg,
                            ),
                            loop,
                        )
                except Exception:
                    pass

            # Initialize AI Pipeline
            pipeline = VideoObjectRemovalPipeline(
                dilation_radius=dilate_size,
                temporal_mask_smoothing=True,
            )

            # Run in thread pool to not block asyncio event loop
            pipeline_result = await asyncio.to_thread(
                pipeline.remove_object,
                frames=frames,
                keyframe_idx=keyframe_idx,
                initial_mask=initial_mask,
                dilation_radius=dilate_size,
                progress_callback=pipeline_progress_cb,
            )

            inpainted_frames = pipeline_result["inpainted_frames"]

            # Save inpainted frames
            for idx, cleaned_frame in enumerate(inpainted_frames):
                out_frame_file = inpainted_frames_dir / f"frame_{idx:05d}.png"
                cv2.imwrite(str(out_frame_file), cleaned_frame)


            # 6. Remuxing with original audio
            await job_manager.update_job(
                job_id,
                status=JobStatus.REMUXING,
                progress=88.0,
                stage="Re-encoding and remuxing video with original audio...",
                log_msg="Encoding cleaned frames to MP4 and restoring synchronized audio.",
            )

            remux_success = await asyncio.to_thread(
                VideoService.remux_video,
                inpainted_frames_dir,
                final_video_path,
                fps,
                audio_file if has_audio else None,
            )

            if not remux_success or not final_video_path.exists():
                raise RuntimeError("Failed to remux clean video frames.")

            # 7. Completed successfully
            result_url = f"/api/video/{final_video_name}"
            await job_manager.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress=100.0,
                stage="Object removal completed successfully!",
                result_video_url=result_url,
                result_video_path=str(final_video_path),
                log_msg=f"Video successfully generated: {final_video_name}",
            )

        except Exception as e:
            logger.exception(f"Error during video processing for job {job_id}: {e}")
            await job_manager.update_job(
                job_id,
                status=JobStatus.FAILED,
                error=str(e),
                stage=f"Failed: {str(e)}",
                log_msg=f"Job failed with error: {str(e)}",
            )
        finally:
            # Clean up temp working files
            try:
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {job_dir}: {e}")
