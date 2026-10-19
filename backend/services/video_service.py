import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import cv2

from config import FFMPEG_PATH, FFPROBE_PATH

logger = logging.getLogger(__name__)


class VideoService:
    @staticmethod
    def get_video_info(video_path: str | Path) -> Dict[str, Any]:
        """Extract metadata: duration, fps, frame count, dimensions, aspect ratio, file size."""
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Handle edge cases for fps or frame count
        if fps <= 0:
            fps = 30.0
        if frame_count <= 0:
            frame_count = 1

        duration = round(frame_count / fps, 2)
        file_size = os.path.getsize(video_path)
        aspect_ratio = f"{width}:{height}" if height > 0 else "16:9"

        return {
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "file_size": file_size,
        }

    @staticmethod
    def extract_preview_frame(video_path: str | Path, output_image_path: str | Path, frame_index: int = 0) -> str:
        """Extract a single keyframe (default first frame) for canvas annotation and preview."""
        video_path = str(video_path)
        output_image_path = str(output_image_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")

        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = cap.read()
        cap.release()

        if not success or frame is None:
            raise RuntimeError(f"Failed to read frame at index {frame_index} from {video_path}")

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, frame)
        return output_image_path

    @staticmethod
    def extract_frames(
        video_path: str | Path,
        output_dir: str | Path,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """Extract video frames into numbered images (e.g. frame_00000.png)."""
        video_path = str(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1
        if max_frames:
            total_frames = min(total_frames, max_frames)

        frame_paths = []
        frame_idx = 0

        while cap.isOpened():
            if max_frames and frame_idx >= max_frames:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_filename = output_dir / f"frame_{frame_idx:05d}.png"
            cv2.imwrite(str(frame_filename), frame)
            frame_paths.append(str(frame_filename))

            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames)

        cap.release()
        if progress_callback:
            progress_callback(frame_idx, frame_idx)

        return frame_paths

    @staticmethod
    def extract_audio(video_path: str | Path, output_audio_path: str | Path) -> bool:
        """Extract the audio track from the original video using FFmpeg if available.
        Returns True if audio was successfully extracted, False otherwise.
        """
        video_path = str(video_path)
        output_audio_path = str(output_audio_path)

        if not FFMPEG_PATH:
            logger.warning("FFmpeg executable not found. Audio extraction skipped.")
            return False

        try:
            os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
            # Try copying original audio codec or fallback to aac
            cmd = [
                FFMPEG_PATH,
                "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "aac",
                "-b:a", "192k",
                output_audio_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            if result.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                logger.info(f"Audio extracted successfully to {output_audio_path}")
                return True
            else:
                logger.warning(f"No audio extracted (video might have no audio track or extraction failed): {result.stderr[:200]}")
                return False
        except Exception as e:
            logger.warning(f"Error during audio extraction: {e}")
            return False

    @staticmethod
    def remux_video(
        frames_dir: str | Path,
        output_video_path: str | Path,
        fps: float,
        audio_path: Optional[str | Path] = None,
        frame_pattern: str = "frame_%05d.png",
    ) -> bool:
        """Remux processed image frames with the original audio track back into a high-compatibility MP4 video.
        Uses FFmpeg with libx264 and yuv420p for universal browser playback, falling back to cv2.VideoWriter if FFmpeg is unavailable.
        """
        frames_dir = Path(frames_dir)
        output_video_path = str(output_video_path)
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        input_frame_pattern = str(frames_dir / frame_pattern)

        # 1. Attempt FFmpeg remuxing first for best quality, standard H.264 profile, and audio muxing
        if FFMPEG_PATH:
            try:
                cmd = [
                    FFMPEG_PATH,
                    "-y",
                    "-framerate", str(fps),
                    "-i", input_frame_pattern,
                ]

                has_audio = False
                if audio_path and os.path.exists(str(audio_path)) and os.path.getsize(str(audio_path)) > 0:
                    cmd.extend(["-i", str(audio_path)])
                    has_audio = True

                cmd.extend([
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "18",
                    "-preset", "medium",
                ])

                if has_audio:
                    cmd.extend(["-c:a", "aac", "-shortest"])

                cmd.append(output_video_path)

                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
                if result.returncode == 0 and os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                    logger.info(f"Remuxing completed successfully with FFmpeg: {output_video_path}")
                    return True
                else:
                    logger.error(f"FFmpeg remuxing failed: {result.stderr}")
            except Exception as e:
                logger.error(f"Exception during FFmpeg remuxing: {e}")

        # 2. Fallback to OpenCV VideoWriter if FFmpeg is not available or failed
        logger.info("Using OpenCV fallback for video assembly (without audio).")
        try:
            frame_files = sorted(list(frames_dir.glob("frame_*.png")))
            if not frame_files:
                raise ValueError(f"No frames found in {frames_dir}")

            first_frame = cv2.imread(str(frame_files[0]))
            if first_frame is None:
                raise ValueError("Could not read first frame for dimensions.")
            h, w, _ = first_frame.shape

            writer = None
            for codec in ["avc1", "H264", "mp4v"]:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
                    if test_writer.isOpened():
                        writer = test_writer
                        logger.info(f"OpenCV VideoWriter initialized with codec: {codec}")
                        break
                except Exception:
                    continue

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

            for fpath in frame_files:
                frame = cv2.imread(str(fpath))
                if frame is not None:
                    writer.write(frame)

            writer.release()
            return os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0
        except Exception as e:
            logger.error(f"OpenCV fallback remuxing failed: {e}")
            return False
