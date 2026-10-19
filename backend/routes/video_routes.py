import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Any, Dict, Literal
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import UPLOADS_DIR, PROCESSED_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE
from services.video_service import VideoService
from services.job_manager import job_manager, JobStatus
from services.inpaint_service import InpaintPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["video"])


class ProcessRequest(BaseModel):
    video_id: str = Field(..., description="The ID of the uploaded video")
    prompt_type: Optional[str] = Field(
        default="brush", description="Type of prompt: 'brush' or 'points'"
    )
    prompt_data: Optional[Any] = Field(
        default=None, description="Base64 mask image or list of point coordinates"
    )
    keyframes: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="List of frame-specific annotations from UI"
    )
    dilation: Optional[int] = Field(default=8)
    feather: Optional[int] = Field(default=4)
    high_quality: Optional[bool] = Field(default=True)
    subpixel: Optional[bool] = Field(default=True)
    horizon: Optional[str] = Field(default="bidirectional")
    settings: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"dilate_mask": 8, "algorithm": "telea", "inpaint_radius": 5},
        description="Optional processing parameters",
    )


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file, extract metadata and keyframe preview for the annotation canvas."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename provided.")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file_ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    video_id = str(uuid.uuid4())
    safe_filename = f"{video_id}{file_ext}"
    saved_video_path = UPLOADS_DIR / safe_filename
    preview_filename = f"{video_id}_preview.jpg"
    preview_path = UPLOADS_DIR / preview_filename

    # Save uploaded file asynchronously with chunking
    total_size = 0
    try:
        async with aiofiles.open(saved_video_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)}MB",
                    )
                await out_file.write(chunk)
    except HTTPException:
        if saved_video_path.exists():
            saved_video_path.unlink()
        raise
    except Exception as e:
        logger.error(f"Error saving upload: {e}")
        if saved_video_path.exists():
            saved_video_path.unlink()
        raise HTTPException(status_code=500, detail="Failed to save uploaded video file.")

    # Extract metadata and keyframe preview
    try:
        video_info = VideoService.get_video_info(saved_video_path)
        VideoService.extract_preview_frame(saved_video_path, preview_path, frame_index=0)
    except Exception as e:
        logger.error(f"Failed to inspect video: {e}")
        if saved_video_path.exists():
            saved_video_path.unlink()
        if preview_path.exists():
            preview_path.unlink()
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted video file: {str(e)}")

    return {
        "video_id": video_id,
        "filename": file.filename,
        "width": video_info["width"],
        "height": video_info["height"],
        "fps": video_info["fps"],
        "frame_count": video_info["frame_count"],
        "duration": video_info["duration"],
        "aspect_ratio": video_info["aspect_ratio"],
        "file_size": total_size,
        "preview_url": f"/api/preview/{video_id}",
        "video_url": f"/api/video/{safe_filename}",
    }


@router.get("/preview/{video_id}")
async def get_preview(video_id: str):
    """Serve the extracted keyframe preview for canvas interaction."""
    preview_path = UPLOADS_DIR / f"{video_id}_preview.jpg"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview frame not found.")
    return FileResponse(preview_path, media_type="image/jpeg")


@router.post("/process")
async def start_process(request: Request, background_tasks: BackgroundTasks):
    """Start an asynchronous video object removal job."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    video_id = data.get("video_id")
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required.")

    # Find matching uploaded video
    matching_files = list(UPLOADS_DIR.glob(f"{video_id}.*"))
    video_files = [f for f in matching_files if not f.name.endswith("_preview.jpg")]

    if not video_files:
        raise HTTPException(status_code=404, detail=f"No video found for video_id {video_id}")

    video_path = video_files[0]
    job_id = str(uuid.uuid4())

    prompt_data = data.get("prompt_data")
    prompt_type = data.get("prompt_type", "brush")
    keyframes = data.get("keyframes") or []

    if prompt_data is None and keyframes:
        for kf in keyframes:
            if kf.get("mask_data_url"):
                prompt_data = kf["mask_data_url"]
                prompt_type = "brush"
                break
            elif kf.get("points") and len(kf["points"]) > 0:
                prompt_data = kf["points"]
                prompt_type = "points"
                break

    # Build merged settings
    settings = dict(data.get("settings") or {})
    if "dilation" in data:
        settings["dilate_mask"] = data["dilation"]
    if "feather" in data:
        settings["feather"] = data["feather"]
    if "high_quality" in data:
        settings["high_quality"] = data["high_quality"]
    if "horizon" in data:
        settings["horizon"] = data["horizon"]
    if "keyframe_idx" in data:
        settings["keyframe_idx"] = data["keyframe_idx"]
    if keyframes:
        settings["keyframes"] = keyframes

    job = job_manager.create_job(
        job_id=job_id,
        video_id=video_id,
        prompt_type=prompt_type,
        prompt_data=prompt_data,
        settings=settings,
    )

    # Launch background processing task
    background_tasks.add_task(InpaintPipeline.process_job, job_id, video_path)

    return {
        "job_id": job.job_id,
        "video_id": job.video_id,
        "status": job.status,
        "stage": job.stage,
        "created_at": job.created_at,
    }


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Query current job status, progress percentage, stage description, and logs."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return job.to_dict()


@router.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download the resulting cleaned MP4 video."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed yet (current status: {job.status}).",
        )

    if not job.result_video_path or not os.path.exists(job.result_video_path):
        raise HTTPException(status_code=404, detail="Processed video file not found on server.")

    return FileResponse(
        job.result_video_path,
        media_type="video/mp4",
        filename=f"object_removed_{job_id[:8]}.mp4",
    )


@router.get("/video/{filename}")
async def stream_video(filename: str):
    """Stream uploaded or processed video file for HTML5 video playback."""
    # Check uploads directory first, then processed directory
    filepath = UPLOADS_DIR / filename
    if not filepath.exists():
        filepath = PROCESSED_DIR / filename

    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="Requested video file not found.")

    return FileResponse(filepath, media_type="video/mp4")
