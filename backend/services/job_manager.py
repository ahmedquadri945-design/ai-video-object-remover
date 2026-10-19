import asyncio
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Set, Optional, Any, List
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    UPLOADING = "uploading"
    PENDING = "pending"
    TRACKING = "tracking"
    INPAINTING = "inpainting"
    REMUXING = "remuxing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    def __init__(
        self,
        job_id: str,
        video_id: str,
        prompt_type: str = "brush",
        prompt_data: Optional[Any] = None,
        settings: Optional[Dict[str, Any]] = None,
    ):
        self.job_id = job_id
        self.video_id = video_id
        self.status = JobStatus.PENDING
        self.progress = 0.0
        self.stage = "Job initialized"
        self.prompt_type = prompt_type
        self.prompt_data = prompt_data or {}
        self.settings = settings or {}
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.error: Optional[str] = None
        self.result_video_url: Optional[str] = None
        self.result_video_path: Optional[str] = None
        self.logs: List[Dict[str, str]] = [
            {"time": self.created_at, "message": "Job created and queued."}
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "video_id": self.video_id,
            "status": self.status.value if isinstance(self.status, JobStatus) else str(self.status),
            "progress": round(self.progress, 1),
            "stage": self.stage,
            "prompt_type": self.prompt_type,
            "settings": self.settings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "result_video_url": self.result_video_url,
            "result_url": self.result_video_url,
            "logs": self.logs,
        }


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    def create_job(
        self,
        job_id: str,
        video_id: str,
        prompt_type: str = "brush",
        prompt_data: Optional[Any] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Job:
        job = Job(
            job_id=job_id,
            video_id=video_id,
            prompt_type=prompt_type,
            prompt_data=prompt_data,
            settings=settings,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    async def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        stage: Optional[str] = None,
        error: Optional[str] = None,
        result_video_url: Optional[str] = None,
        result_video_path: Optional[str] = None,
        log_msg: Optional[str] = None,
    ) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if not job:
            return None

        now = datetime.now(timezone.utc).isoformat()
        job.updated_at = now

        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = max(0.0, min(100.0, float(progress)))
        if stage is not None:
            job.stage = stage
        if error is not None:
            job.error = error
        if result_video_url is not None:
            job.result_video_url = result_video_url
        if result_video_path is not None:
            job.result_video_path = result_video_path
        if log_msg:
            job.logs.append({"time": now, "message": log_msg})

        # Broadcast update to connected WebSockets
        await self.broadcast(job_id, job.to_dict())
        return job

    async def register_websocket(self, job_id: str, websocket: WebSocket):
        async with self._lock:
            if job_id not in self._connections:
                self._connections[job_id] = set()
            self._connections[job_id].add(websocket)

        # Send current job status immediately upon connection if job exists
        job = self.get_job(job_id)
        if job:
            try:
                await websocket.send_json(job.to_dict())
            except Exception as e:
                logger.warning(f"Failed to send initial job state to websocket: {e}")

    async def unregister_websocket(self, job_id: str, websocket: WebSocket):
        async with self._lock:
            if job_id in self._connections:
                self._connections[job_id].discard(websocket)
                if not self._connections[job_id]:
                    del self._connections[job_id]

    async def broadcast(self, job_id: str, message: Dict[str, Any]):
        subscribers = set()
        async with self._lock:
            if job_id in self._connections:
                subscribers = set(self._connections[job_id])

        disconnected = set()
        for ws in subscribers:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.add(ws)

        if disconnected:
            async with self._lock:
                if job_id in self._connections:
                    self._connections[job_id].difference_update(disconnected)


# Global job manager singleton instance
job_manager = JobManager()
