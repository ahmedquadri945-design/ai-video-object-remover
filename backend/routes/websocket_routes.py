import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.job_manager import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])


@router.websocket("/ws/{job_id}")
async def websocket_job_progress(websocket: WebSocket, job_id: str):
    """WebSocket connection that streams real-time progress updates and logs for a specific job."""
    await websocket.accept()
    logger.info(f"WebSocket client connected for job {job_id}")
    await job_manager.register_websocket(job_id, websocket)

    try:
        while True:
            # Keep connection open and listen for optional ping/client commands
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from job {job_id}")
    except Exception as e:
        logger.warning(f"WebSocket error for job {job_id}: {e}")
    finally:
        await job_manager.unregister_websocket(job_id, websocket)
