import os
import shutil
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Espresso_Analysis import (
    build_workspace,
    denormalize_ellipse,
    ellipse_to_dict,
    normalize_ellipse,
    preload_detection_models,
    prepare_video_review,
    run_full_analysis,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_STORAGE_DIR = os.path.join(BASE_DIR, "api_storage")
SESSIONS_DIR = os.path.join(API_STORAGE_DIR, "sessions")
STAGE_TOTAL = 7


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _safe_remove_tree(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


@dataclass
class AnalysisSession:
    session_id: str
    root_dir: str
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    status: str = "created"
    message: str = "Session created."
    stage_index: int = 0
    stage_total: int = STAGE_TOTAL
    stage_key: str = "created"
    stage_label: str = "Created"
    error_code: str | None = None
    video_path: str | None = None
    source: str = "unknown"
    frame_width: int | None = None
    frame_height: int | None = None
    preview: dict[str, Any] | None = None
    approved_roi_source: str | None = None
    approved_ellipse: Any | None = None
    results_payload: dict[str, Any] | None = None
    raw_results_json: dict[str, Any] | None = None
    worker: threading.Thread | None = None

    def to_dict(self):
        payload = {
            "job_id": self.session_id,
            "session_id": self.session_id,
            "status": self.status,
            "message": self.message,
            "stage_index": self.stage_index,
            "stage_total": self.stage_total,
            "stage_key": self.stage_key,
            "stage_label": self.stage_label,
            "progress": float(self.stage_index) / float(self.stage_total or 1),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.preview:
            payload.update(self.preview)
        if self.results_payload:
            payload.update(self.results_payload)
            payload["result_payload"] = self.results_payload
        if self.raw_results_json:
            payload["results_json"] = self.raw_results_json
        return payload


class StartAnalysisRequest(BaseModel):
    job_id: str
    roi_ellipse: dict[str, Any] | None = None
    roi_source: str | None = "auto"


app = FastAPI(title="EspExAnalyser API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(SESSIONS_DIR, exist_ok=True)
_sessions: dict[str, AnalysisSession] = {}
_sessions_lock = threading.Lock()


@app.on_event("startup")
def _preload_models_on_startup():
    threading.Thread(target=preload_detection_models, daemon=True).start()


def _set_stage(
    session: AnalysisSession,
    *,
    status: str,
    message: str,
    stage_index: int,
    stage_key: str,
    stage_label: str,
    error_code: str | None = None,
):
    session.status = status
    session.message = message
    session.stage_index = stage_index
    session.stage_key = stage_key
    session.stage_label = stage_label
    session.updated_at = _utc_now_iso()
    session.error_code = error_code


def _get_session_or_404(job_id: str):
    with _sessions_lock:
        session = _sessions.get(job_id)
    if session is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown job_id: {job_id}"})
    return session


def _session_workspace(session: AnalysisSession):
    return build_workspace(session.root_dir, include_video_dir=False)


def _build_results_payload(session: AnalysisSession, run_output: dict[str, Any]):
    results_json = run_output.get("results_json") or {}
    feature_results = run_output.get("feature_results") or {}
    tracking_result = run_output.get("tracking_result") or {}
    fps = float(results_json.get("fps") or feature_results.get("fps") or 1.0)
    start_frame = int(results_json.get("flow_start", feature_results.get("start_frame", 0)) or 0)
    end_frame = int(results_json.get("flow_end", feature_results.get("end_frame", start_frame)) or start_frame)
    shot_time_seconds = max(0.0, float(end_frame - start_frame + 1) / max(1.0, fps))

    quality = results_json.get("quality") or {}
    model_prediction = results_json.get("model_prediction") or {}
    diagnostics = results_json.get("diagnostics") or {}
    combined_assessment = results_json.get("combined_assessment") or {}
    confidence = model_prediction.get("confidence")
    if confidence is None:
        confidence = quality.get("overall_score", 0.0)

    payload = {
        "job_id": session.session_id,
        "session_id": session.session_id,
        "video_name": results_json.get("video_name") or run_output.get("video_name"),
        "fps": fps,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "blond_frame": results_json.get("blond_frame"),
        "shot_time_seconds": shot_time_seconds,
        "confidence": float(confidence or 0.0),
        "quality_score": float(quality.get("overall_score", 0.0)) * 10.0,
        "brightness_curve": results_json.get("brightness_curve") or feature_results.get("brightness_curve") or [],
        "channeling_counts": results_json.get("channeling_counts") or feature_results.get("channeling_counts") or [],
        "roi_source": session.approved_roi_source or "auto",
        "analysis_roi": results_json.get("analysis_roi"),
        "portafilter_ellipse": results_json.get("portafilter_ellipse")
        or ellipse_to_dict(tracking_result.get("ellipse_in_crop")),
        "diagnostics": diagnostics,
        "quality": quality,
        "model_prediction": model_prediction,
        "combined_assessment": combined_assessment,
        "tracking": tracking_result,
        "results_json_path": run_output.get("results_json_path"),
    }

    if session.approved_ellipse and session.frame_width and session.frame_height:
        payload["roi_ellipse"] = normalize_ellipse(
            session.approved_ellipse,
            session.frame_width,
            session.frame_height,
        )

    return payload


def _run_analysis_worker(session_id: str):
    session = _get_session_or_404(session_id)
    try:
        workspace = _session_workspace(session)

        def progress_callback(*, stage_index: int, stage_key: str, stage_label: str, message: str):
            _set_stage(
                session,
                status="processing",
                message=message,
                stage_index=stage_index,
                stage_key=stage_key,
                stage_label=stage_label,
            )

        run_output = run_full_analysis(
            video_path=session.video_path,
            workspace=workspace,
            approved_ellipse=session.approved_ellipse,
            approved_mode_size=(
                session.preview.get("auto_mode_size")
                if session.approved_roi_source == "auto" and session.preview
                else None
            ),
            approved_fast_params=(
                session.preview.get("auto_fast_params")
                if session.approved_roi_source == "auto" and session.preview
                else None
            ),
            capture_channeling_frames=False,
            progress_callback=progress_callback,
            save_crops_to_disk=False,
        )
        session.raw_results_json = run_output.get("results_json") or {}
        session.results_payload = _build_results_payload(session, run_output)
        _set_stage(
            session,
            status="completed",
            message="Analysis completed successfully.",
            stage_index=7,
            stage_key="completed",
            stage_label="Completed",
        )
    except Exception as exc:
        traceback.print_exc()
        _set_stage(
            session,
            status="failed",
            message=f"Analysis failed: {exc}",
            stage_index=max(session.stage_index, 3),
            stage_key="failed",
            stage_label="Failed",
            error_code="E-ANALYSIS-FAILED",
        )


@app.get("/health")
def health():
    return {"status": "ok", "service": "EspExAnalyser API"}


@app.post("/api/analyse-video/upload")
async def upload_video(video: UploadFile = File(...), source: str = "mobile_app"):
    extension = os.path.splitext(video.filename or "")[1] or ".mp4"
    session_id = uuid.uuid4().hex
    session_root = os.path.join(SESSIONS_DIR, session_id)
    workspace = _session_workspace(AnalysisSession(session_id=session_id, root_dir=session_root))
    os.makedirs(session_root, exist_ok=True)

    stored_video_path = os.path.join(session_root, f"uploaded{extension}")
    with open(stored_video_path, "wb") as file_ref:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            file_ref.write(chunk)
    await video.close()

    session = AnalysisSession(
        session_id=session_id,
        root_dir=session_root,
        source=source,
        video_path=stored_video_path,
    )
    _set_stage(
        session,
        status="processing",
        message="Extracting preview frames and detecting the ROI...",
        stage_index=1,
        stage_key="extracting_preview",
        stage_label="Preparing Preview",
    )
    with _sessions_lock:
        _sessions[session_id] = session

    try:
        review = prepare_video_review(stored_video_path, workspace)
    except Exception as exc:
        _safe_remove_tree(session_root)
        with _sessions_lock:
            _sessions.pop(session_id, None)
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "E-ROI-DETECTION",
                "message": f"Could not prepare ROI review: {exc}",
            },
        ) from exc

    session.frame_width = int(review["frame_width"])
    session.frame_height = int(review["frame_height"])
    session.preview = {
        "auto_roi_ellipse": review["ellipse_normalized"],
        "auto_roi_ellipse_pixels": review["ellipse_dict"],
        "auto_mode_size": review["mode_size"],
        "auto_fast_params": review["fast_params"],
        "frame_width": session.frame_width,
        "frame_height": session.frame_height,
    }
    _set_stage(
        session,
        status="roi_pending",
        message="Video uploaded. Review and confirm the ROI to continue.",
        stage_index=2,
        stage_key="roi_detected",
        stage_label="ROI Review",
    )
    return session.to_dict()


@app.post("/api/analyse-video/start")
def start_analysis(request: StartAnalysisRequest):
    session = _get_session_or_404(request.job_id)
    if session.status == "completed":
        return session.to_dict()
    if session.worker is not None and session.worker.is_alive():
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "E-ANALYSIS-RUNNING",
                "message": "Analysis is already running for this job.",
            },
        )
    if session.video_path is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "E-ANALYSIS-NO-VIDEO",
                "message": "This session has no uploaded video.",
            },
        )

    if request.roi_ellipse is not None:
        approved_ellipse = denormalize_ellipse(
            request.roi_ellipse,
            frame_width=session.frame_width or 1,
            frame_height=session.frame_height or 1,
        )
    else:
        ellipse_dict = (session.preview or {}).get("auto_roi_ellipse_pixels")
        approved_ellipse = denormalize_ellipse(
            ellipse_dict,
            frame_width=session.frame_width or 1,
            frame_height=session.frame_height or 1,
        )

    session.approved_ellipse = approved_ellipse
    session.approved_roi_source = (request.roi_source or "auto").strip().lower() or "auto"
    _set_stage(
        session,
        status="processing",
        message="ROI confirmed. Starting analysis...",
        stage_index=3,
        stage_key="roi_confirmed",
        stage_label="ROI Confirmed",
    )
    worker = threading.Thread(
        target=_run_analysis_worker,
        args=(session.session_id,),
        daemon=True,
        name=f"analysis-{session.session_id}",
    )
    session.worker = worker
    worker.start()
    return session.to_dict()


@app.get("/api/analyse-video/status")
def analysis_status(job_id: str):
    session = _get_session_or_404(job_id)
    return session.to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("Api_Server:app", host="0.0.0.0", port=8000, reload=False)
