from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sonicframe_hitl.audio import build_audio_engine_from_env
from sonicframe_hitl.config import ensure_workspace, sample_fps
from sonicframe_hitl.exporters import export_project_bundle
from sonicframe_hitl.feedback import FeedbackInterpreter
from sonicframe_hitl.models import FeedbackLog, ProjectState, SoundStyle
from sonicframe_hitl.planner import SoundPlanner
from sonicframe_hitl.storage import ProjectStore
from sonicframe_hitl.video_analysis import VideoAnalyzer

app = FastAPI(
    title="SonicFrame HITL",
    description="Production Human-in-the-loop Video-to-Audio Sound Design System with mandatory vision/audio backends",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = ProjectStore(ensure_workspace())
planner = SoundPlanner()
interpreter = FeedbackInterpreter()
engine = build_audio_engine_from_env()


class FeedbackRequest(BaseModel):
    action: str
    target_event_id: str | None = None
    scene_id: str | None = None
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    text: str | None = None
    replan: bool = True


class CandidateChoiceRequest(BaseModel):
    candidate_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sonicframe-hitl"}


@app.get("/projects")
def list_projects() -> list[dict[str, Any]]:
    return store.list_projects()


@app.post("/analyze")
def analyze_video(
    file: UploadFile = File(...),
    style: str = Form("balanced"),
    render_audio: bool = Form(True),
) -> ProjectState:
    upload_path = store.uploads_dir / safe_filename(file.filename or "video.mp4")
    with upload_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    analyzer = VideoAnalyzer(sample_fps=sample_fps())
    video, scenes, visual_events = analyzer.analyze(upload_path)
    profile_style = SoundStyle(style)
    project = ProjectState(video=video, scenes=scenes, visual_events=visual_events)
    project.profile.default_style = profile_style
    project.timeline = planner.plan(video, scenes, visual_events, project.profile, style=profile_style)
    project.candidates = [c for e in project.timeline.events[:8] for c in planner.make_candidates(e, project.profile)]
    if render_audio:
        wav = store.exports_dir / f"{project.id}_mix.wav"
        engine.render_timeline(project.timeline, wav)
        project.artifacts["mix_wav"] = str(wav)
    return store.save(project)


@app.get("/projects/{project_id}")
def get_project(project_id: str) -> ProjectState:
    return load_or_404(project_id)


@app.get("/projects/{project_id}/timeline")
def get_timeline(project_id: str) -> dict[str, Any]:
    project = load_or_404(project_id)
    if not project.timeline:
        raise HTTPException(status_code=404, detail="Timeline not generated")
    return project.timeline.model_dump()


@app.post("/projects/{project_id}/feedback")
def add_feedback(project_id: str, payload: FeedbackRequest) -> ProjectState:
    project = load_or_404(project_id)
    log = FeedbackLog(
        project_id=project.id,
        action=payload.action,  # type: ignore[arg-type]
        target_event_id=payload.target_event_id,
        scene_id=payload.scene_id,
        before=payload.before,
        after=payload.after,
        text=payload.text,
    )
    project.feedback_logs.append(log)
    project.profile = interpreter.update_profile(project.profile, [log], timeline=project.timeline)
    if payload.replan:
        project.timeline = planner.replan_with_feedback(project.video, project.scenes, project.visual_events, project.profile)
        project.candidates = [c for e in project.timeline.events[:8] for c in planner.make_candidates(e, project.profile)]
    return store.save(project)


@app.post("/projects/{project_id}/candidates")
def regenerate_candidates(project_id: str) -> ProjectState:
    project = load_or_404(project_id)
    if not project.timeline:
        raise HTTPException(status_code=404, detail="Timeline not generated")
    project.candidates = [c for event in project.timeline.events[:12] for c in planner.make_candidates(event, project.profile)]
    # Create production backend previews for the first candidates.
    by_event = {e.id: e for e in project.timeline.events}
    for cand in project.candidates[:12]:
        event = by_event.get(cand.event_id)
        if event:
            path = store.exports_dir / f"{project.id}_{cand.id}.wav"
            cand.preview_path = engine.render_candidate_preview(event, cand, path)
    return store.save(project)


@app.post("/projects/{project_id}/choose-candidate")
def choose_candidate(project_id: str, payload: CandidateChoiceRequest) -> ProjectState:
    project = load_or_404(project_id)
    if not project.timeline:
        raise HTTPException(status_code=404, detail="Timeline not generated")
    candidate = next((c for c in project.candidates if c.id == payload.candidate_id), None)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    project.timeline = planner.apply_candidate(project.timeline, candidate)
    log = FeedbackLog(
        project_id=project.id,
        action="choose_candidate",
        target_event_id=candidate.event_id,
        after={"candidate_id": candidate.id, "variant_name": candidate.variant_name, "style": candidate.style.value},
    )
    project.feedback_logs.append(log)
    project.profile = interpreter.update_profile(project.profile, [log], project.timeline)
    return store.save(project)


@app.post("/projects/{project_id}/render")
def render_mix(project_id: str) -> dict[str, str]:
    project = load_or_404(project_id)
    if not project.timeline:
        raise HTTPException(status_code=404, detail="Timeline not generated")
    wav = store.exports_dir / f"{project.id}_mix.wav"
    path = engine.render_timeline(project.timeline, wav)
    project.artifacts["mix_wav"] = path
    store.save(project)
    return {"mix_wav": path}


@app.get("/projects/{project_id}/download/mix")
def download_mix(project_id: str) -> FileResponse:
    project = load_or_404(project_id)
    path = project.artifacts.get("mix_wav")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Rendered mix not found")
    return FileResponse(path, filename=Path(path).name, media_type="audio/wav")


@app.post("/projects/{project_id}/export")
def export_project(project_id: str) -> dict[str, str]:
    project = load_or_404(project_id)
    out_dir = store.exports_dir / f"{project.id}_bundle"
    files = export_project_bundle(project, out_dir)
    project.artifacts.update(files)
    store.save(project)
    return files


def load_or_404(project_id: str) -> ProjectState:
    try:
        return store.load(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def safe_filename(name: str) -> str:
    keep = [c if c.isalnum() or c in {".", "-", "_"} else "_" for c in name]
    return "".join(keep)[:160] or "video.mp4"
