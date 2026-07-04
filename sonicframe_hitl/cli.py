from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .audio import build_audio_engine_from_env
from .config import ensure_workspace, sample_fps
from .exporters import export_project_bundle
from .feedback import FeedbackInterpreter
from .models import FeedbackLog, ProjectState, SoundStyle
from .planner import SoundPlanner
from .storage import ProjectStore
from .video_analysis import VideoAnalyzer

app = typer.Typer(help="SonicFrame HITL command line interface")
console = Console()


@app.command()
def analyze(video_path: Path, style: SoundStyle = SoundStyle.balanced, render: bool = True) -> None:
    """Analyze a video, plan an explainable sound timeline, and optionally render a WAV."""
    store = ProjectStore(ensure_workspace())
    analyzer = VideoAnalyzer(sample_fps=sample_fps())
    planner = SoundPlanner()
    video, scenes, visual_events = analyzer.analyze(video_path)
    project = ProjectState(video=video, scenes=scenes, visual_events=visual_events)
    project.profile.default_style = style
    project.timeline = planner.plan(video, scenes, visual_events, project.profile, style=style)
    project.candidates = [c for event in project.timeline.events[:8] for c in planner.make_candidates(event, project.profile)]
    if render:
        wav = store.exports_dir / f"{project.id}_mix.wav"
        project.artifacts["mix_wav"] = build_audio_engine_from_env().render_timeline(project.timeline, wav)
    store.save(project)
    console.print(f"[bold green]Project created:[/] {project.id}")
    show_timeline(project.id)


@app.command("show")
def show_timeline(project_id: str) -> None:
    """Print a project timeline with explanations."""
    store = ProjectStore(ensure_workspace())
    project = store.load(project_id)
    table = Table(title=f"SoundTimeline {project_id}")
    table.add_column("time")
    table.add_column("label")
    table.add_column("vol")
    table.add_column("why")
    if not project.timeline:
        console.print("No timeline")
        return
    for event in project.timeline.events:
        table.add_row(
            f"{event.start:.2f}-{event.end:.2f}",
            event.label,
            f"{event.volume:.2f}",
            event.explanation.visual_reason[:90] + "...",
        )
    console.print(table)
    console.print(project.timeline.global_explanation)


@app.command()
def feedback(project_id: str, text: str, target_event_id: str | None = None) -> None:
    """Add natural-language feedback and replan."""
    store = ProjectStore(ensure_workspace())
    project = store.load(project_id)
    log = FeedbackLog(project_id=project.id, action="text_feedback", target_event_id=target_event_id, text=text)
    project.feedback_logs.append(log)
    project.profile = FeedbackInterpreter().update_profile(project.profile, [log], project.timeline)
    project.timeline = SoundPlanner().replan_with_feedback(project.video, project.scenes, project.visual_events, project.profile)
    store.save(project)
    console.print(f"[bold green]Feedback applied:[/] {text}")
    console.print(FeedbackInterpreter().summarize_profile(project.profile))


@app.command()
def render(project_id: str) -> None:
    """Render the current SoundTimeline to a WAV mix."""
    store = ProjectStore(ensure_workspace())
    project = store.load(project_id)
    if not project.timeline:
        raise typer.BadParameter("Project has no timeline")
    wav = store.exports_dir / f"{project.id}_mix.wav"
    path = build_audio_engine_from_env().render_timeline(project.timeline, wav)
    project.artifacts["mix_wav"] = path
    store.save(project)
    console.print(f"[bold green]Rendered:[/] {path}")


@app.command("export")
def export_cmd(project_id: str) -> None:
    """Export JSON/CSV/profile/submission note bundle."""
    store = ProjectStore(ensure_workspace())
    project = store.load(project_id)
    files = export_project_bundle(project, store.exports_dir / f"{project.id}_bundle")
    project.artifacts.update(files)
    store.save(project)
    for key, value in files.items():
        console.print(f"{key}: {value}")


if __name__ == "__main__":
    app()
