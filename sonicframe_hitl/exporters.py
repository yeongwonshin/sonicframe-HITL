from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import ProjectState, SoundTimeline


def timeline_to_rows(timeline: SoundTimeline) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in timeline.events:
        rows.append(
            {
                "id": event.id,
                "start": round(event.start, 3),
                "end": round(event.end, 3),
                "label": event.label,
                "sound_type": event.sound_type,
                "object": event.object_label,
                "volume": round(event.volume, 3),
                "intensity": round(event.intensity, 3),
                "style": event.style.value,
                "visual_reason": event.explanation.visual_reason,
                "feedback_reason": event.explanation.feedback_reason or "",
                "planning_reason": event.explanation.planning_reason,
            }
        )
    return rows


def export_project_bundle(project: ProjectState, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    project_json = out / f"{project.id}_project.json"
    project_json.write_text(project.model_dump_json(indent=2), encoding="utf-8")
    files["project_json"] = str(project_json)
    if project.timeline:
        timeline_json = out / f"{project.id}_timeline.json"
        timeline_json.write_text(project.timeline.model_dump_json(indent=2), encoding="utf-8")
        files["timeline_json"] = str(timeline_json)
        timeline_csv = out / f"{project.id}_timeline.csv"
        rows = timeline_to_rows(project.timeline)
        with timeline_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
            writer.writeheader()
            writer.writerows(rows)
        files["timeline_csv"] = str(timeline_csv)
    profile_json = out / f"{project.id}_profile.json"
    profile_json.write_text(project.profile.model_dump_json(indent=2), encoding="utf-8")
    files["profile_json"] = str(profile_json)
    readme = out / f"{project.id}_submission_note.md"
    readme.write_text(_submission_note(project), encoding="utf-8")
    files["submission_note"] = str(readme)
    manifest = out / f"{project.id}_manifest.json"
    manifest.write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8")
    files["manifest"] = str(manifest)
    return files


def _submission_note(project: ProjectState) -> str:
    events = len(project.timeline.events) if project.timeline else 0
    return f"""# SonicFrame HITL Export

- Project: `{project.id}`
- Video: `{project.video.filename}`
- Duration: {project.video.duration:.2f}s
- Visual events: {len(project.visual_events)}
- Sound events: {events}
- Feedback logs: {len(project.feedback_logs)}

이 번들은 해커톤 제출용으로 생성된 설명 가능한 영상-사운드 타임라인, 사용자 선호 프로필, 편집 로그를 포함합니다.
"""
