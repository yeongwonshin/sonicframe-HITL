from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ensure_workspace
from .models import ProjectState


class ProjectStore:
    """JSON-backed project storage for production backend runs and local review sessions."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or ensure_workspace()
        self.projects_dir = self.root / "projects"
        self.exports_dir = self.root / "exports"
        self.uploads_dir = self.root / "uploads"
        for path in [self.projects_dir, self.exports_dir, self.uploads_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def project_path(self, project_id: str) -> Path:
        return self.projects_dir / f"{project_id}.json"

    def save(self, project: ProjectState) -> ProjectState:
        project.updated_at = datetime.now(timezone.utc)
        self.project_path(project.id).write_text(
            project.model_dump_json(indent=2), encoding="utf-8"
        )
        return project

    def load(self, project_id: str) -> ProjectState:
        path = self.project_path(project_id)
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return ProjectState.model_validate_json(path.read_text(encoding="utf-8"))

    def list_projects(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.projects_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                project = ProjectState.model_validate_json(path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "id": project.id,
                        "filename": project.video.filename,
                        "duration": project.video.duration,
                        "events": len(project.timeline.events) if project.timeline else 0,
                        "feedback": len(project.feedback_logs),
                        "updated_at": project.updated_at.isoformat(),
                    }
                )
            except Exception:
                continue
        return rows

    def write_json_export(self, project: ProjectState, filename: str, payload: Any) -> str:
        path = self.exports_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        project.artifacts[filename] = str(path)
        self.save(project)
        return str(path)
