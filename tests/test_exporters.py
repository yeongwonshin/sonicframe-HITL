from pathlib import Path

from sonicframe_hitl.exporters import export_project_bundle
from sonicframe_hitl.models import ProjectState, SoundTimeline, VideoAsset


def test_export_project_bundle(tmp_path: Path):
    project = ProjectState(video=VideoAsset(filename="a.mp4", path="a.mp4", duration=1.0))
    project.timeline = SoundTimeline(video_id=project.video.id, duration=1.0)
    files = export_project_bundle(project, tmp_path)
    assert "project_json" in files
    assert "profile_json" in files
    assert Path(files["manifest"]).exists()
