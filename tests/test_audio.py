from pathlib import Path

from sonicframe_hitl.audio import ProceduralAudioEngine
from sonicframe_hitl.models import Explanation, SoundEvent, SoundTimeline


def test_render_timeline_creates_wav(tmp_path: Path):
    event = SoundEvent(
        start=0.1,
        end=0.4,
        label="door contact",
        sound_type="contact",
        object_label="door",
        volume=0.5,
        intensity=0.7,
        explanation=Explanation(visual_reason="x", planning_reason="y"),
    )
    timeline = SoundTimeline(video_id="v1", duration=1.0, events=[event])
    out = tmp_path / "mix.wav"
    path = ProceduralAudioEngine(sample_rate=16000).render_timeline(timeline, out)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 1000
