from pathlib import Path
import wave

import numpy as np
import pytest

from sonicframe_hitl.audio import AudioBackendConfigurationError, FoleyAssetEngine, FoleyAssetNotFound, build_audio_engine_from_env
from sonicframe_hitl.models import Explanation, SoundEvent, SoundTimeline


def _write_test_wav(path: Path, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 0.25, int(sample_rate * 0.25), endpoint=False)
    signal = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    stereo = np.column_stack([signal, signal]).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(stereo.tobytes())


def _event() -> SoundEvent:
    return SoundEvent(
        start=0.1,
        end=0.4,
        label="door contact",
        sound_type="contact",
        object_label="door",
        volume=0.5,
        intensity=0.7,
        explanation=Explanation(visual_reason="x", planning_reason="y"),
    )


def test_foley_engine_renders_timeline_from_assets(tmp_path: Path):
    _write_test_wav(tmp_path / "foley" / "contact" / "door" / "hit.wav")
    timeline = SoundTimeline(video_id="v1", duration=1.0, events=[_event()])
    out = tmp_path / "mix.wav"
    path = FoleyAssetEngine(tmp_path / "foley", sample_rate=16000).render_timeline(timeline, out)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 1000


def test_foley_engine_fails_when_asset_missing(tmp_path: Path):
    (tmp_path / "foley").mkdir()
    with pytest.raises(FoleyAssetNotFound):
        FoleyAssetEngine(tmp_path / "foley", sample_rate=16000).synthesize_event(_event())


def test_audio_engine_env_requires_production_backend(monkeypatch):
    monkeypatch.delenv("SONICFRAME_AUDIO_BACKEND", raising=False)
    with pytest.raises(AudioBackendConfigurationError):
        build_audio_engine_from_env(sample_rate=16000)
