from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import ConfigurationError, env_float, required_env
from .models import CandidateSound, SoundEvent, SoundTimeline

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None


class AudioBackendError(RuntimeError):
    """Raised when a required production audio backend cannot render audio."""


class AudioBackendConfigurationError(ConfigurationError):
    """Raised when Foley or generative audio backend configuration is incomplete."""


class FoleyAssetNotFound(AudioBackendError):
    """Raised when no curated Foley asset matches a requested event."""


class AudioRenderBackend(Protocol):
    def render_timeline(self, timeline: SoundTimeline, output_path: str | Path) -> str: ...

    def render_candidate_preview(self, event: SoundEvent, candidate: CandidateSound, output_path: str | Path) -> str: ...


class BaseAudioEngine:
    """Shared WAV mixing/output helpers for production audio engines."""

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate

    def render_timeline(self, timeline: SoundTimeline, output_path: str | Path) -> str:
        duration = max(timeline.duration, max((e.end for e in timeline.events), default=0.0) + 0.2)
        audio = np.zeros((int(duration * self.sample_rate) + 1, 2), dtype=np.float32)
        for event in timeline.events:
            clip = self.synthesize_event(event)
            start = int(event.start * self.sample_rate)
            end = min(audio.shape[0], start + clip.shape[0])
            if end > start:
                audio[start:end] += clip[: end - start]
        return self.write_wav(self._limiter(audio), output_path)

    def render_candidate_preview(self, event: SoundEvent, candidate: CandidateSound, output_path: str | Path) -> str:
        preview_event = event.model_copy(deep=True)
        preview_event.volume = candidate.volume
        preview_event.intensity = candidate.intensity
        preview_event.style = candidate.style
        preview_event.metadata = dict(preview_event.metadata)
        preview_event.metadata["candidate"] = candidate.model_dump(mode="json")
        clip = self.synthesize_event(preview_event, min_duration=1.2)
        return self.write_wav(self._limiter(clip), output_path)

    def synthesize_event(self, event: SoundEvent, min_duration: float = 0.0) -> np.ndarray:
        raise NotImplementedError

    def write_wav(self, audio: np.ndarray, output_path: str | Path) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if sf is not None:
            sf.write(str(path), audio, self.sample_rate)
        else:
            self._write_wav_stdlib(audio, path)
        return str(path)

    def _decode_wav_bytes(self, wav_bytes: bytes) -> np.ndarray:
        if sf is not None:
            data, source_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
            audio = np.asarray(data, dtype=np.float32)
        else:
            import wave

            with wave.open(io.BytesIO(wav_bytes), "rb") as f:
                source_rate = f.getframerate()
                channels = f.getnchannels()
                frames = f.readframes(f.getnframes())
            pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
            audio = pcm.reshape(-1, channels)
        audio = self._ensure_stereo(audio)
        return self._resample(audio, int(source_rate)) if int(source_rate) != self.sample_rate else audio

    def _ensure_stereo(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.shape[1] == 1:
            return np.repeat(audio, 2, axis=1).astype(np.float32)
        if audio.shape[1] > 2:
            return audio[:, :2].astype(np.float32)
        return audio.astype(np.float32)

    def _resample(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate <= 0 or audio.size == 0:
            return audio
        old_x = np.linspace(0.0, 1.0, audio.shape[0], endpoint=False)
        new_len = max(1, int(audio.shape[0] * self.sample_rate / source_rate))
        new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)
        left = np.interp(new_x, old_x, audio[:, 0])
        right = np.interp(new_x, old_x, audio[:, 1])
        return np.stack([left, right], axis=1).astype(np.float32)

    def _fit_duration(self, clip: np.ndarray, duration: float) -> np.ndarray:
        target = max(16, int(duration * self.sample_rate))
        if clip.shape[0] >= target:
            fitted = clip[:target].copy()
        else:
            fitted = np.zeros((target, 2), dtype=np.float32)
            fitted[: clip.shape[0]] = clip
        fade = min(target // 8, int(0.04 * self.sample_rate))
        if fade > 0:
            fitted[:fade] *= np.linspace(0, 1, fade)[:, None]
            fitted[-fade:] *= np.linspace(1, 0, fade)[:, None]
        return fitted

    def _apply_gain_and_pan(self, clip: np.ndarray, volume: float, intensity: float, pan: float) -> np.ndarray:
        mono = clip.mean(axis=1) * float(volume) * (0.65 + 0.35 * float(intensity))
        return self._pan(mono.astype(np.float32), pan)

    def _pan(self, mono: np.ndarray, pan: float) -> np.ndarray:
        pan = max(-1.0, min(1.0, pan))
        left = np.cos((pan + 1) * np.pi / 4)
        right = np.sin((pan + 1) * np.pi / 4)
        return np.stack([mono * left, mono * right], axis=1)

    def _limiter(self, audio: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.98:
            audio = audio / peak * 0.98
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def _write_wav_stdlib(self, audio: np.ndarray, path: Path) -> None:
        import wave

        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as f:
            f.setnchannels(2)
            f.setsampwidth(2)
            f.setframerate(self.sample_rate)
            f.writeframes(pcm.tobytes())


class FoleyAssetEngine(BaseAudioEngine):
    """Strict curated Foley renderer.

    Unlike the former demo renderer, this class never synthesizes a procedural
    fallback. Missing or unreadable assets are production errors that should be
    fixed in the Foley library or routing policy.
    """

    def __init__(self, asset_dir: str | Path, sample_rate: int = 44100) -> None:
        super().__init__(sample_rate=sample_rate)
        self.asset_dir = Path(asset_dir)
        if not self.asset_dir.exists():
            raise AudioBackendConfigurationError(f"Foley asset directory does not exist: {self.asset_dir}")

    def synthesize_event(self, event: SoundEvent, min_duration: float = 0.0) -> np.ndarray:
        asset = self._find_asset(event)
        if asset is None:
            raise FoleyAssetNotFound(
                f"No Foley asset found for sound_type={event.sound_type!r}, object_label={event.object_label!r}"
            )
        try:
            clip = self._load_asset(asset)
        except Exception as exc:
            raise AudioBackendError(f"Failed to load Foley asset {asset}: {exc}") from exc
        duration = max(min_duration, event.end - event.start, 0.05)
        clip = self._fit_duration(clip, duration)
        return self._apply_gain_and_pan(clip, event.volume, event.intensity, event.pan).astype(np.float32)

    def _find_asset(self, event: SoundEvent) -> Path | None:
        safe_object = _safe_path_token(event.object_label)
        safe_type = _safe_path_token(event.sound_type)
        extra_tags = [_safe_path_token(str(tag)) for tag in event.metadata.get("foley_tags", []) if str(tag).strip()]
        candidates = [
            self.asset_dir / safe_type / safe_object,
            self.asset_dir / safe_type,
            self.asset_dir / safe_object,
        ]
        candidates.extend(self.asset_dir / safe_type / tag for tag in extra_tags)
        files: list[Path] = []
        for folder in candidates:
            if folder.exists():
                files.extend(sorted(folder.glob("*.wav")))
        if not files:
            return None
        idx = abs(hash((event.sound_type, event.object_label, event.id, event.style.value))) % len(files)
        return files[idx]

    def _load_asset(self, path: Path) -> np.ndarray:
        return self._decode_wav_bytes(path.read_bytes())


class HostedGenerativeAudioEngine(BaseAudioEngine):
    """Strict adapter for AudioLDM/Stable Audio/custom generative backends."""

    def __init__(self, endpoint: str, sample_rate: int = 44100, timeout: float = 45.0) -> None:
        super().__init__(sample_rate=sample_rate)
        if not endpoint:
            raise AudioBackendConfigurationError("Generative audio endpoint is required")
        self.endpoint = endpoint
        self.timeout = timeout

    def synthesize_event(self, event: SoundEvent, min_duration: float = 0.0) -> np.ndarray:
        import requests

        duration = max(min_duration, event.end - event.start, 0.05)
        payload = {
            "prompt": self._prompt_for_event(event),
            "duration": duration,
            "sample_rate": self.sample_rate,
            "event": event.model_dump(mode="json"),
            "constraints": {
                "no_music": True,
                "no_speech": True,
                "sync_point_seconds": max(0.0, event.start),
            },
        }
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        wav_bytes = self._extract_wav_bytes(response)
        clip = self._decode_wav_bytes(wav_bytes)
        clip = self._fit_duration(clip, duration)
        return self._apply_gain_and_pan(clip, event.volume, event.intensity, event.pan).astype(np.float32)

    def _prompt_for_event(self, event: SoundEvent) -> str:
        texture = event.metadata.get("visual_attributes", {}).get("material") or event.metadata.get("texture")
        texture_text = f", {texture} texture" if texture else ""
        return (
            f"{event.style.value} production Foley for {event.object_label} {event.sound_type}{texture_text}, "
            f"intensity {event.intensity:.2f}, clean transient, synchronized to picture, no music, no speech"
        )

    def _extract_wav_bytes(self, response: object) -> bytes:
        content_type = getattr(response, "headers", {}).get("content-type", "")
        if content_type.startswith("audio/") or content_type in {"application/octet-stream", "audio/wav"}:
            return getattr(response, "content")
        data = response.json()  # type: ignore[attr-defined]
        if "wav_b64" in data:
            return base64.b64decode(data["wav_b64"])
        if "audio_b64" in data:
            return base64.b64decode(data["audio_b64"])
        raise AudioBackendError("Generative audio backend must return audio bytes, wav_b64, or audio_b64")


class HybridProductionAudioEngine(BaseAudioEngine):
    """Foley-first, generative-repair production engine without procedural fallback."""

    def __init__(self, foley: FoleyAssetEngine, generative: HostedGenerativeAudioEngine, sample_rate: int = 44100) -> None:
        super().__init__(sample_rate=sample_rate)
        self.foley = foley
        self.generative = generative

    def synthesize_event(self, event: SoundEvent, min_duration: float = 0.0) -> np.ndarray:
        try:
            return self.foley.synthesize_event(event, min_duration=min_duration)
        except FoleyAssetNotFound:
            return self.generative.synthesize_event(event, min_duration=min_duration)


def build_audio_engine_from_env(sample_rate: int = 44100) -> AudioRenderBackend:
    backend = os.getenv("SONICFRAME_AUDIO_BACKEND", "").strip().lower()
    if not backend:
        raise AudioBackendConfigurationError(
            "SONICFRAME_AUDIO_BACKEND must be set to 'foley', 'generative', or 'hybrid'. Procedural demo fallback was removed."
        )
    if backend in {"foley", "assets", "asset"}:
        return FoleyAssetEngine(required_env("SONICFRAME_FOLEY_DIR"), sample_rate=sample_rate)
    if backend in {"hosted", "generative", "diffusion"}:
        return HostedGenerativeAudioEngine(
            required_env("SONICFRAME_AUDIO_ENDPOINT"),
            sample_rate=sample_rate,
            timeout=env_float("SONICFRAME_AUDIO_TIMEOUT", 60.0),
        )
    if backend == "hybrid":
        foley = FoleyAssetEngine(required_env("SONICFRAME_FOLEY_DIR"), sample_rate=sample_rate)
        generative = HostedGenerativeAudioEngine(
            required_env("SONICFRAME_AUDIO_ENDPOINT"),
            sample_rate=sample_rate,
            timeout=env_float("SONICFRAME_AUDIO_TIMEOUT", 60.0),
        )
        return HybridProductionAudioEngine(foley, generative, sample_rate=sample_rate)
    raise AudioBackendConfigurationError(f"Unsupported SONICFRAME_AUDIO_BACKEND={backend!r}")


def _safe_path_token(value: str) -> str:
    token = value.strip().lower().replace(" ", "_").replace("/", "_")
    return "".join(ch for ch in token if ch.isalnum() or ch in {"_", "-"}) or "unknown"
