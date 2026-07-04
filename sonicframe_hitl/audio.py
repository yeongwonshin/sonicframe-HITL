from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from .models import CandidateSound, SoundEvent, SoundTimeline

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None


class AudioRenderBackend(Protocol):
    def render_timeline(self, timeline: SoundTimeline, output_path: str | Path) -> str: ...

    def render_candidate_preview(self, event: SoundEvent, candidate: CandidateSound, output_path: str | Path) -> str: ...


class ProceduralAudioEngine:
    """Fast procedural renderer for previewable demos.

    The class remains the reliable offline fallback. New production paths are
    exposed through FoleyAssetEngine and HostedGenerativeAudioEngine while keeping
    this same public interface, so API/CLI code can swap engines by environment.
    """

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
        audio = self._limiter(audio)
        return self.write_wav(audio, output_path)

    def render_candidate_preview(self, event: SoundEvent, candidate: CandidateSound, output_path: str | Path) -> str:
        preview_event = event.model_copy(deep=True)
        preview_event.volume = candidate.volume
        preview_event.intensity = candidate.intensity
        preview_event.style = candidate.style
        clip = self.synthesize_event(preview_event, min_duration=1.2)
        clip = self._limiter(clip)
        return self.write_wav(clip, output_path)

    def synthesize_event(self, event: SoundEvent, min_duration: float = 0.0) -> np.ndarray:
        duration = max(min_duration, event.end - event.start, 0.05)
        n = max(16, int(duration * self.sample_rate))
        if event.sound_type == "footstep":
            mono = self._footstep(n, event.intensity)
        elif event.sound_type == "contact":
            mono = self._impact(n, event.intensity)
        elif event.sound_type == "ambient":
            mono = self._ambient(n, event.intensity)
        else:
            mono = self._motion_whoosh(n, event.intensity)
        mono *= float(event.volume)
        return self._pan(mono.astype(np.float32), event.pan)

    def write_wav(self, audio: np.ndarray, output_path: str | Path) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if sf is None:
            self._write_wav_stdlib(audio, path)
        else:
            sf.write(str(path), audio, self.sample_rate)
        return str(path)

    def _impact(self, n: int, intensity: float) -> np.ndarray:
        rng = np.random.default_rng(12345 + n)
        t = np.arange(n) / self.sample_rate
        env = np.exp(-t * (14 + intensity * 8))
        thump = np.sin(2 * np.pi * (70 + 40 * intensity) * t) * np.exp(-t * 18)
        crack = rng.normal(0, 1, n) * np.exp(-t * 55)
        body = 0.7 * thump + 0.28 * crack
        if intensity > 0.9:
            body += 0.12 * np.sin(2 * np.pi * 180 * t) * env
        return self._normalize(body * env * (0.8 + intensity * 0.5))

    def _footstep(self, n: int, intensity: float) -> np.ndarray:
        rng = np.random.default_rng(54321 + n)
        t = np.arange(n) / self.sample_rate
        heel_env = np.exp(-t * 35)
        toe_env = np.exp(-np.maximum(0, t - 0.09) * 45) * (t > 0.09)
        heel = np.sin(2 * np.pi * (95 + 25 * intensity) * t) * heel_env
        grit = rng.normal(0, 0.55, n) * (heel_env + 0.6 * toe_env)
        toe = np.sin(2 * np.pi * (160 + 20 * intensity) * t) * toe_env
        return self._normalize((0.65 * heel + 0.22 * toe + 0.18 * grit) * (0.55 + intensity * 0.45))

    def _motion_whoosh(self, n: int, intensity: float) -> np.ndarray:
        rng = np.random.default_rng(777 + n)
        t = np.linspace(0, 1, n, endpoint=False)
        noise = rng.normal(0, 1, n)
        hp = noise - np.concatenate([[0], noise[:-1]])
        sweep = np.sin(2 * np.pi * (120 + 500 * t**1.6) * np.arange(n) / self.sample_rate)
        env = np.sin(np.pi * t) ** 1.6
        return self._normalize((0.62 * hp + 0.22 * sweep) * env * (0.35 + intensity * 0.5))

    def _ambient(self, n: int, intensity: float) -> np.ndarray:
        rng = np.random.default_rng(999 + n)
        t = np.arange(n) / self.sample_rate
        noise = rng.normal(0, 1, n)
        smoothed = np.cumsum(noise)
        smoothed -= np.mean(smoothed)
        smoothed = smoothed / (np.max(np.abs(smoothed)) + 1e-6)
        hum = np.sin(2 * np.pi * 55 * t) * 0.08
        env = np.ones(n)
        fade = min(n // 4, int(0.3 * self.sample_rate))
        if fade > 0:
            env[:fade] = np.linspace(0, 1, fade)
            env[-fade:] = np.linspace(1, 0, fade)
        return self._normalize((0.22 * smoothed + hum) * env * (0.25 + intensity * 0.25))

    def _pan(self, mono: np.ndarray, pan: float) -> np.ndarray:
        pan = max(-1.0, min(1.0, pan))
        left = np.cos((pan + 1) * np.pi / 4)
        right = np.sin((pan + 1) * np.pi / 4)
        return np.stack([mono * left, mono * right], axis=1)

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        return signal / peak * 0.85 if peak > 1e-6 else signal

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


class FoleyAssetEngine(ProceduralAudioEngine):
    """Renderer that prefers curated Foley assets and falls back to procedural audio.

    Directory convention examples:
      assets/foley/contact/door/*.wav
      assets/foley/footstep/person/*.wav
      assets/foley/motion/*.wav

    This gives a concrete path beyond procedural preview without forcing any paid
    API dependency into the base repository.
    """

    def __init__(self, asset_dir: str | Path, sample_rate: int = 44100) -> None:
        super().__init__(sample_rate=sample_rate)
        self.asset_dir = Path(asset_dir)

    def synthesize_event(self, event: SoundEvent, min_duration: float = 0.0) -> np.ndarray:
        asset = self._find_asset(event)
        if asset is None:
            return super().synthesize_event(event, min_duration=min_duration)
        try:
            clip = self._load_asset(asset)
        except Exception:
            return super().synthesize_event(event, min_duration=min_duration)
        clip = self._fit_duration(clip, max(min_duration, event.end - event.start, 0.05))
        clip = self._apply_gain_and_pan(clip, event.volume, event.intensity, event.pan)
        return clip.astype(np.float32)

    def _find_asset(self, event: SoundEvent) -> Path | None:
        safe_object = event.object_label.replace("/", "_")
        candidates = [
            self.asset_dir / event.sound_type / safe_object,
            self.asset_dir / event.sound_type,
            self.asset_dir / safe_object,
        ]
        files: list[Path] = []
        for folder in candidates:
            if folder.exists():
                files.extend(sorted(folder.glob("*.wav")))
        if not files:
            return None
        idx = abs(hash((event.sound_type, event.object_label, event.id))) % len(files)
        return files[idx]

    def _load_asset(self, path: Path) -> np.ndarray:
        if sf is not None:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            audio = np.asarray(data, dtype=np.float32)
        else:
            import wave

            with wave.open(str(path), "rb") as f:
                sr = f.getframerate()
                channels = f.getnchannels()
                frames = f.readframes(f.getnframes())
            pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
            audio = pcm.reshape(-1, channels)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]
        return self._resample(audio, int(sr)) if int(sr) != self.sample_rate else audio

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
        mono = clip.mean(axis=1) * volume * (0.65 + 0.35 * intensity)
        return self._pan(mono.astype(np.float32), pan)


class HostedGenerativeAudioEngine(ProceduralAudioEngine):
    """Adapter for diffusion/audio-generation backends.

    The endpoint may return raw WAV bytes or JSON {"wav_b64": "..."}. If the
    backend is unavailable, the event automatically falls back to ProceduralAudioEngine.
    """

    def __init__(self, endpoint: str, sample_rate: int = 44100, timeout: float = 45.0) -> None:
        super().__init__(sample_rate=sample_rate)
        self.endpoint = endpoint
        self.timeout = timeout

    def synthesize_event(self, event: SoundEvent, min_duration: float = 0.0) -> np.ndarray:
        try:
            import base64
            import requests

            prompt = self._prompt_for_event(event)
            payload = {
                "prompt": prompt,
                "duration": max(min_duration, event.end - event.start, 0.05),
                "sample_rate": self.sample_rate,
                "metadata": event.metadata,
            }
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            if response.headers.get("content-type", "").startswith("audio/"):
                wav_bytes = response.content
            else:
                wav_bytes = base64.b64decode(response.json()["wav_b64"])
            return self._decode_wav_bytes(wav_bytes, event)
        except Exception:
            return super().synthesize_event(event, min_duration=min_duration)

    def _prompt_for_event(self, event: SoundEvent) -> str:
        return (
            f"{event.style.value} Foley sound for {event.object_label} {event.sound_type}, "
            f"intensity {event.intensity:.2f}, clean sync point, no music, no speech"
        )

    def _decode_wav_bytes(self, wav_bytes: bytes, event: SoundEvent) -> np.ndarray:
        if sf is not None:
            data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
            audio = np.asarray(data, dtype=np.float32)
            if audio.shape[1] == 1:
                audio = np.repeat(audio, 2, axis=1)
            elif audio.shape[1] > 2:
                audio = audio[:, :2]
            audio = FoleyAssetEngine(".", self.sample_rate)._resample(audio, int(sr)) if int(sr) != self.sample_rate else audio
            return self._limiter(audio * event.volume)
        return super().synthesize_event(event)


def build_audio_engine_from_env(sample_rate: int = 44100) -> AudioRenderBackend:
    backend = os.getenv("SONICFRAME_AUDIO_BACKEND", "procedural").strip().lower()
    if backend in {"foley", "assets", "asset"} and os.getenv("SONICFRAME_FOLEY_DIR"):
        return FoleyAssetEngine(os.environ["SONICFRAME_FOLEY_DIR"], sample_rate=sample_rate)
    if backend in {"hosted", "generative", "diffusion"} and os.getenv("SONICFRAME_AUDIO_ENDPOINT"):
        return HostedGenerativeAudioEngine(os.environ["SONICFRAME_AUDIO_ENDPOINT"], sample_rate=sample_rate)
    return ProceduralAudioEngine(sample_rate=sample_rate)
