from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import CandidateSound, SoundEvent, SoundTimeline

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None


class ProceduralAudioEngine:
    """Fast procedural renderer for previewable hackathon demos.

    Each timeline event is synthesized locally so the project can run without paid APIs.
    Real deployments can swap this class for Foley libraries, diffusion audio, or DAWs.
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
        # crude high-pass sweep without scipy
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
        # constant-power panning
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
