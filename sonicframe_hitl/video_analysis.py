from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .models import SceneSegment, VideoAsset, VisualEvidence, VisualEvent

try:  # OpenCV is optional at import time to keep CLI metadata commands usable.
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


class VideoAnalyzer:
    """
    Lightweight visual-event extractor.

    It does not try to replace a large vision model. For hackathon execution it provides fast,
    deterministic signals: scene cuts, motion energy, coarse object boxes, contact-like peaks.
    The planner can be swapped later with GroundingDINO, SAM, YOLO, or a hosted VLM.
    """

    def __init__(self, sample_fps: int = 6, scene_threshold: float = 0.32) -> None:
        self.sample_fps = max(1, sample_fps)
        self.scene_threshold = scene_threshold

    def analyze(self, video_path: str | Path) -> tuple[VideoAsset, list[SceneSegment], list[VisualEvent]]:
        path = Path(video_path)
        if cv2 is None:
            return self._fallback(path)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return self._fallback(path)

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        step = max(1, int(round(fps / self.sample_fps)))

        video = VideoAsset(filename=path.name, path=str(path), duration=duration, fps=fps, width=width, height=height)
        samples: list[dict[str, float | np.ndarray | None]] = []
        prev_gray: np.ndarray | None = None
        prev_hist: np.ndarray | None = None
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue
            t = idx / fps if fps else len(samples) / self.sample_fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (96, 54)) if gray.size else gray
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            scene_diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)) if prev_hist is not None else 0.0
            motion = float(np.mean(cv2.absdiff(prev_gray, small)) / 255.0) if prev_gray is not None else 0.0
            samples.append({"time": t, "gray": small, "hist": hist, "scene_diff": scene_diff, "motion": motion})
            prev_gray = small
            prev_hist = hist
            idx += 1
        cap.release()

        if not samples:
            return self._fallback(path)

        scenes = self._build_scenes(samples, duration)
        events = self._build_events(samples, scenes, width, height)
        return video, scenes, events

    def _fallback(self, path: Path) -> tuple[VideoAsset, list[SceneSegment], list[VisualEvent]]:
        video = VideoAsset(filename=path.name, path=str(path), duration=8.0, fps=24.0, width=1280, height=720)
        scenes = [
            SceneSegment(start=0.0, end=2.8, motion_mean=0.35, visual_energy=0.4, style_hint="quiet"),
            SceneSegment(start=2.8, end=5.2, motion_mean=0.75, visual_energy=0.8, style_hint="impact"),
            SceneSegment(start=5.2, end=8.0, motion_mean=0.45, visual_energy=0.5, style_hint="ambient"),
        ]
        events = [
            VisualEvent(
                start=1.0,
                end=1.35,
                evidence=VisualEvidence(
                    object_label="person", event_type="footstep", confidence=0.62, motion_score=0.42,
                    scene_id=scenes[0].id, notes=["fallback synthetic walking cue"]
                ),
            ),
            VisualEvent(
                start=3.1,
                end=3.55,
                evidence=VisualEvidence(
                    object_label="door", event_type="contact", confidence=0.7, motion_score=0.78, contact_score=0.86,
                    scene_id=scenes[1].id, notes=["fallback synthetic contact cue"]
                ),
            ),
            VisualEvent(
                start=6.0,
                end=6.8,
                evidence=VisualEvidence(
                    object_label="background", event_type="motion", confidence=0.55, motion_score=0.45,
                    scene_id=scenes[2].id, notes=["fallback synthetic ambient cue"]
                ),
            ),
        ]
        return video, scenes, events

    def _build_scenes(self, samples: list[dict[str, object]], duration: float) -> list[SceneSegment]:
        cut_times = [0.0]
        for sample in samples[1:]:
            if float(sample["scene_diff"]) >= self.scene_threshold:
                t = float(sample["time"])
                if t - cut_times[-1] > 0.75:
                    cut_times.append(t)
        if duration <= 0:
            duration = float(samples[-1]["time"]) + 1 / self.sample_fps
        cut_times.append(duration)
        scenes: list[SceneSegment] = []
        for start, end in zip(cut_times[:-1], cut_times[1:]):
            scene_samples = [s for s in samples if start <= float(s["time"]) < end]
            motion_mean = float(np.mean([float(s["motion"]) for s in scene_samples])) if scene_samples else 0.0
            diff_mean = float(np.mean([float(s["scene_diff"]) for s in scene_samples])) if scene_samples else 0.0
            hint = "quiet" if motion_mean < 0.08 else "impact" if motion_mean > 0.25 else "neutral"
            scenes.append(SceneSegment(start=start, end=max(end, start + 0.1), motion_mean=motion_mean, visual_energy=motion_mean + diff_mean, style_hint=hint))
        return scenes

    def _build_events(
        self, samples: list[dict[str, object]], scenes: list[SceneSegment], width: int, height: int
    ) -> list[VisualEvent]:
        motions = np.array([float(s["motion"]) for s in samples])
        if len(motions) < 2:
            return []
        threshold = max(0.004, float(np.mean(motions) + 0.75 * np.std(motions)))
        events: list[VisualEvent] = []
        i = 1
        while i < len(samples):
            if float(samples[i]["motion"]) < threshold:
                i += 1
                continue
            start_idx = i
            peak = float(samples[i]["motion"])
            while i < len(samples) and float(samples[i]["motion"]) >= threshold * 0.65:
                peak = max(peak, float(samples[i]["motion"]))
                i += 1
            end_idx = max(start_idx + 1, i - 1)
            start = max(0.0, float(samples[start_idx]["time"]) - 0.05)
            end = float(samples[end_idx]["time"]) + 0.18
            scene = self._scene_for(start, scenes)
            contact_score = self._contact_score(samples, start_idx, end_idx)
            object_label = self._object_guess(peak, contact_score, scene.style_hint)
            event_type = self._event_type(peak, contact_score, scene.style_hint)
            bbox = self._motion_bbox(samples, start_idx, width, height)
            confidence = min(0.95, 0.58 + peak * 5.0 + contact_score * 0.25)
            events.append(
                VisualEvent(
                    start=start,
                    end=end,
                    evidence=VisualEvidence(
                        object_label=object_label,
                        event_type=event_type,
                        bbox=bbox,
                        confidence=confidence,
                        motion_score=peak,
                        contact_score=contact_score,
                        scene_id=scene.id,
                        notes=[
                            f"motion peak={peak:.3f}",
                            f"contact score={contact_score:.3f}",
                            "bbox estimated from frame difference",
                        ],
                    ),
                )
            )
            i += 1
        return self._merge_close_events(events)

    def _scene_for(self, t: float, scenes: list[SceneSegment]) -> SceneSegment:
        for scene in scenes:
            if scene.start <= t < scene.end:
                return scene
        return scenes[-1]

    def _contact_score(self, samples: list[dict[str, object]], start_idx: int, end_idx: int) -> float:
        local = [float(samples[j]["motion"]) for j in range(max(1, start_idx - 1), min(len(samples), end_idx + 2))]
        if len(local) < 3:
            return 0.0
        return max(0.0, min(1.0, (max(local) - np.median(local)) * 18.0))

    def _event_type(self, peak: float, contact: float, style_hint: str) -> str:
        if contact > 0.2 or peak > 0.28 or style_hint == "impact":
            return "contact"
        if 0.08 < peak < 0.22:
            return "footstep"
        return "motion"

    def _object_guess(self, peak: float, contact: float, style_hint: str) -> str:
        if contact > 0.2 or style_hint == "impact":
            return "door_or_prop"
        if 0.08 < peak < 0.22:
            return "person"
        if math.isclose(peak, 0.0):
            return "background"
        return "moving_object"

    def _motion_bbox(self, samples: list[dict[str, object]], idx: int, width: int, height: int) -> tuple[int, int, int, int] | None:
        if cv2 is None or idx <= 0:
            return None
        prev_gray = samples[idx - 1].get("gray")
        gray = samples[idx].get("gray")
        if not isinstance(prev_gray, np.ndarray) or not isinstance(gray, np.ndarray):
            return None
        diff = cv2.absdiff(prev_gray, gray)
        _, mask = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        scale_x = width / max(1, gray.shape[1])
        scale_y = height / max(1, gray.shape[0])
        return (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))

    def _merge_close_events(self, events: list[VisualEvent]) -> list[VisualEvent]:
        if not events:
            return []
        merged: list[VisualEvent] = [events[0]]
        for ev in events[1:]:
            prev = merged[-1]
            same_scene = prev.evidence.scene_id == ev.evidence.scene_id
            same_kind = prev.evidence.event_type == ev.evidence.event_type
            if same_scene and same_kind and ev.start - prev.end < 0.25:
                prev.end = max(prev.end, ev.end)
                if ev.evidence.motion_score > prev.evidence.motion_score:
                    prev.evidence.motion_score = ev.evidence.motion_score
                    prev.evidence.bbox = ev.evidence.bbox
                prev.evidence.contact_score = max(prev.evidence.contact_score, ev.evidence.contact_score)
                prev.evidence.notes.extend(ev.evidence.notes)
            else:
                merged.append(ev)
        return merged
