from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .models import SceneSegment, VideoAsset, VisualEvent
from .vision_backends import VisionBackendError, VisionCascade, build_vision_cascade_from_env

try:  # OpenCV is required at runtime for production video analysis.
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


class VideoAnalysisError(RuntimeError):
    """Raised when production video analysis cannot complete."""


class VideoAnalyzer:
    """Production visual-event extractor powered by YOLO/GroundingDINO/SAM/VLM.

    The previous demo path created synthetic or motion-heuristic visual events.
    That behavior has been removed: every VisualEvent now comes from the mandatory
    production cascade. OpenCV is still used for frame sampling and scene grouping,
    but not for object or event labeling.
    """

    def __init__(
        self,
        sample_fps: int = 6,
        scene_threshold: float = 0.32,
        vision_cascade: VisionCascade | None = None,
    ) -> None:
        if cv2 is None:
            raise VideoAnalysisError("OpenCV is required for production video analysis")
        self.sample_fps = max(1, sample_fps)
        self.scene_threshold = scene_threshold
        self.vision_cascade = vision_cascade if vision_cascade is not None else build_vision_cascade_from_env()

    def analyze(self, video_path: str | Path) -> tuple[VideoAsset, list[SceneSegment], list[VisualEvent]]:
        path = Path(video_path)
        if not path.exists():
            raise VideoAnalysisError(f"Video file does not exist: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoAnalysisError(f"OpenCV cannot open video file: {path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
            step = max(1, int(round(fps / self.sample_fps)))

            video = VideoAsset(filename=path.name, path=str(path), duration=duration, fps=fps, width=width, height=height)
            samples = self._sample_frames(cap, fps, step)
        finally:
            cap.release()

        if not samples:
            raise VideoAnalysisError("No frames were sampled from the video")

        if duration <= 0:
            duration = float(samples[-1]["time"]) + 1.0 / self.sample_fps
            video.duration = duration

        scenes = self._build_scenes(samples, duration)
        visual_events = self._build_events_with_production_cascade(samples, scenes)
        visual_events = self._merge_close_events(visual_events)
        if not visual_events:
            raise VisionBackendError(
                "YOLO/GroundingDINO/SAM/VLM completed but returned no sound-worthy VisualEvents; "
                "adjust prompts, thresholds, or VLM event policy instead of using a heuristic fallback."
            )
        return video, scenes, visual_events

    def _sample_frames(self, cap: Any, fps: float, step: int) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
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
            timestamp = idx / fps if fps else len(samples) / self.sample_fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (96, 54)) if gray.size else gray
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            scene_diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)) if prev_hist is not None else 0.0
            motion = float(np.mean(cv2.absdiff(prev_gray, small)) / 255.0) if prev_gray is not None else 0.0
            samples.append(
                {
                    "time": timestamp,
                    "hist": hist,
                    "scene_diff": scene_diff,
                    "motion": motion,
                    "frame": frame.copy(),
                }
            )
            prev_gray = small
            prev_hist = hist
            idx += 1
        return samples

    def _build_scenes(self, samples: list[dict[str, Any]], duration: float) -> list[SceneSegment]:
        cut_times = [0.0]
        for sample in samples[1:]:
            if float(sample["scene_diff"]) >= self.scene_threshold:
                t = float(sample["time"])
                if t - cut_times[-1] > 0.75:
                    cut_times.append(t)
        cut_times.append(duration)
        scenes: list[SceneSegment] = []
        for start, end in zip(cut_times[:-1], cut_times[1:]):
            scene_samples = [s for s in samples if start <= float(s["time"]) < end]
            motion_mean = float(np.mean([float(s["motion"]) for s in scene_samples])) if scene_samples else 0.0
            diff_mean = float(np.mean([float(s["scene_diff"]) for s in scene_samples])) if scene_samples else 0.0
            style_hint = "quiet" if motion_mean < 0.08 else "impact" if motion_mean > 0.25 else "neutral"
            scenes.append(
                SceneSegment(
                    start=start,
                    end=max(end, start + 0.1),
                    motion_mean=motion_mean,
                    visual_energy=motion_mean + diff_mean,
                    style_hint=style_hint,
                )
            )
        return scenes or [SceneSegment(start=0.0, end=max(duration, 0.1))]

    def _build_events_with_production_cascade(
        self, samples: list[dict[str, Any]], scenes: list[SceneSegment]
    ) -> list[VisualEvent]:
        events: list[VisualEvent] = []
        for sample in samples:
            timestamp = float(sample["time"])
            scene = self._scene_for(timestamp, scenes)
            frame = sample["frame"]
            frame_events = self.vision_cascade.analyze_frame(frame, timestamp=timestamp, scene_id=scene.id if scene else None)
            events.extend(frame_events)
        return sorted(events, key=lambda event: event.start)

    def _scene_for(self, timestamp: float, scenes: list[SceneSegment]) -> SceneSegment | None:
        for scene in scenes:
            if scene.start <= timestamp < scene.end:
                return scene
        return scenes[-1] if scenes else None

    def _merge_close_events(self, events: list[VisualEvent]) -> list[VisualEvent]:
        if not events:
            return []
        merged: list[VisualEvent] = []
        for event in sorted(events, key=lambda e: (e.start, e.evidence.object_label, e.evidence.event_type)):
            previous = self._find_merge_target(merged, event)
            if previous is None:
                merged.append(event)
                continue
            previous.end = max(previous.end, event.end)
            previous.evidence.confidence = max(previous.evidence.confidence, event.evidence.confidence)
            previous.evidence.mask_area = max(
                previous.evidence.mask_area or 0.0,
                event.evidence.mask_area or 0.0,
            ) or None
            previous.evidence.source = "+".join(
                dict.fromkeys(previous.evidence.source.split("+") + event.evidence.source.split("+"))
            )
            previous.evidence.notes.extend(event.evidence.notes)
        return sorted(merged, key=lambda event: event.start)

    def _find_merge_target(self, existing: list[VisualEvent], event: VisualEvent) -> VisualEvent | None:
        for candidate in reversed(existing):
            if event.start - candidate.end > 0.35:
                break
            same_object = candidate.evidence.object_label == event.evidence.object_label
            same_type = candidate.evidence.event_type == event.evidence.event_type
            overlaps = self._iou(candidate.evidence.bbox, event.evidence.bbox) >= 0.55
            if same_object and same_type and overlaps:
                return candidate
        return None

    def _iou(self, a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> float:
        if not a or not b:
            return 0.0
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0
