from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .models import VisualEvent

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class ObjectDetection:
    """Unified detection output shared by YOLO, GroundingDINO, SAM and hosted VLMs."""

    label: str
    bbox: tuple[int, int, int, int]
    confidence: float = 0.5
    source: str = "detector"
    event_type: str | None = None
    mask_area: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


class ObjectDetector(Protocol):
    def detect(self, frame_bgr: Any, prompts: Sequence[str]) -> list[ObjectDetection]: ...


class Segmenter(Protocol):
    def segment(self, frame_bgr: Any, detection: ObjectDetection) -> ObjectDetection: ...


class VisualLanguageModel(Protocol):
    def describe(self, frame_bgr: Any, detection: ObjectDetection, event: VisualEvent) -> dict[str, Any]: ...


class VisionCascade:
    """Detector -> segmenter -> VLM refiner.

    The cascade is deliberately optional. Without external model dependencies the
    VideoAnalyzer still produces deterministic motion events. In production, pass
    adapters for YOLO/GroundingDINO, SAM and a VLM to upgrade labels, masks and
    action descriptions while keeping the same VisualEvent schema.
    """

    def __init__(
        self,
        detector: ObjectDetector | None = None,
        segmenter: Segmenter | None = None,
        vlm: VisualLanguageModel | None = None,
        prompts: Sequence[str] | None = None,
        min_confidence: float = 0.25,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.vlm = vlm
        self.prompts = list(prompts or DEFAULT_PROMPTS)
        self.min_confidence = min_confidence

    def refine_event(self, frame_bgr: Any, event: VisualEvent) -> VisualEvent:
        if self.detector is None:
            return event
        detections = [d for d in self.detector.detect(frame_bgr, self.prompts) if d.confidence >= self.min_confidence]
        if not detections:
            return event

        best = self._select_detection(event, detections)
        if self.segmenter is not None:
            best = self.segmenter.segment(frame_bgr, best)
        vlm_payload: dict[str, Any] = {}
        if self.vlm is not None:
            vlm_payload = self.vlm.describe(frame_bgr, best, event) or {}

        refined = event.model_copy(deep=True)
        ev = refined.evidence
        previous_source = ev.source
        ev.object_label = str(vlm_payload.get("object_label") or best.label or ev.object_label)
        ev.event_type = str(vlm_payload.get("event_type") or best.event_type or ev.event_type)
        ev.bbox = tuple(best.bbox)  # type: ignore[assignment]
        ev.mask_area = best.mask_area
        ev.confidence = max(ev.confidence, min(1.0, 0.55 * ev.confidence + 0.45 * best.confidence))
        ev.source = self._source_name(best.source)
        ev.attributes.update(best.attributes)
        ev.attributes.update({k: v for k, v in vlm_payload.items() if k not in {"object_label", "event_type"}})
        ev.notes.append(f"vision cascade refined {previous_source} -> {ev.source}")
        if "description" in vlm_payload:
            ev.notes.append(f"vlm: {vlm_payload['description']}")
        return refined

    def _select_detection(self, event: VisualEvent, detections: list[ObjectDetection]) -> ObjectDetection:
        event_box = event.evidence.bbox
        if event_box:
            return max(detections, key=lambda d: (self._iou(event_box, d.bbox), d.confidence))
        return max(detections, key=lambda d: d.confidence)

    def _iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
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

    def _source_name(self, detector_source: str) -> str:
        names = [detector_source]
        if self.segmenter:
            names.append("sam")
        if self.vlm:
            names.append("vlm")
        return "+".join(names)


class HostedDetectorAdapter:
    """HTTP detector adapter for GroundingDINO/YOLO services.

    Expected response JSON:
    {"detections": [{"label": "door", "bbox": [x,y,w,h], "confidence": 0.81}]}
    """

    def __init__(self, endpoint: str, source: str = "hosted_detector", timeout: float = 15.0) -> None:
        self.endpoint = endpoint
        self.source = source
        self.timeout = timeout

    def detect(self, frame_bgr: Any, prompts: Sequence[str]) -> list[ObjectDetection]:
        import requests

        payload = {"image_b64": _encode_frame_b64(frame_bgr), "prompts": list(prompts)}
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        detections = []
        for raw in data.get("detections", []):
            bbox = tuple(int(v) for v in raw.get("bbox", [0, 0, 0, 0])[:4])
            detections.append(
                ObjectDetection(
                    label=str(raw.get("label", "unknown_object")),
                    bbox=bbox,  # type: ignore[arg-type]
                    confidence=float(raw.get("confidence", 0.5)),
                    source=str(raw.get("source", self.source)),
                    event_type=raw.get("event_type"),
                    attributes=dict(raw.get("attributes", {})),
                )
            )
        return detections


class HostedSegmenterAdapter:
    """HTTP SAM adapter. Expected response can include mask_area and/or refined bbox."""

    def __init__(self, endpoint: str, timeout: float = 20.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def segment(self, frame_bgr: Any, detection: ObjectDetection) -> ObjectDetection:
        import requests

        payload = {"image_b64": _encode_frame_b64(frame_bgr), "bbox": list(detection.bbox)}
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        refined = ObjectDetection(**detection.__dict__)
        if "bbox" in data:
            refined.bbox = tuple(int(v) for v in data["bbox"][:4])  # type: ignore[assignment]
        if "mask_area" in data:
            refined.mask_area = float(data["mask_area"])
        refined.attributes.update(dict(data.get("attributes", {})))
        return refined


class HostedVLMAdapter:
    """HTTP VLM adapter. Returns object/action/context fields used by VisionCascade."""

    def __init__(self, endpoint: str, timeout: float = 20.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def describe(self, frame_bgr: Any, detection: ObjectDetection, event: VisualEvent) -> dict[str, Any]:
        import requests

        payload = {
            "image_b64": _encode_frame_b64(frame_bgr),
            "bbox": list(detection.bbox),
            "detected_label": detection.label,
            "motion_event": event.model_dump(mode="json"),
        }
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return dict(data)


class YOLODetectorAdapter:
    """Optional local YOLO adapter using ultralytics when installed.

    This class is not imported by default, so the base project does not require the
    heavy ultralytics dependency. Set SONICFRAME_VISION_BACKEND=yolo and provide
    SONICFRAME_YOLO_WEIGHTS in an environment that has ultralytics installed.
    """

    def __init__(self, weights: str = "yolov8n.pt") -> None:
        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(weights)

    def detect(self, frame_bgr: Any, prompts: Sequence[str]) -> list[ObjectDetection]:
        results = self.model(frame_bgr, verbose=False)
        detections: list[ObjectDetection] = []
        for result in results:
            names = getattr(result, "names", {})
            for box in getattr(result, "boxes", []):
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                cls_id = int(box.cls[0])
                detections.append(
                    ObjectDetection(
                        label=str(names.get(cls_id, f"class_{cls_id}")),
                        bbox=(x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                        confidence=float(box.conf[0]),
                        source="yolo",
                    )
                )
        return detections


class PromptedGroundingDINOAdapter(HostedDetectorAdapter):
    """Semantic alias for a hosted GroundingDINO detector service."""

    def __init__(self, endpoint: str, timeout: float = 20.0) -> None:
        super().__init__(endpoint=endpoint, source="groundingdino", timeout=timeout)


DEFAULT_PROMPTS = [
    "person",
    "foot",
    "hand",
    "door",
    "vehicle",
    "glass",
    "metal object",
    "wood object",
    "falling object",
    "moving object",
    "background ambience",
]


def build_vision_cascade_from_env() -> VisionCascade | None:
    """Build an optional cascade without making heavy dependencies mandatory."""

    backend = os.getenv("SONICFRAME_VISION_BACKEND", "").strip().lower()
    detector: ObjectDetector | None = None
    if backend == "yolo":
        detector = YOLODetectorAdapter(os.getenv("SONICFRAME_YOLO_WEIGHTS", "yolov8n.pt"))
    elif backend in {"groundingdino", "dino"} and os.getenv("SONICFRAME_DETECTOR_ENDPOINT"):
        detector = PromptedGroundingDINOAdapter(os.environ["SONICFRAME_DETECTOR_ENDPOINT"])
    elif os.getenv("SONICFRAME_DETECTOR_ENDPOINT"):
        detector = HostedDetectorAdapter(os.environ["SONICFRAME_DETECTOR_ENDPOINT"])

    if detector is None:
        return None

    segmenter = HostedSegmenterAdapter(os.environ["SONICFRAME_SAM_ENDPOINT"]) if os.getenv("SONICFRAME_SAM_ENDPOINT") else None
    vlm = HostedVLMAdapter(os.environ["SONICFRAME_VLM_ENDPOINT"]) if os.getenv("SONICFRAME_VLM_ENDPOINT") else None
    prompts = [p.strip() for p in os.getenv("SONICFRAME_VISION_PROMPTS", "").split(",") if p.strip()] or DEFAULT_PROMPTS
    return VisionCascade(detector=detector, segmenter=segmenter, vlm=vlm, prompts=prompts)


def _encode_frame_b64(frame_bgr: Any) -> str:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to encode frames for hosted vision adapters")
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
