from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Sequence

from .config import ConfigurationError, csv_env, env_float, env_int, required_env
from .models import VisualEvidence, VisualEvent

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


class VisionBackendError(RuntimeError):
    """Raised when a required production vision backend fails."""


class VisionBackendConfigurationError(ConfigurationError):
    """Raised when YOLO/GroundingDINO/SAM/VLM configuration is incomplete."""


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
    def describe(self, frame_bgr: Any, detection: ObjectDetection, timestamp: float, scene_id: str | None) -> dict[str, Any]: ...


class VisionCascade:
    """Mandatory production cascade: YOLO + GroundingDINO -> SAM -> VLM.

    This class intentionally has no motion-heuristic or synthetic-event fallback.
    If any required backend is missing or fails, the caller receives an exception
    instead of silently continuing with demo behavior.
    """

    def __init__(
        self,
        yolo_detector: ObjectDetector,
        grounding_dino_detector: ObjectDetector,
        segmenter: Segmenter,
        vlm: VisualLanguageModel,
        prompts: Sequence[str] | None = None,
        min_confidence: float = 0.25,
        max_detections_per_frame: int = 12,
    ) -> None:
        if yolo_detector is None or grounding_dino_detector is None or segmenter is None or vlm is None:
            raise VisionBackendConfigurationError("YOLO, GroundingDINO, SAM and VLM backends are all required")
        self.yolo_detector = yolo_detector
        self.grounding_dino_detector = grounding_dino_detector
        self.segmenter = segmenter
        self.vlm = vlm
        self.prompts = list(prompts or DEFAULT_PROMPTS)
        self.min_confidence = min_confidence
        self.max_detections_per_frame = max(1, max_detections_per_frame)

    def analyze_frame(self, frame_bgr: Any, timestamp: float, scene_id: str | None = None) -> list[VisualEvent]:
        detections = self._collect_detections(frame_bgr)
        detections = self._dedupe_detections(detections)[: self.max_detections_per_frame]
        events: list[VisualEvent] = []
        for detection in detections:
            segmented = self.segmenter.segment(frame_bgr, detection)
            vlm_payload = self.vlm.describe(frame_bgr, segmented, timestamp, scene_id) or {}
            for event_payload in self._event_payloads(vlm_payload):
                if event_payload.get("is_sound_event") is False:
                    continue
                events.append(self._visual_event_from_payload(timestamp, scene_id, segmented, event_payload))
        return events

    def refine_event(self, frame_bgr: Any, event: VisualEvent) -> VisualEvent:
        """Refine an existing event with the production cascade.

        Kept for API compatibility, but it still requires the full production stack
        and never returns the original event as a fallback on backend failure.
        """
        events = self.analyze_frame(frame_bgr, event.start, event.evidence.scene_id)
        if not events:
            raise VisionBackendError("Production vision cascade returned no VLM sound events for the supplied frame")
        best = self._select_event(event, events)
        best.start = event.start
        best.end = event.end
        return best

    def _collect_detections(self, frame_bgr: Any) -> list[ObjectDetection]:
        detections: list[ObjectDetection] = []
        for role, detector in [("yolo", self.yolo_detector), ("groundingdino", self.grounding_dino_detector)]:
            results = detector.detect(frame_bgr, self.prompts)
            for detection in results:
                if detection.confidence < self.min_confidence:
                    continue
                enriched = replace(detection)
                enriched.attributes = dict(detection.attributes)
                enriched.attributes["detector_role"] = role
                if role not in enriched.source.lower():
                    enriched.source = f"{role}:{enriched.source}"
                detections.append(enriched)
        if not detections:
            raise VisionBackendError("YOLO and GroundingDINO returned no detections above the confidence threshold")
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def _dedupe_detections(self, detections: list[ObjectDetection]) -> list[ObjectDetection]:
        kept: list[ObjectDetection] = []
        for detection in detections:
            duplicate = None
            for existing in kept:
                if self._iou(existing.bbox, detection.bbox) >= 0.72:
                    duplicate = existing
                    break
            if duplicate is None:
                kept.append(detection)
                continue
            duplicate.confidence = max(duplicate.confidence, detection.confidence)
            duplicate.source = "+".join(sorted(set(duplicate.source.split("+") + detection.source.split("+"))))
            duplicate.attributes.setdefault("merged_labels", [])
            duplicate.attributes["merged_labels"].append(detection.label)
            duplicate.attributes.update({f"{detection.source}_attributes": detection.attributes})
        return kept

    def _event_payloads(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw_events = payload.get("events")
        if isinstance(raw_events, list):
            return [dict(item) for item in raw_events if isinstance(item, dict)]
        return [payload]

    def _visual_event_from_payload(
        self,
        timestamp: float,
        scene_id: str | None,
        detection: ObjectDetection,
        payload: dict[str, Any],
    ) -> VisualEvent:
        event_type = str(payload.get("event_type") or detection.event_type or "").strip()
        if not event_type:
            raise VisionBackendError("VLM response must include event_type for every accepted sound event")
        object_label = str(payload.get("object_label") or detection.label or "unknown_object")
        start_offset = float(payload.get("start_offset", 0.0) or 0.0)
        duration = float(payload.get("duration", payload.get("event_duration", 0.35)) or 0.35)
        start = max(0.0, timestamp + start_offset)
        end = max(start + 0.05, start + duration)
        confidence = max(0.0, min(1.0, float(payload.get("confidence", detection.confidence))))
        attributes = dict(detection.attributes)
        attributes.update({k: v for k, v in payload.items() if k not in CORE_VLM_FIELDS})
        attributes["detector_label"] = detection.label
        attributes["detector_confidence"] = detection.confidence
        attributes["production_stack"] = "yolo+groundingdino+sam+vlm"
        if "description" in payload:
            attributes["description"] = payload["description"]
        return VisualEvent(
            start=start,
            end=end,
            evidence=VisualEvidence(
                object_label=object_label,
                event_type=event_type,
                bbox=detection.bbox,
                confidence=confidence,
                motion_score=max(0.0, float(payload.get("motion_score", 0.0) or 0.0)),
                contact_score=max(0.0, float(payload.get("contact_score", 0.0) or 0.0)),
                scene_id=str(payload.get("scene_id") or scene_id) if (payload.get("scene_id") or scene_id) else None,
                mask_area=detection.mask_area,
                source=self._source_name(detection.source),
                attributes=attributes,
                notes=["production cascade generated this event; no motion-heuristic fallback used"],
            ),
        )

    def _select_event(self, event: VisualEvent, candidates: list[VisualEvent]) -> VisualEvent:
        event_box = event.evidence.bbox
        if event_box:
            return max(candidates, key=lambda c: (self._iou(event_box, c.evidence.bbox or (0, 0, 0, 0)), c.evidence.confidence))
        return max(candidates, key=lambda c: c.evidence.confidence)

    def _source_name(self, detector_source: str) -> str:
        pieces = detector_source.split("+") + ["sam", "vlm"]
        return "+".join(dict.fromkeys(pieces))

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


class HostedDetectorAdapter:
    """HTTP detector adapter for YOLO or GroundingDINO services.

    Expected response JSON:
    {"detections": [{"label": "door", "bbox": [x,y,w,h], "confidence": 0.81}]}
    `xyxy` is also accepted and normalized to `[x, y, w, h]`.
    """

    def __init__(self, endpoint: str, source: str, timeout: float = 20.0) -> None:
        if not endpoint:
            raise VisionBackendConfigurationError(f"{source} endpoint is required")
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
            bbox = _normalize_bbox(raw)
            detections.append(
                ObjectDetection(
                    label=str(raw.get("label", "unknown_object")),
                    bbox=bbox,
                    confidence=float(raw.get("confidence", 0.5)),
                    source=str(raw.get("source", self.source)),
                    event_type=raw.get("event_type"),
                    attributes=dict(raw.get("attributes", {})),
                )
            )
        return detections


class HostedSegmenterAdapter:
    """HTTP SAM adapter. Response may include `mask_area`, `bbox`, and attributes."""

    def __init__(self, endpoint: str, timeout: float = 25.0) -> None:
        if not endpoint:
            raise VisionBackendConfigurationError("SAM endpoint is required")
        self.endpoint = endpoint
        self.timeout = timeout

    def segment(self, frame_bgr: Any, detection: ObjectDetection) -> ObjectDetection:
        import requests

        payload = {
            "image_b64": _encode_frame_b64(frame_bgr),
            "detection": {
                "label": detection.label,
                "bbox": list(detection.bbox),
                "confidence": detection.confidence,
                "source": detection.source,
                "attributes": detection.attributes,
            },
        }
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        refined = replace(detection)
        refined.attributes = dict(detection.attributes)
        if "bbox" in data or "xyxy" in data:
            refined.bbox = _normalize_bbox(data)
        if "mask_area" in data:
            refined.mask_area = float(data["mask_area"])
        refined.attributes.update(dict(data.get("attributes", {})))
        refined.attributes["sam_applied"] = True
        return refined


class HostedVLMAdapter:
    """HTTP VLM adapter for action/context/sound-event classification."""

    def __init__(self, endpoint: str, timeout: float = 35.0) -> None:
        if not endpoint:
            raise VisionBackendConfigurationError("VLM endpoint is required")
        self.endpoint = endpoint
        self.timeout = timeout

    def describe(self, frame_bgr: Any, detection: ObjectDetection, timestamp: float, scene_id: str | None) -> dict[str, Any]:
        import requests

        payload = {
            "image_b64": _encode_frame_b64(frame_bgr),
            "timestamp": timestamp,
            "scene_id": scene_id,
            "detection": {
                "label": detection.label,
                "bbox": list(detection.bbox),
                "confidence": detection.confidence,
                "source": detection.source,
                "mask_area": detection.mask_area,
                "attributes": detection.attributes,
            },
            "task": "return sound-worthy visual events with object_label, event_type, confidence, duration, description",
        }
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return dict(response.json())


class YOLODetectorAdapter:
    """Local YOLO adapter using ultralytics."""

    def __init__(self, weights: str) -> None:
        if not weights:
            raise VisionBackendConfigurationError("SONICFRAME_YOLO_WEIGHTS is required when no YOLO endpoint is configured")
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
    """Hosted GroundingDINO detector service."""

    def __init__(self, endpoint: str, timeout: float = 20.0) -> None:
        super().__init__(endpoint=endpoint, source="groundingdino", timeout=timeout)


class HostedYOLOAdapter(HostedDetectorAdapter):
    """Hosted YOLO detector service."""

    def __init__(self, endpoint: str, timeout: float = 20.0) -> None:
        super().__init__(endpoint=endpoint, source="yolo", timeout=timeout)


DEFAULT_PROMPTS = [
    "person",
    "foot",
    "door",
    "hand",
    "glass",
    "metal object",
    "wooden object",
    "vehicle",
    "animal",
    "falling object",
    "impact",
    "collision",
]

CORE_VLM_FIELDS = {
    "object_label",
    "event_type",
    "confidence",
    "duration",
    "event_duration",
    "start_offset",
    "scene_id",
    "motion_score",
    "contact_score",
    "is_sound_event",
}


def build_vision_cascade_from_env() -> VisionCascade:
    yolo_endpoint = os.getenv("SONICFRAME_YOLO_ENDPOINT", "").strip()
    yolo_weights = os.getenv("SONICFRAME_YOLO_WEIGHTS", "").strip()
    if yolo_endpoint:
        yolo: ObjectDetector = HostedYOLOAdapter(yolo_endpoint, timeout=env_float("SONICFRAME_YOLO_TIMEOUT", 20.0))
    elif yolo_weights:
        yolo = YOLODetectorAdapter(yolo_weights)
    else:
        raise VisionBackendConfigurationError("Configure either SONICFRAME_YOLO_ENDPOINT or SONICFRAME_YOLO_WEIGHTS")

    grounding_endpoint = os.getenv("SONICFRAME_GROUNDINGDINO_ENDPOINT", "").strip() or os.getenv(
        "SONICFRAME_DETECTOR_ENDPOINT", ""
    ).strip()
    if not grounding_endpoint:
        raise VisionBackendConfigurationError("Configure SONICFRAME_GROUNDINGDINO_ENDPOINT")

    return VisionCascade(
        yolo_detector=yolo,
        grounding_dino_detector=PromptedGroundingDINOAdapter(
            grounding_endpoint, timeout=env_float("SONICFRAME_GROUNDINGDINO_TIMEOUT", 25.0)
        ),
        segmenter=HostedSegmenterAdapter(required_env("SONICFRAME_SAM_ENDPOINT"), timeout=env_float("SONICFRAME_SAM_TIMEOUT", 30.0)),
        vlm=HostedVLMAdapter(required_env("SONICFRAME_VLM_ENDPOINT"), timeout=env_float("SONICFRAME_VLM_TIMEOUT", 40.0)),
        prompts=csv_env("SONICFRAME_VISION_PROMPTS", DEFAULT_PROMPTS),
        min_confidence=env_float("SONICFRAME_MIN_DETECTION_CONFIDENCE", 0.25),
        max_detections_per_frame=env_int("SONICFRAME_MAX_DETECTIONS_PER_FRAME", 12),
    )


def _encode_frame_b64(frame_bgr: Any) -> str:
    if cv2 is None:
        raise VisionBackendError("OpenCV is required to encode frames for production vision backends")
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise VisionBackendError("Failed to encode frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _normalize_bbox(raw: dict[str, Any]) -> tuple[int, int, int, int]:
    if "xyxy" in raw:
        x1, y1, x2, y2 = [int(v) for v in raw["xyxy"][:4]]
        return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
    bbox = raw.get("bbox", [0, 0, 1, 1])
    if len(bbox) < 4:
        raise VisionBackendError(f"Invalid bbox payload: {bbox!r}")
    x, y, w, h = [int(v) for v in bbox[:4]]
    return (x, y, max(1, w), max(1, h))
