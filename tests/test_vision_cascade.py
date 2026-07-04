import numpy as np
import pytest

from sonicframe_hitl.vision_backends import ObjectDetection, VisionBackendError, VisionCascade


class FakeYOLO:
    def detect(self, frame_bgr, prompts):
        return [ObjectDetection(label="door", bbox=(10, 20, 30, 40), confidence=0.9, source="yolo")]


class FakeGroundingDINO:
    def detect(self, frame_bgr, prompts):
        return [ObjectDetection(label="wooden door", bbox=(11, 21, 31, 41), confidence=0.86, source="groundingdino")]


class EmptyDetector:
    def detect(self, frame_bgr, prompts):
        return []


class FakeSegmenter:
    def segment(self, frame_bgr, detection):
        detection.mask_area = 1200.0
        detection.attributes["material"] = "wood"
        return detection


class FakeVLM:
    def describe(self, frame_bgr, detection, timestamp, scene_id):
        return {
            "object_label": "wooden_door",
            "event_type": "contact",
            "confidence": 0.92,
            "duration": 0.4,
            "contact_score": 0.8,
            "description": "door closes",
        }


def test_vision_cascade_generates_events_from_all_required_backends():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    cascade = VisionCascade(FakeYOLO(), FakeGroundingDINO(), FakeSegmenter(), FakeVLM())
    events = cascade.analyze_frame(frame, timestamp=1.0, scene_id="scene_1")
    assert len(events) == 1
    event = events[0]
    assert event.evidence.object_label == "wooden_door"
    assert event.evidence.event_type == "contact"
    assert event.evidence.mask_area == 1200.0
    assert "yolo" in event.evidence.source
    assert "groundingdino" in event.evidence.source
    assert "sam" in event.evidence.source
    assert "vlm" in event.evidence.source
    assert event.evidence.attributes["production_stack"] == "yolo+groundingdino+sam+vlm"


def test_vision_cascade_fails_instead_of_falling_back_when_detectors_empty():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    cascade = VisionCascade(EmptyDetector(), EmptyDetector(), FakeSegmenter(), FakeVLM())
    with pytest.raises(VisionBackendError):
        cascade.analyze_frame(frame, timestamp=1.0, scene_id="scene_1")
