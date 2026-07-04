import numpy as np

from sonicframe_hitl.models import VisualEvidence, VisualEvent
from sonicframe_hitl.vision_backends import ObjectDetection, VisionCascade


class FakeDetector:
    def detect(self, frame_bgr, prompts):
        return [ObjectDetection(label="door", bbox=(10, 20, 30, 40), confidence=0.9, source="fake_dino")]


class FakeSegmenter:
    def segment(self, frame_bgr, detection):
        detection.mask_area = 1200.0
        return detection


class FakeVLM:
    def describe(self, frame_bgr, detection, event):
        return {"object_label": "wooden_door", "event_type": "contact", "description": "door closes"}


def test_vision_cascade_refines_motion_event():
    event = VisualEvent(
        start=1.0,
        end=1.2,
        evidence=VisualEvidence(object_label="moving_object", event_type="motion", bbox=(8, 18, 32, 42)),
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    refined = VisionCascade(FakeDetector(), FakeSegmenter(), FakeVLM()).refine_event(frame, event)
    assert refined.evidence.object_label == "wooden_door"
    assert refined.evidence.event_type == "contact"
    assert refined.evidence.mask_area == 1200.0
    assert "vlm" in refined.evidence.source
