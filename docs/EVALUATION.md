# Evaluation

## Metrics

| Area | Metric | Notes |
| --- | --- | --- |
| Vision | Detector coverage | YOLO and GroundingDINO detections per sampled frame |
| Vision | Segmentation evidence | SAM mask area/bbox refinement availability |
| Vision | VLM event precision | Accepted sound-worthy events vs rejected detections |
| Audio | Foley hit rate | Ratio of events resolved by curated assets |
| Audio | Generative repair rate | Hybrid events generated because assets were missing |
| HITL | Preference adaptation | Candidate ranking changes after user choices/deletions/volume edits |
| Export | Reproducibility | Project JSON includes backend source and evidence metadata |

## Success criteria

- Every `VisualEvent.evidence.source` contains YOLO/GroundingDINO, SAM, and VLM provenance.
- No event is created from a synthetic or motion-only fallback.
- Rendering uses `foley`, `generative`, or `hybrid`; procedural audio is unavailable.
- Backend errors are visible and actionable.
