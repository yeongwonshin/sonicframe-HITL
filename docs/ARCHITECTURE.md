# Architecture

SonicFrame HITL now runs as a production-backend pipeline. It does not create synthetic visual events, does not label events with motion heuristics, and does not render procedural audio.

## Pipeline

```text
Video upload
  ↓
OpenCV frame sampling + scene grouping
  ↓
YOLO detector + GroundingDINO detector
  ↓
Detection deduplication
  ↓
SAM segmentation refinement
  ↓
VLM action/context/sound-event decision
  ↓
VisualEvent evidence
  ↓
SoundPlanner + UserPreferenceProfile
  ↓
Foley asset backend or hosted generative audio backend
  ↓
WAV mix + JSON/CSV/profile export
```

## Failure policy

All production backends are mandatory. Missing configuration, empty detections, SAM/VLM errors, and audio asset/backend failures raise errors. The user must fix model configuration, prompts, thresholds, Foley routing, or backend availability rather than relying on demo substitutions.

## Vision contracts

- YOLO: local `ultralytics` weights or hosted endpoint.
- GroundingDINO: hosted prompted detector endpoint.
- SAM: hosted segmentation endpoint that can refine bbox and provide mask area.
- VLM: hosted multimodal endpoint that decides whether a detection is sound-worthy and returns `object_label`, `event_type`, `confidence`, `duration`, and optional scores/description.

## Audio contracts

- `foley`: strict asset retrieval from `SONICFRAME_FOLEY_DIR`; missing asset is an error.
- `generative`: hosted backend returning WAV bytes or base64 WAV.
- `hybrid`: Foley first, generative only when an asset is missing; still no procedural fallback.

## HITL loop

Feedback logs update `UserPreferenceProfile`. Replanning uses the original VLM-backed `VisualEvent` evidence plus learned profile multipliers and contextual candidate scores.
