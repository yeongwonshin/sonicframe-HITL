# SonicFrame HITL

**Production human-in-the-loop video-to-audio sound design system**

SonicFrame HITL is a backend-integrated video-to-audio sound design system that analyzes visual events from video and generates synchronized sound timelines through a human-in-the-loop workflow. Visual events are produced through a mandatory **YOLO + GroundingDINO + SAM + VLM** cascade, and audio is rendered exclusively through **Foley assets**, a **hosted generative audio backend**, or a hybrid combination of both.

The system is designed for workflows where sound decisions must be explainable, editable, and reproducible. Users can review detected sound-worthy events, provide natural-language feedback, select candidate sound plans, refine preferences, regenerate timelines, and export project outputs.

## Key Features

- **Multi-stage vision analysis**: YOLO and GroundingDINO perform object detection, SAM adds segmentation evidence, and a VLM confirms sound-worthy events, actions, and context.
- **Strict backend dependency model**: Missing detector responses, VLM failures, SAM failures, OpenCV errors, unset backend configuration, or missing Foley assets are surfaced as explicit errors instead of being silently replaced.
- **Production audio rendering pipeline**: Audio rendering supports only `foley`, `generative`, and `hybrid` backends.
- **Human-in-the-loop workflow**: Natural-language feedback, candidate selection, preference profiling, replanning, rendering, and export flows are preserved throughout the project lifecycle.
- **Explainable sound planning**: The planner creates editable sound timelines based on detected events, context, confidence, style, and user feedback.

## Required Environment Variables

```bash
export SONICFRAME_WORKSPACE=workspace
export SONICFRAME_SAMPLE_FPS=6
export SONICFRAME_DEFAULT_STYLE=balanced

# YOLO: either a hosted endpoint or local Ultralytics weights is required
export SONICFRAME_YOLO_ENDPOINT=http://localhost:9000/detect
# Or: export SONICFRAME_YOLO_WEIGHTS=/models/yolo/best.pt

# GroundingDINO / SAM / VLM are required
export SONICFRAME_GROUNDINGDINO_ENDPOINT=http://localhost:9001/detect
export SONICFRAME_SAM_ENDPOINT=http://localhost:9002/segment
export SONICFRAME_VLM_ENDPOINT=http://localhost:9003/describe

export SONICFRAME_VISION_PROMPTS=person,foot,door,hand,glass,metal object,wooden object,vehicle,animal,falling object,impact,collision
export SONICFRAME_MIN_DETECTION_CONFIDENCE=0.25
export SONICFRAME_MAX_DETECTIONS_PER_FRAME=12
```

Audio rendering must be configured with one of the following backend modes.

### Foley-only Backend

```bash
export SONICFRAME_AUDIO_BACKEND=foley
export SONICFRAME_FOLEY_DIR=/data/foley
```

In `foley` mode, rendering fails when a required Foley asset is missing.

### Generative Audio Backend

```bash
export SONICFRAME_AUDIO_BACKEND=generative
export SONICFRAME_AUDIO_ENDPOINT=http://localhost:9010/generate
```

### Hybrid Backend

```bash
export SONICFRAME_AUDIO_BACKEND=hybrid
export SONICFRAME_FOLEY_DIR=/data/foley
export SONICFRAME_AUDIO_ENDPOINT=http://localhost:9010/generate
```

In `hybrid` mode, SonicFrame attempts Foley matching first and falls back to the hosted generative audio backend when no matching Foley asset exists.

## Backend HTTP Contracts

### YOLO / GroundingDINO Detector

Request:

```json
{
  "image_b64": "...",
  "prompts": ["person", "door", "impact"]
}
```

Response:

```json
{
  "detections": [
    {
      "label": "door",
      "bbox": [120, 40, 180, 320],
      "confidence": 0.88,
      "attributes": {
        "material": "wood"
      }
    }
  ]
}
```

`xyxy: [x1, y1, x2, y2]` is also accepted and is normalized internally to `[x, y, w, h]`.

### SAM Segmenter

The request includes `image_b64` and the detection payload.

Response:

```json
{
  "bbox": [118, 39, 184, 322],
  "mask_area": 52320,
  "attributes": {
    "mask_confidence": 0.91
  }
}
```

### VLM

The request includes the image, timestamp, scene ID, and SAM-enhanced detection payload.

The response may return either a single event or an `events` array.

```json
{
  "events": [
    {
      "object_label": "wooden_door",
      "event_type": "contact",
      "confidence": 0.86,
      "duration": 0.42,
      "motion_score": 0.2,
      "contact_score": 0.8,
      "description": "door closes against the frame"
    }
  ]
}
```

An accepted event without `event_type` is treated as an error. If `is_sound_event: false` is returned, the detection is excluded from the sound timeline.

### Generative Audio Backend

Request:

```json
{
  "prompt": "realistic production Foley for wooden_door contact...",
  "duration": 0.42,
  "sample_rate": 44100,
  "event": {
    "...": "SoundEvent JSON"
  },
  "constraints": {
    "no_music": true,
    "no_speech": true,
    "sync_point_seconds": 1.25
  }
}
```

The response may contain raw WAV bytes, `wav_b64`, or `audio_b64`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

For local YOLO weights:

```bash
pip install -e '.[vision]'
```

## Usage

### CLI

```bash
sonicframe analyze path/to/video.mp4 --style balanced
sonicframe feedback <project_id> "Make this scene quieter and the footsteps heavier."
sonicframe render <project_id>
sonicframe export <project_id>
```

### API Server

```bash
uvicorn sonicframe_hitl.api.main:app --reload
```

### Streamlit Workbench

```bash
streamlit run sonicframe_hitl/ui/app.py
```

## Foley Asset Structure

```text
/data/foley/
  contact/
    wooden_door/*.wav
    door/*.wav
  footstep/
    person/*.wav
  motion/*.wav
```

Foley lookup order:

1. `sound_type/object_label`
2. `sound_type`
3. `object_label`
4. `metadata.foley_tags`

When no matching WAV asset is found, `SONICFRAME_AUDIO_BACKEND=foley` fails immediately. In `hybrid` mode, the system routes the request to the generative audio backend.

## Project Structure

```text
sonicframe_hitl/
  sonicframe_hitl/
    api/main.py          # FastAPI endpoints
    ui/app.py            # Streamlit HITL workbench
    models.py            # VideoAsset, VisualEvent, SoundTimeline, FeedbackLog
    video_analysis.py    # production frame sampling + mandatory vision cascade events
    vision_backends.py   # YOLO/GroundingDINO/SAM/VLM adapters and contracts
    planner.py           # explainable sound timeline planner
    feedback.py          # edit logs -> rules + contextual preference stats
    audio.py             # strict Foley/generative/hybrid audio backends
    exporters.py         # JSON/CSV/submission bundle
    storage.py           # JSON-backed local project store
    cli.py               # command line interface
  docs/
  tests/
  workspace/
```

## System Flow

```text
Video Upload
  -> Frame Sampling
  -> YOLO + GroundingDINO Detection
  -> SAM Segmentation Evidence
  -> VLM Event Confirmation
  -> Explainable Sound Timeline Planning
  -> HITL Feedback and Candidate Selection
  -> Foley / Generative / Hybrid Audio Rendering
  -> Export Bundle
```

## Export Outputs

SonicFrame can export project artifacts for review, integration, or downstream production workflows, including:

- Sound timeline JSON
- Event-level CSV summaries
- Feedback logs
- Rendered audio outputs
- Submission bundle metadata

## Technology Stack

- **Language**: Python
- **API**: FastAPI, Uvicorn
- **UI**: Streamlit
- **Vision**: YOLO, GroundingDINO, SAM, VLM backend adapters
- **Audio**: Foley asset retrieval, hosted generative audio backend integration
- **Data / Storage**: JSON-backed local project store
- **CLI**: Python package entry points
- **Workflow**: Human-in-the-loop feedback, preference profiling, timeline replanning, export pipeline
