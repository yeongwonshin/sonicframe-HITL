# SonicFrame HITL

**Production human-in-the-loop video-to-audio sound design system**

이 버전은 데모용 motion heuristic / synthetic fallback / procedural fallback을 제거하고, 실제 백엔드 연결을 전제로 동작합니다. 영상 이벤트는 반드시 **YOLO + GroundingDINO + SAM + VLM** cascade에서 생성되고, 오디오는 반드시 **Foley asset** 또는 **hosted generative audio backend**에서 렌더링됩니다.

## 핵심 변경 사항

- 영상 분석: YOLO와 GroundingDINO가 모두 detection을 수행하고, SAM이 segmentation evidence를 보강한 뒤, VLM이 sound-worthy event/action/context를 확정합니다.
- fallback 제거: OpenCV 실패, 백엔드 미설정, detector/VLM/SAM 응답 실패, Foley asset 누락은 조용히 대체하지 않고 오류로 처리합니다.
- 오디오 렌더링: procedural renderer를 제거하고 `foley`, `generative`, `hybrid` backend만 허용합니다.
- HITL 유지: 자연어 피드백, 후보 선택, 사용자 선호 프로필, 재계획, export 흐름은 유지합니다.

## 필수 환경 변수

```bash
export SONICFRAME_WORKSPACE=workspace
export SONICFRAME_SAMPLE_FPS=6
export SONICFRAME_DEFAULT_STYLE=balanced

# YOLO: hosted endpoint 또는 local ultralytics weights 중 하나 필수
export SONICFRAME_YOLO_ENDPOINT=http://localhost:9000/detect
# 또는: export SONICFRAME_YOLO_WEIGHTS=/models/yolo/best.pt

# GroundingDINO / SAM / VLM은 필수
export SONICFRAME_GROUNDINGDINO_ENDPOINT=http://localhost:9001/detect
export SONICFRAME_SAM_ENDPOINT=http://localhost:9002/segment
export SONICFRAME_VLM_ENDPOINT=http://localhost:9003/describe

export SONICFRAME_VISION_PROMPTS=person,foot,door,hand,glass,metal object,wooden object,vehicle,animal,falling object,impact,collision
export SONICFRAME_MIN_DETECTION_CONFIDENCE=0.25
export SONICFRAME_MAX_DETECTIONS_PER_FRAME=12
```

오디오는 아래 중 하나를 설정합니다.

```bash
# Foley asset만 사용: asset 누락 시 실패
export SONICFRAME_AUDIO_BACKEND=foley
export SONICFRAME_FOLEY_DIR=/data/foley

# 생성형 오디오만 사용
export SONICFRAME_AUDIO_BACKEND=generative
export SONICFRAME_AUDIO_ENDPOINT=http://localhost:9010/generate

# Foley 우선, asset 누락 시 generative backend로 생성
export SONICFRAME_AUDIO_BACKEND=hybrid
export SONICFRAME_FOLEY_DIR=/data/foley
export SONICFRAME_AUDIO_ENDPOINT=http://localhost:9010/generate
```

## 백엔드 HTTP 계약

### YOLO / GroundingDINO detector

요청:

```json
{"image_b64": "...", "prompts": ["person", "door", "impact"]}
```

응답:

```json
{
  "detections": [
    {"label": "door", "bbox": [120, 40, 180, 320], "confidence": 0.88, "attributes": {"material": "wood"}}
  ]
}
```

`xyxy: [x1, y1, x2, y2]`도 허용되며 내부에서 `[x, y, w, h]`로 정규화됩니다.

### SAM segmenter

요청에는 `image_b64`와 detection payload가 포함됩니다.

응답:

```json
{"bbox": [118, 39, 184, 322], "mask_area": 52320, "attributes": {"mask_confidence": 0.91}}
```

### VLM

요청에는 image, timestamp, scene_id, SAM이 보강한 detection이 포함됩니다.

응답은 단일 이벤트 또는 `events` 배열을 반환할 수 있습니다.

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

`event_type`이 없는 accepted event는 오류입니다. `is_sound_event: false`를 반환하면 해당 detection은 sound timeline에서 제외됩니다.

### Generative audio

요청:

```json
{
  "prompt": "realistic production Foley for wooden_door contact...",
  "duration": 0.42,
  "sample_rate": 44100,
  "event": {"...": "SoundEvent JSON"},
  "constraints": {"no_music": true, "no_speech": true, "sync_point_seconds": 1.25}
}
```

응답은 raw WAV bytes, `wav_b64`, 또는 `audio_b64`를 허용합니다.

## 설치 및 실행

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

로컬 YOLO weights를 사용할 경우:

```bash
pip install -e '.[vision]'
```

CLI:

```bash
sonicframe analyze path/to/video.mp4 --style balanced
sonicframe feedback <project_id> "이 장면은 더 조용하게, 발소리는 더 무겁게"
sonicframe render <project_id>
sonicframe export <project_id>
```

API:

```bash
uvicorn sonicframe_hitl.api.main:app --reload
```

Streamlit:

```bash
streamlit run sonicframe_hitl/ui/app.py
```

## Foley asset 구조

```text
/data/foley/
  contact/
    wooden_door/*.wav
    door/*.wav
  footstep/
    person/*.wav
  motion/*.wav
```

검색 순서는 `sound_type/object_label`, `sound_type`, `object_label`, `metadata.foley_tags`입니다. 매칭되는 WAV가 없으면 `SONICFRAME_AUDIO_BACKEND=foley`에서는 실패하고, `hybrid`에서는 generative backend로 넘어갑니다.

## 디렉토리 구조

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
