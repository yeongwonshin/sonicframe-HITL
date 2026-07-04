# SonicFrame HITL

**Human-in-the-loop Explainable Video-to-Audio Sound Design System**

영상에서 장면, 움직임, 접촉 이벤트를 추출하고 설명 가능한 `SoundTimeline`을 생성한 뒤, 사용자의 편집 로그와 자연어 피드백을 선호 프로필로 변환해 다음 사운드 계획에 반영하는 코드베이스입니다.

## 핵심 기능

- 영상 분석: 장면 분할, motion/contact score, coarse bbox 기반 시각 이벤트 추출 + 선택적 YOLO/GroundingDINO/SAM/VLM cascade
- 설명 가능한 사운드 타임라인: 각 사운드가 왜, 어디에, 어떤 강도로 들어갔는지 설명
- Human-in-the-loop: 삭제, 볼륨 조정, 타이밍 조정, 스타일 변경, 후보 선택, 자연어 피드백 기록
- 선호 프로필 학습: 이벤트별 강도, 객체별 소리 프로필, 밀도, 스타일 선호 + contextual preference ranking 반영
- 재계획 에이전트: `VideoAsset + SceneSegment + VisualEvent + UserPreferenceProfile` 기반 SoundTimeline 재생성
- 후보 비교 UI: realistic / cinematic / restrained 후보 미리듣기 및 선택 로그 저장
- 오디오 렌더링: procedural fallback, Foley asset retrieval, hosted generative audio backend 교체 지원
- 제출 번들: 프로젝트 JSON, Timeline CSV/JSON, Profile JSON, submission note export

## 빠른 실행

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Streamlit 데모 UI

```bash
streamlit run sonicframe_hitl/ui/app.py
```

브라우저에서 영상을 업로드하면 자동으로 분석, 타임라인 생성, WAV 렌더링까지 실행됩니다.

### FastAPI 서버

```bash
uvicorn sonicframe_hitl.api.main:app --reload
```

- API docs: `http://127.0.0.1:8000/docs`
- Health check: `GET /health`
- Analyze: `POST /analyze`
- Feedback: `POST /projects/{project_id}/feedback`
- Render: `POST /projects/{project_id}/render`
- Export: `POST /projects/{project_id}/export`

### CLI

```bash
sonicframe analyze examples/demo_motion.mp4 --style balanced
# 또는 본인 영상으로: sonicframe analyze path/to/video.mp4 --style balanced
sonicframe feedback <project_id> "이 장면은 더 조용하게, 발소리는 더 무겁게"
sonicframe render <project_id>
sonicframe export <project_id>
```

### Docker

```bash
docker compose up --build
```

- Streamlit: `http://localhost:8501`
- API: `http://localhost:8000`

## 디렉토리 구조

```text
sonicframe_hitl/
  sonicframe_hitl/
    api/main.py          # FastAPI endpoints
    ui/app.py            # Streamlit browser UI
    models.py            # VideoAsset, VisualEvent, SoundTimeline, FeedbackLog
    video_analysis.py    # scene/event extraction + optional vision cascade refinement
    vision_backends.py   # YOLO/GroundingDINO/SAM/VLM adapter interfaces
    planner.py           # explainable sound timeline planner
    feedback.py          # edit logs -> rules + contextual preference stats
    audio.py             # procedural/Foley/hosted audio rendering backends
    exporters.py         # JSON/CSV/submission bundle
    storage.py           # JSON-backed local project store
    cli.py               # command line interface
  examples/              # sample profile/timeline/feedback data
  docs/                  # architecture and evaluation notes
  tests/                 # unit tests
  scripts/               # demo video generator
  workspace/             # uploads, exports, project JSON
```


## 데이터 모델 요약

- `VideoAsset`: 파일 경로, FPS, 해상도, duration
- `SceneSegment`: start/end, motion mean, style hint
- `VisualEvent`: start/end, object label, event type, motion/contact score, bbox
- `SoundEvent`: start/end, sound type, volume, intensity, pan, explanation
- `FeedbackLog`: 사용자의 삭제/조정/자연어/후보 선택 로그
- `UserPreferenceProfile`: 이벤트별 강도, 객체별 preference, density, style preference
- `ProjectState`: 하나의 제출 단위 전체 상태


## 개선된 확장 구조

### 1) Vision cascade

기본 실행은 여전히 lightweight motion heuristic으로 동작합니다. 다만 `sonicframe_hitl/vision_backends.py`가 추가되어 실제 제출 환경에서는 다음처럼 cascade를 연결할 수 있습니다.

```text
motion/contact keyframe 후보
  ↓
YOLO 또는 GroundingDINO detector
  ↓
SAM segmenter
  ↓
VLM action/context refiner
  ↓
VisualEvent.evidence(object_label, event_type, bbox, mask_area, source, attributes)
```

환경 변수 예시:

```bash
export SONICFRAME_VISION_BACKEND=groundingdino
export SONICFRAME_DETECTOR_ENDPOINT=http://localhost:9001/detect
export SONICFRAME_SAM_ENDPOINT=http://localhost:9002/segment
export SONICFRAME_VLM_ENDPOINT=http://localhost:9003/describe
```

로컬 YOLO를 쓰는 환경에서는 `SONICFRAME_VISION_BACKEND=yolo`, `SONICFRAME_YOLO_WEIGHTS=yolov8n.pt`를 설정하면 됩니다. 외부 모델이 없어도 fallback으로 기존 분석이 유지됩니다.

### 2) Audio backend

`audio.py`는 이제 세 가지 backend를 같은 인터페이스로 제공합니다.

- `ProceduralAudioEngine`: 외부 의존성 없는 기본 fallback
- `FoleyAssetEngine`: `SONICFRAME_FOLEY_DIR` 아래의 WAV Foley asset을 검색해 이벤트에 매칭
- `HostedGenerativeAudioEngine`: AudioLDM/Stable Audio/자체 diffusion backend 같은 HTTP 생성 서버와 연결

환경 변수 예시:

```bash
export SONICFRAME_AUDIO_BACKEND=foley
export SONICFRAME_FOLEY_DIR=assets/foley
# 또는
export SONICFRAME_AUDIO_BACKEND=generative
export SONICFRAME_AUDIO_ENDPOINT=http://localhost:9010/generate
```

### 3) Feedback learning

`FeedbackInterpreter`는 기존 규칙 기반 업데이트에 더해 `UserPreferenceProfile.preference_stats`에 후보 선택/삭제/볼륨 조정 보상을 누적합니다. `SoundPlanner.make_candidates()`는 이 contextual preference score로 후보를 정렬합니다. 로그가 적을 때도 exploration bonus를 둬서 특정 후보로 너무 빨리 고정되지 않게 했습니다.
