# SonicFrame HITL

**Human-in-the-loop Explainable Video-to-Audio Sound Design System**

영상에서 장면, 움직임, 접촉 이벤트를 추출하고 설명 가능한 `SoundTimeline`을 생성한 뒤, 사용자의 편집 로그와 자연어 피드백을 선호 프로필로 변환해 다음 사운드 계획에 반영하는 코드베이스입니다.

## 핵심 기능

- 영상 분석: 장면 분할, motion/contact score, coarse bbox 기반 시각 이벤트 추출
- 설명 가능한 사운드 타임라인: 각 사운드가 왜, 어디에, 어떤 강도로 들어갔는지 설명
- Human-in-the-loop: 삭제, 볼륨 조정, 타이밍 조정, 스타일 변경, 후보 선택, 자연어 피드백 기록
- 선호 프로필 학습: 이벤트별 강도, 객체별 소리 프로필, 밀도, 스타일 선호 반영
- 재계획 에이전트: `VideoAsset + SceneSegment + VisualEvent + UserPreferenceProfile` 기반 SoundTimeline 재생성
- 후보 비교 UI: realistic / cinematic / restrained 후보 미리듣기 및 선택 로그 저장
- 오디오 렌더링: 외부 API 없이 procedural WAV mix 생성
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
    video_analysis.py    # scene/event extraction
    planner.py           # explainable sound timeline planner
    feedback.py          # edit logs -> user preference profile
    audio.py             # procedural sound synthesis + WAV render
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


## 구현 한계와 교체 지점

- 객체 인식은 lightweight motion heuristic입니다. 실제 제출 환경에서 YOLO, GroundingDINO, SAM, VLM을 연결하면 객체 라벨 품질을 높일 수 있습니다.
- 오디오는 procedural preview입니다. 상용 수준 Foley는 diffusion/audio generation backend로 `ProceduralAudioEngine`만 교체하면 됩니다.
- 피드백 학습은 규칙 기반입니다. 충분한 로그가 쌓이면 pairwise preference model 또는 contextual bandit으로 확장할 수 있습니다.
