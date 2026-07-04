# Architecture

## Pipeline

```text
Video Upload
  ↓
VideoAnalyzer
  - metadata
  - scene split by histogram difference
  - motion/contact events by frame difference
  - optional VisionCascade refinement: YOLO/GroundingDINO → SAM → VLM
  ↓
SoundPlanner
  - visual event → sound event mapping
  - preference profile scaling
  - explanation generation
  ↓
Audio backend
  - ProceduralAudioEngine fallback
  - FoleyAssetEngine asset retrieval
  - HostedGenerativeAudioEngine diffusion/audio-generation adapter
  - stereo pan and limiter
  ↓
FeedbackInterpreter
  - edit logs and natural language feedback
  - profile update
  ↓
Replanning / Candidate Comparison / Export
```

## Explainability design

Every `SoundEvent` contains three reasons:

1. `visual_reason`: detected object/event, motion score, contact score, bbox.
2. `feedback_reason`: prior edits such as lower collision intensity, shorter footstep, sparse density.
3. `planning_reason`: style and scene-level choice such as restrained/cinematic and duration/intensity.

This supports user trust: the user can see both the video evidence and the learned preference behind the decision.

## Human-in-the-loop learning

The system stores each user action as a structured `FeedbackLog`.

- Delete sound → reduce density and lower that event type intensity.
- Adjust volume → update event/object multiplier.
- Adjust time shorter → mark object/sound type as `prefer_short`.
- Change style → update default style.
- Choose candidate → increment variant preference and record contextual reward stats.
- Natural language feedback → keyword parser for Korean/English intent.

The result is a `UserPreferenceProfile` that is immediately applied during replanning.

## Performance choices

The project is designed to run on a laptop during a hackathon demo.

- Video is sampled at `SONICFRAME_SAMPLE_FPS`, default 6 fps.
- Audio rendering is vectorized with NumPy.
- Project state is JSON-backed, no database required.
- UI and API can run independently.

## Extension points

- Configure `VisionCascade` with YOLO/GroundingDINO detector, SAM segmenter and a VLM instead of replacing the whole `VideoAnalyzer`.
- Configure `build_audio_engine_from_env()` for Foley retrieval or hosted diffusion/audio generation while preserving `ProceduralAudioEngine` as fallback.
- Replace the lightweight contextual reward stats in `FeedbackInterpreter` with a pairwise preference model/reward model when enough logs are available.
- Add DAW export such as EDL, AAF, or Reaper project files.
