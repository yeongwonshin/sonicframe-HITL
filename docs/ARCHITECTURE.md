# Architecture

## Pipeline

```text
Video Upload
  ↓
VideoAnalyzer
  - metadata
  - scene split by histogram difference
  - motion/contact events by frame difference
  ↓
SoundPlanner
  - visual event → sound event mapping
  - preference profile scaling
  - explanation generation
  ↓
ProceduralAudioEngine
  - footstep/contact/motion/ambient synthesis
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
- Choose candidate → increment variant preference.
- Natural language feedback → keyword parser for Korean/English intent.

The result is a `UserPreferenceProfile` that is immediately applied during replanning.

## Performance choices

The project is designed to run on a laptop during a hackathon demo.

- Video is sampled at `SONICFRAME_SAMPLE_FPS`, default 6 fps.
- Audio rendering is vectorized with NumPy.
- Project state is JSON-backed, no database required.
- UI and API can run independently.

## Extension points

- Replace `VideoAnalyzer` with YOLO/SAM/GroundingDINO or a video-language model.
- Replace `ProceduralAudioEngine` with AudioLDM, Stable Audio, ElevenLabs SFX, or a Foley retrieval database.
- Replace `FeedbackInterpreter` with a learned reward model.
- Add DAW export such as EDL, AAF, or Reaper project files.
