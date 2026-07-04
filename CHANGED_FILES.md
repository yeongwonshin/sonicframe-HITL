# SonicFrame HITL improvement patch

This archive contains only files that were changed or added.
Unzip it over the original `sonicframe_hitl/` project root to apply the patch.

## Main changes

- Added optional `VisionCascade` adapters for YOLO, GroundingDINO-style hosted detectors, SAM-style segmenters and VLM refinement.
- Upgraded `VideoAnalyzer` to keep the local motion heuristic fallback while refining top events through the optional cascade.
- Added `FoleyAssetEngine`, `HostedGenerativeAudioEngine` and `build_audio_engine_from_env()` so procedural audio is no longer the only backend path.
- Added contextual preference statistics to `UserPreferenceProfile` and ranked candidate sounds using feedback-derived preference scores.
- Updated README, architecture notes, submission notes, env examples and tests.

## Validation

`pytest -q` passed with 8 tests.
