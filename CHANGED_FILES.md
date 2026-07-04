# Production backend patch

This patch removes demo fallback behavior and requires real vision/audio backends.
Unzip the changed-file archive over the original `sonicframe_hitl/` project root.
Apply deletions listed in `DELETED_FILES.txt` if your unzip tool does not remove files automatically.

## Main changes

- Replaced optional vision refinement with mandatory YOLO + GroundingDINO + SAM + VLM cascade.
- Removed synthetic visual fallback and motion-heuristic event labeling from `VideoAnalyzer`.
- Removed procedural audio backend from runtime selection.
- Added strict `foley`, `generative`, and `hybrid` audio backends.
- Updated CLI/API/UI/docs/env examples to require production backend configuration.
- Updated tests to validate strict backend behavior and Foley rendering.

## Validation

`pytest -q` passes in the modified project.
