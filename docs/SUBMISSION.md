# Submission Notes

## What is included

- Mandatory YOLO + GroundingDINO + SAM + VLM visual event cascade
- Strict Foley / hosted generative / hybrid audio rendering backend
- Explainable `SoundTimeline` planner
- Natural-language and edit-log feedback interpreter
- Candidate generation and contextual preference ranking
- JSON/CSV/profile/export bundle generation
- FastAPI, Streamlit, and CLI entry points

## What is intentionally removed

- Synthetic visual fallback events
- Motion-heuristic object/event labeling
- Procedural audio renderer
- Silent fallback when external backends fail

## Production run checklist

1. Start YOLO, GroundingDINO, SAM, VLM, and audio services.
2. Export all required environment variables from `.env.example`.
3. Run `sonicframe analyze path/to/video.mp4 --style balanced`.
4. Review explanations and feedback logs.
5. Render previews/mix with Foley or generative backend.
6. Export the project bundle.
