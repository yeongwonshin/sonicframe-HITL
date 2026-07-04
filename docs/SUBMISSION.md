# Hackathon Submission Notes

## Project title

사용자 피드백을 반영하는 설명 가능한 인터랙티브 영상-사운드 생성 시스템

English: Human-in-the-loop Explainable Video-to-Audio Sound Design System

## Problem

Most video-to-audio systems focus on generating a single audio result. Real sound design is iterative: designers delete sounds, adjust timing, reduce loud impacts, and compare alternatives. Those edits are valuable preference signals, but typical systems discard them.

## Solution

SonicFrame HITL converts visual events into explainable sound events and continuously updates a user preference profile from edit logs and natural-language feedback. The system replans the sound timeline using both visual evidence and learned user preference.

## Technical novelty

- Sound events include visual evidence and feedback rationale.
- Edit logs are interpreted into reusable preference rules.
- Candidate comparison produces implicit preference data.
- The workflow is end-to-end and usable in a browser.

## What is included

- FastAPI backend
- Streamlit browser UI
- CLI
- Video analysis module
- Sound timeline planner
- Feedback interpreter
- Procedural audio renderer
- Exporters and tests

## Demo script

1. Launch Streamlit.
2. Upload a short video.
3. Show generated timeline and explanations.
4. Type: `이 장면은 더 조용하게, 충돌음은 과하다`.
5. Show replanned timeline with lower contact intensity.
6. Generate candidates and choose restrained or cinematic.
7. Export bundle.
