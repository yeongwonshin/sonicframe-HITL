#!/usr/bin/env bash
set -euo pipefail
python scripts/create_demo_video.py
sonicframe analyze workspace/uploads/demo_motion.mp4 --style balanced
streamlit run sonicframe_hitl/ui/app.py
