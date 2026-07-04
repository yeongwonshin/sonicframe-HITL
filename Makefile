.PHONY: install test api ui demo clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

api:
	uvicorn sonicframe_hitl.api.main:app --reload

ui:
	streamlit run sonicframe_hitl/ui/app.py

demo:
	python scripts/create_demo_video.py
	sonicframe analyze workspace/uploads/demo_motion.mp4 --style balanced

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ sonicframe_hitl.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
