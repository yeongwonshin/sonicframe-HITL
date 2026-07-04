.PHONY: install install-vision test api ui clean

install:
	pip install -r requirements.txt
	pip install -e .

install-vision:
	pip install -e '.[vision]'

test:
	pytest -q

api:
	uvicorn sonicframe_hitl.api.main:app --reload

ui:
	streamlit run sonicframe_hitl/ui/app.py

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ sonicframe_hitl.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
