FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SONICFRAME_WORKSPACE=/app/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY sonicframe_hitl ./sonicframe_hitl
COPY examples ./examples
COPY docs ./docs
RUN pip install --no-cache-dir -r requirements.txt && pip install -e .
RUN mkdir -p /app/workspace/uploads /app/workspace/exports /app/workspace/projects

EXPOSE 8000 8501
CMD ["bash", "-lc", "uvicorn sonicframe_hitl.api.main:app --host 0.0.0.0 --port 8000 & streamlit run sonicframe_hitl/ui/app.py --server.port 8501 --server.address 0.0.0.0"]
