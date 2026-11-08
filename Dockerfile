# API and worker images deliberately contain no model weights or subtitle data.
FROM python:3.11-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml alembic.ini ./
COPY api ./api
COPY ml ./ml
RUN pip install --no-cache-dir '.[api]' \
    && groupadd --gid 10001 sorani \
    && useradd --uid 10001 --gid 10001 --create-home sorani \
    && mkdir -p /app/data/private-storage /app/data/cache/huggingface /app/outputs/models \
    && chmod 700 /app/data/private-storage \
    && chown -R 10001:10001 /app
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

FROM api AS worker
USER root
# The GPU overlay switches this to the official CUDA 12.4 wheel index.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" 'torch==2.6.0' \
    && pip install --no-cache-dir '.[api,train]'
USER 10001:10001
CMD ["celery", "-A", "api.worker:celery_app", "worker", "--loglevel=INFO", "--pool=solo", "--concurrency=1"]
