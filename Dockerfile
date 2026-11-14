# BookTalk — one image: the FastAPI backend serving the built React frontend.
# Port 7860 by default (a host-provided $PORT wins), runs as uid 1000:
#   docker build -t booktalk . && docker run -p 7860:7860 --env-file backend/.env booktalk

FROM node:20-slim AS frontend
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BOOKTALK_DATA_DIR=/data \
    BOOKTALK_STATIC_DIR=/app/frontend/dist \
    BOOKTALK_DEMO=1 \
    BOOKTALK_MAX_UPLOAD_MB=20 \
    PORT=7860

RUN useradd -m -u 1000 user
WORKDIR /app/backend
COPY backend/pyproject.toml ./
COPY backend/app ./app
RUN pip install .

COPY --from=frontend /src/dist /app/frontend/dist
COPY deploy/seed /app/seed
COPY deploy/entrypoint.sh /app/entrypoint.sh
RUN mkdir -p /data && chown -R user:user /data /app && chmod +x /app/entrypoint.sh

USER user
EXPOSE 7860
CMD ["/app/entrypoint.sh"]
