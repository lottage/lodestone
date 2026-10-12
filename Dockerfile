# Lodestone gateway image (also runs the one-shot model downloader of the Compose stack). Standard library only: no pip.
#
# Shipped as `Dockerfile` in the lodestone repository, whose root holds the lodestone/ package:
#     docker build -t ghcr.io/lottage/lodestone:latest .
# Run by docker-compose.yml (python -m lodestone.installer --docker); alone:
#     docker run --rm -p 127.0.0.1:8000:8000 -e COORDINATOR_URL=http://host.docker.internal:8001/v1 \
#         -e LODESTONE_API_KEY=... ghcr.io/lottage/lodestone:latest
# (From the private monorepo the same file builds with the context StoneSage/backend.)
# Base image: an exact version tag, never a floating one. digest: pin at release (docker buildx imagetools inspect python:3.12.7-slim-bookworm,
# then FROM python:3.12.7-slim-bookworm@sha256:<digest>); tests/test_docker_files.py checks every FROM has an exact tag.
FROM python:3.12.7-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app

# Unprivileged user. /models is where the Compose stack mounts the model volume: a named volume copies this ownership on
# first use, so the one-shot downloader can write there without root.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin lodestone \
 && mkdir -p /models /config \
 && chown lodestone:lodestone /models /config

WORKDIR /app
COPY lodestone /app/lodestone
USER lodestone

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

# No secrets in the image: the key comes from LODESTONE_API_KEY (or config.json gateway.api_key) at run time.
CMD ["python", "-m", "lodestone.gateway", "--host", "0.0.0.0", "--port", "8000"]
