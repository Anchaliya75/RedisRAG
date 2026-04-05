#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Setup venv + deps
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv..."
    python3 -m venv "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "${SCRIPT_DIR}/requirements.txt"
echo "Dependencies ready."

# Ensure Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Docker not running. Starting Docker Desktop..."
    open -a Docker
    # Wait up to 30s for Docker to be ready
    for i in $(seq 1 30); do
        docker info >/dev/null 2>&1 && break
        sleep 1
    done
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: Docker failed to start after 30s. Please start it manually."
        exit 1
    fi
    echo "Docker started."
fi

# Start redis-stack if not running
if docker ps --format '{{.Names}}' | grep -q '^redis-stack$'; then
    echo "redis-stack already running."
elif lsof -i :6379 >/dev/null 2>&1; then
    echo "Port 6379 already in use, skipping docker start."
else
    echo "Starting redis-stack..."
    docker rm -f redis-stack 2>/dev/null || true
    docker run -d --name redis-stack -p 6379:6379 \
        -e REDIS_ARGS="--requirepass redis" \
        redis/redis-stack:latest
    sleep 3
fi
echo "Redis ready."

# Pre-download the embedding model
echo "Downloading embedding model..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
echo "Model ready."

echo ""
echo "Setup complete. Now run:"
echo "  make ingest"
echo "  make retrieve"
