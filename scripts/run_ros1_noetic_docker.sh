#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="duckiebot_ros1_noetic:latest"
BUILD_ARGS=(--progress=plain)

if [[ "${1:-}" == "--no-cache" ]]; then
  BUILD_ARGS+=(--no-cache --pull)
fi

echo "[1/2] Building Docker image: ${IMAGE_NAME}"
echo "      Build args: ${BUILD_ARGS[*]}"
DOCKER_BUILDKIT=1 docker build "${BUILD_ARGS[@]}" \
  -t "${IMAGE_NAME}" \
  -f "${REPO_ROOT}/docker/ros1-noetic/Dockerfile" \
  "${REPO_ROOT}"

echo "[2/2] Starting container with /dev/input passthrough"
docker run --rm -it \
  --network host \
  --device /dev/input:/dev/input \
  -v "${REPO_ROOT}:/workspace/duckiebot" \
  -w /workspace/duckiebot \
  "${IMAGE_NAME}" \
  bash
