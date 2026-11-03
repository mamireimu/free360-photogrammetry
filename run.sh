#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: bash run.sh INPUT.mp4 OUTPUT_DIR [--gpus 0,1,2,3] [--duration 60] [--resume]" >&2
  exit 2
fi
command -v docker >/dev/null || { echo "Docker is required." >&2; exit 1; }
input=$(realpath -- "$1")
[[ -f "$input" ]] || { echo "Input is not a file: $input" >&2; exit 1; }
mkdir -p -- "$2"
output=$(realpath -- "$2")
shift 2
# Docker --mount separates fields by comma.
if [[ "$input" == *,* || "$output" == *,* ]]; then
  echo "Use input/output paths without commas." >&2
  exit 2
fi
exec docker run --rm --init --gpus all \
  --user "$(id -u):$(id -g)" --env HOME=/tmp --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,video --shm-size=2g \
  --mount "type=bind,source=$input,target=/input/video.mp4,readonly" \
  --mount "type=bind,source=$output,target=/output" \
  "${MP4_MODEL_IMAGE:-free360-photogrammetry:1}" /input/video.mp4 -o /output "$@"
