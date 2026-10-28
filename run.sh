#!/usr/bin/env bash

set -o pipefail
cd "$(dirname "$0")" || exit 1

IMAGE=cotracker3
DEMO=""
REBUILD=0
GPU=0
MAKE_REF=0
EXTRA=()

usage() {
  cat <<'EOF'
Usage:
  bash run.sh                     build the image if needed and run all tests
  bash run.sh --demo video.mp4    demo on your video -> results/demo/<name>/
  bash run.sh --rebuild           rebuild the image first
  bash run.sh --gpu               GPU image, run with --gpus all
  bash run.sh --make-reference    recreate reference/
  bash run.sh -- -k shift         everything after -- goes to pytest (or run_demo.py)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --demo)
      [ $# -ge 2 ] || { echo "error: --demo needs a path to a video" >&2; exit 2; }
      DEMO="$2"; shift 2 ;;
    --rebuild) REBUILD=1; shift ;;
    --gpu) GPU=1; shift ;;
    --make-reference) MAKE_REF=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "error: unknown option $1" >&2; usage; exit 2 ;;
  esac
done

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) export MSYS_NO_PATHCONV=1; host_path() { (cd "$1" && pwd -W); } ;;
  *) host_path() { (cd "$1" && pwd); } ;;
esac

USER_ARGS=()
if [ "$(uname -s)" = "Linux" ] && [ "$(id -u)" != "0" ]; then
  USER_ARGS=(--user "$(id -u):$(id -g)")
fi

PLATFORM_ARGS=(--platform linux/amd64)
BUILD_ARGS=()
GPU_ARGS=()
TAG="$IMAGE:test"
if [ "$GPU" = 1 ]; then
  TAG="$IMAGE-gpu:test"
  BUILD_ARGS=(--build-arg REQUIREMENTS=requirements-gpu.txt)
  GPU_ARGS=(--gpus all -e COTRACKER_DEVICE=cuda)
fi

command -v docker >/dev/null 2>&1 || { echo "error: docker not found, install Docker Desktop or Docker Engine" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "error: Docker daemon is not running (start Docker Desktop)" >&2; exit 1; }

echo "== CoTracker3 in Docker =="
if [ "$REBUILD" = 1 ] || ! docker image inspect "$TAG" >/dev/null 2>&1; then
  echo "[1/3] Building image $TAG (first build takes 3-5 min, needs internet)"
  docker build "${PLATFORM_ARGS[@]}" ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} --target test -t "$TAG" . \
    || { echo "error: docker build failed, see TESTING.md -> troubleshooting" >&2; exit 1; }
else
  echo "[1/3] Image $TAG found (use --rebuild to rebuild)"
fi

mkdir -p results
RUN_ARGS=(run --rm --network none "${PLATFORM_ARGS[@]}"
          ${USER_ARGS[@]+"${USER_ARGS[@]}"} ${GPU_ARGS[@]+"${GPU_ARGS[@]}"}
          -v "$(host_path results):/app/results")

if [ -n "$DEMO" ]; then
  [ -f "$DEMO" ] || { echo "error: video not found: $DEMO" >&2; exit 2; }
  name="$(basename "$DEMO")"
  MASK_ARGS=()
  DEMO_ARGS=()
  i=0
  while [ "$i" -lt "${#EXTRA[@]}" ]; do
    if [ "${EXTRA[$i]}" = "--mask" ] && [ $((i + 1)) -lt "${#EXTRA[@]}" ]; then
      mask="${EXTRA[$((i + 1))]}"
      [ -f "$mask" ] || { echo "error: mask not found: $mask" >&2; exit 2; }
      MASK_ARGS=(-v "$(host_path "$(dirname "$mask")"):/mask:ro")
      DEMO_ARGS+=(--mask "/mask/$(basename "$mask")")
      i=$((i + 2))
    else
      DEMO_ARGS+=("${EXTRA[$i]}")
      i=$((i + 1))
    fi
  done
  echo "[2/3] Running demo on $DEMO (offline)"
  docker "${RUN_ARGS[@]}" -v "$(host_path "$(dirname "$DEMO")"):/input:ro" \
    ${MASK_ARGS[@]+"${MASK_ARGS[@]}"} "$TAG" \
    python scripts/run_demo.py --video "/input/$name" ${DEMO_ARGS[@]+"${DEMO_ARGS[@]}"}
  rc=$?
  out="results/demo/${name%.*}"
  DONE_MSG="DEMO DONE!"
elif [ "$MAKE_REF" = 1 ]; then
  echo "[2/3] Creating reference/ (offline)"
  mkdir -p reference
  docker "${RUN_ARGS[@]}" -v "$(host_path reference):/app/reference" "$TAG" \
    python scripts/make_reference.py
  rc=$?
  out="reference"
  DONE_MSG="REFERENCE DONE!"
else
  echo "[2/3] Running tests (offline, --network none)"
  docker "${RUN_ARGS[@]}" -v "$(host_path reference):/app/reference:ro" "$TAG" \
    python -m pytest ${EXTRA[@]+"${EXTRA[@]}"}
  rc=$?
  out="results/tests"
  DONE_MSG="TESTS DONE!"
fi

echo "[3/3] Summary"
if [ -d "$out" ]; then
  echo "Files in ./$out:"
  (cd "$out" && find . -type f | sort | sed 's|^\./|  |')
fi
echo "Elapsed: ${SECONDS}s"
if [ "$rc" = 0 ]; then
  echo "$DONE_MSG"
else
  echo "FAILED (exit code $rc), see the output above and TESTING.md -> troubleshooting"
fi
exit "$rc"
