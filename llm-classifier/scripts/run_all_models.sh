#!/usr/bin/env bash
# run_all_models.sh — Run LLM classification for all three models sequentially,
# then consolidate verdicts into result xlsx files.
#
# Usage (from any directory):
#   bash papers/tg-roger-joshua/scripts/run_all_models.sh
#
# Already-cached verdicts are skipped automatically (re-runs are cheap).
# GPU is used when available via nvidia container toolkit.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CORPUS="papers/tg-roger-joshua/sections/_auto/corpus.xlsx"
MODELS=(
    "qwen2.5:14b-instruct"
    "mistral:7b-instruct"
    "llama3.1:8b"
)

# Detect GPU support (CDI-first, falls back to legacy --gpus flag).
GPU_FLAGS=()
if docker info 2>/dev/null | grep -q "nvidia.com/gpu"; then
    GPU_FLAGS=(--device=nvidia.com/gpu=all)
    echo "[run_all_models] GPU detected (CDI) — using NVIDIA acceleration"
elif docker info 2>/dev/null | grep -qi "nvidia"; then
    GPU_FLAGS=(--gpus all)
    echo "[run_all_models] GPU detected (legacy runtime) — using NVIDIA acceleration"
else
    echo "[run_all_models] No GPU — running on CPU (slow)"
fi

TGII_FLAG=()
if [[ -d "/home/cardel/repositorios/tgIITulua" ]]; then
    TGII_FLAG=(-v "/home/cardel/repositorios/tgIITulua:/home/cardel/repositorios/tgIITulua:ro")
fi

docker volume create papers-project-ollama-models >/dev/null 2>&1 || true

for MODEL in "${MODELS[@]}"; do
    echo "========================================================"
    echo "  Model: $MODEL  |  $(date)"
    echo "========================================================"
    # `|| true` keeps the pipeline going when classify.py exits 1
    # (some project verdicts failed); per-model failures are summarised
    # in the classify.py stdout and again by evaluate.py downstream.
    docker run --rm \
      -e PYTHONUNBUFFERED=1 \
      "${GPU_FLAGS[@]}" \
      -v papers-project-ollama-models:/root/.ollama \
      "${TGII_FLAG[@]}" \
      -v "$PROJECT_ROOT:/workspace" \
      -w /workspace \
      papers-project/paper-data:latest \
      python papers/tg-roger-joshua/scripts/classify.py \
        --input-xlsx "$CORPUS" \
        --model "$MODEL" || true
    echo "  Done: $MODEL  |  $(date)"
done

echo "========================================================"
echo "  All models complete. Consolidating results ..."
echo "========================================================"
docker run --rm \
  -e PYTHONUNBUFFERED=1 \
  "${GPU_FLAGS[@]}" \
  -v papers-project-ollama-models:/root/.ollama \
  "${TGII_FLAG[@]}" \
  -v "$PROJECT_ROOT:/workspace" \
  -w /workspace \
  papers-project/paper-data:latest \
  python papers/tg-roger-joshua/scripts/generate_results.py

echo "========================================================"
echo "  Evaluating on gold set ..."
echo "========================================================"
docker run --rm \
  -e PYTHONUNBUFFERED=1 \
  "${GPU_FLAGS[@]}" \
  -v papers-project-ollama-models:/root/.ollama \
  "${TGII_FLAG[@]}" \
  -v "$PROJECT_ROOT:/workspace" \
  -w /workspace \
  papers-project/paper-data:latest \
  python papers/tg-roger-joshua/scripts/evaluate.py

echo "========================================================"
echo "  Comparing models ..."
echo "========================================================"
docker run --rm \
  -e PYTHONUNBUFFERED=1 \
  "${GPU_FLAGS[@]}" \
  -v papers-project-ollama-models:/root/.ollama \
  "${TGII_FLAG[@]}" \
  -v "$PROJECT_ROOT:/workspace" \
  -w /workspace \
  papers-project/paper-data:latest \
  python papers/tg-roger-joshua/scripts/compare_models.py

echo ""
echo "All done. Outputs in papers/tg-roger-joshua/sections/_auto/"
echo "  llm_results_*.xlsx       — per-project verdicts"
echo "  llm_distribution_*.xlsx  — category distributions"
echo "  metrics_*.json           — accuracy / F1 on gold set"
echo "  model_comparison.xlsx    — cross-model comparison table"
