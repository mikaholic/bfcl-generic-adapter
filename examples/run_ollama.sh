#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export BFCL_PROJECT_ROOT="$REPO_DIR"
: "${BFCL_GENERIC_MODEL:?Set BFCL_GENERIC_MODEL to an installed Ollama model}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:11434/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
export BFCL_REGISTRY_NAME="${BFCL_REGISTRY_NAME:-generic-ollama-FC}"
export BFCL_CALLS_PER_MINUTE="${BFCL_CALLS_PER_MINUTE:-infinite}"

TEST_CATEGORIES="${BFCL_TEST_CATEGORIES:-simple_python,multiple,parallel,irrelevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param}"

if [[ ! -f "$REPO_DIR/test_case_ids_to_generate.json" ]]; then
    echo "Missing $REPO_DIR/test_case_ids_to_generate.json" >&2
    echo "Run: cp examples/test_case_ids_to_generate.example.json test_case_ids_to_generate.json" >&2
    exit 1
fi

"$PYTHON_BIN" "$REPO_DIR/bfcl_generic.py" models | grep -F "$BFCL_REGISTRY_NAME"

"$PYTHON_BIN" "$REPO_DIR/bfcl_generic.py" generate \
    --model "$BFCL_REGISTRY_NAME" \
    --run-ids \
    --num-threads "${BFCL_NUM_THREADS:-1}"

"$PYTHON_BIN" "$REPO_DIR/bfcl_generic.py" evaluate \
    --model "$BFCL_REGISTRY_NAME" \
    --test-category "$TEST_CATEGORIES" \
    --partial-eval
