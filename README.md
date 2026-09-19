# BFCL Generic Adapter

A small adapter that lets [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) test models served by OpenAI-compatible endpoints such as vLLM and Ollama.

It registers a native function-calling model with BFCL and records every Chat Completions HTTP attempt. After generation, it creates a JSONL timestamp log and an SVG chart showing calls in the preceding 60 seconds.

## Requirements

- Python 3.12 (tested)
- A model and server that support native OpenAI-style tool calls

Install the tested dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Create the smoke-test selection file:

```bash
cp examples/test_case_ids_to_generate.example.json test_case_ids_to_generate.json
```

For vLLM:

```bash
export BFCL_GENERIC_MODEL='exact-model-id-from-v1-models'
export OPENAI_BASE_URL='http://vllm-host:8000/v1'
export OPENAI_API_KEY='EMPTY'  # Replace when authentication is enabled.
./examples/run_vllm.sh
```

For Ollama:

```bash
export BFCL_GENERIC_MODEL='qwen3-coder:30b'
./examples/run_ollama.sh
```

`BFCL_GENERIC_MODEL` must be the exact model ID accepted by the server. `BFCL_REGISTRY_NAME` is an optional local BFCL label.

## Test selection

The example scripts use `--run-ids`. Add or remove IDs in `test_case_ids_to_generate.json` to change the selected cases.

To run complete categories instead, remove `--run-ids` from the generation command and pass `--test-category`.

## Output

BFCL writes benchmark output under `result/` and `score/`. Request monitoring is written under `bfcl_call_logs/`:

- `calls_*.jsonl`: one UTC timestamp per HTTP attempt, including retries
- `calls_per_minute_*.svg`: rolling 60-second request count over the run

The adapter measures only requests made by its own BFCL process. It does not include calls from other users of the model server.

