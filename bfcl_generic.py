#!/usr/bin/env python3

import json
import math
import os
import sys
import threading
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

from openai import DefaultHttpxClient

from bfcl_eval.constants.model_config import (
    MODEL_CONFIG_MAPPING,
    ModelConfig,
)
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)


_LOG_DIR = Path(
    os.environ.get("BFCL_CALL_LOG_DIR", Path(__file__).resolve().parent / "bfcl_call_logs")
)
_RUN_ID = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{os.getpid()}"
_CALL_LOG_PATH = _LOG_DIR / f"calls_{_RUN_ID}.jsonl"
_PLOT_PATH = _LOG_DIR / f"calls_per_minute_{_RUN_ID}.svg"
_LOG_LOCK = threading.Lock()


def write_call_rate_plot():
    """Plot the number of HTTP attempts in the preceding 60 seconds."""
    if not _CALL_LOG_PATH.exists():
        if len(sys.argv) > 1 and sys.argv[1] == "generate":
            print("No new LLM calls were made; no call-rate plot was created.")
        return

    with _CALL_LOG_PATH.open(encoding="utf-8") as log_file:
        timestamps = sorted(
            json.loads(line)["unix_seconds"] for line in log_file if line.strip()
        )
    if not timestamps:
        return

    first, last = timestamps[0], timestamps[-1]
    duration = max(last - first, 1.0)
    # Limit the SVG size for long runs while retaining one-second resolution
    # for the usual BFCL run.
    sample_step = max(1, math.ceil(duration / 2000))
    sample_times = sorted(
        set(
            [
                first + second
                for second in range(0, math.ceil(duration) + 1, sample_step)
                if first + second <= last
            ]
            + timestamps
        )
    )

    def calls_in_last_minute(at_time):
        return bisect_right(timestamps, at_time) - bisect_right(
            timestamps, at_time - 60
        )

    counts = [calls_in_last_minute(at_time) for at_time in sample_times]
    peak = max(calls_in_last_minute(at_time) for at_time in timestamps)

    width, height = 1000, 480
    left, right, top, bottom = 85, 30, 90, 65
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = max(peak, 1)

    def x_pos(at_time):
        return left + (at_time - first) / duration * plot_width

    def y_pos(count):
        return top + (1 - count / y_max) * plot_height

    points = " ".join(
        f"{x_pos(at_time):.1f},{y_pos(count):.1f}"
        for at_time, count in zip(sample_times, counts)
    )
    start_utc = datetime.fromtimestamp(first, timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    end_utc = datetime.fromtimestamp(last, timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="85" y="32" font-family="sans-serif" font-size="22" fill="#17202a">LLM calls per minute</text>',
        f'<text x="85" y="56" font-family="sans-serif" font-size="13" fill="#52616b">'
        f"{start_utc} to {end_utc} · {len(timestamps)} HTTP attempts · peak {peak} calls / 60 s</text>",
    ]

    for count in sorted({round(y_max * index / 5) for index in range(6)}):
        y = y_pos(count)
        svg.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            'stroke="#e4e9ed"/>'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12" fill="#52616b">{count}</text>'
        )

    for index in range(6):
        elapsed_minutes = duration * index / 5 / 60
        x = left + plot_width * index / 5
        svg.append(
            f'<line x1="{x:.1f}" y1="{height - bottom}" x2="{x:.1f}" '
            f'y2="{height - bottom + 5}" stroke="#52616b"/>'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{height - bottom + 23}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#52616b">{elapsed_minutes:.1f}</text>'
        )

    svg.extend(
        [
            f'<polyline points="{points}" fill="none" stroke="#1976d2" stroke-width="2.5"/>',
            f'<circle cx="{x_pos(last):.1f}" cy="{y_pos(counts[-1]):.1f}" r="3.5" fill="#1976d2"/>',
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 12}" text-anchor="middle" '
            'font-family="sans-serif" font-size="13" fill="#17202a">Minutes since first call</text>',
            f'<text x="22" y="{top + plot_height / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 22 {top + plot_height / 2:.1f})" '
            'font-family="sans-serif" font-size="13" '
            'fill="#17202a">Calls in preceding 60 seconds</text>',
            "</svg>",
        ]
    )

    _PLOT_PATH.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"LLM call timestamps: {_CALL_LOG_PATH}")
    print(f"LLM calls/minute plot: {_PLOT_PATH}")
    print(f"Total HTTP attempts: {len(timestamps)}; peak rolling 60s: {peak}")


class GenericOpenAICompatibleHandler(OpenAICompletionsHandler):
    """
    Generic handler for any OpenAI-compatible endpoint:
      - Ollama
      - vLLM
      - SGLang
      - other compatible servers
    """

    def _build_client_kwargs(self):
        kwargs = super()._build_client_kwargs()
        # A request hook runs once per HTTP attempt, including SDK retries.
        # DefaultHttpxClient preserves the OpenAI SDK's normal timeout settings.
        kwargs["http_client"] = DefaultHttpxClient(
            event_hooks={"request": [self._record_llm_request]}
        )
        return kwargs

    def _record_llm_request(self, request):
        if request.method != "POST" or not request.url.path.rstrip("/").endswith(
            "/chat/completions"
        ):
            return

        now = datetime.now(timezone.utc)
        event = {
            "timestamp_utc": now.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "unix_seconds": now.timestamp(),
            "model": self.model_name,
            "registry_name": self.registry_name,
        }
        with _LOG_LOCK:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            with _CALL_LOG_PATH.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(event) + "\n")

    def _query_FC(self, inference_data: dict):
        messages = inference_data["message"]
        tools = inference_data["tools"]

        inference_data["inference_input_log"] = {
            "message": repr(messages),
            "tools": tools,
        }

        kwargs = {
            "messages": messages,
            "model": self.model_name,
            "temperature": self.temperature,
        }

        if tools:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {
            "message": repr(inference_data["message"])
        }

        return self.generate_with_backoff(
            messages=inference_data["message"],
            model=self.model_name,
            temperature=self.temperature,
        )


def register_generic_model():
    actual_model = os.environ.get("BFCL_GENERIC_MODEL")
    if not actual_model:
        raise RuntimeError(
            "BFCL_GENERIC_MODEL is not set.\n"
            "Example:\n"
            "  export BFCL_GENERIC_MODEL='qwen2.5-coder:14b'"
        )

    registry_name = os.environ.get(
        "BFCL_REGISTRY_NAME",
        "generic-openai-fc",
    )

    MODEL_CONFIG_MAPPING[registry_name] = ModelConfig(
        model_name=actual_model,
        display_name=f"{actual_model} (Generic OpenAI FC)",
        url="local",
        org="Local",
        license="N/A",
        model_handler=GenericOpenAICompatibleHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    )

    return registry_name


if __name__ == "__main__":
    register_generic_model()

    # Import BFCL CLI only AFTER registering our model.
    from bfcl_eval.__main__ import cli

    try:
        cli()
    finally:
        write_call_rate_plot()
