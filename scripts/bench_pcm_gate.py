#!/usr/bin/env python3
"""Latency gate for the whisper.cpp raw PCM endpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.parse
import urllib.request
import wave


DEFAULT_AUDIO = "/Users/alan/ai-inbound-backend/data/audio/TRANSCRIBE_TEST-986b6dc2.wav"
DEFAULT_URL = "http://127.0.0.1:50061/inference-pcm"

FIELDS = [
    "mel_ms",
    "encode_ms",
    "decode_ms",
    "sample_ms",
    "batchd_ms",
    "prompt_ms",
    "postprocess_ms",
    "overhead_ms",
    "e2e_ms",
    "client_e2e_ms",
]

GATES = {
    "encode_ms": 70.0,
    "decode_ms": 15.0,
    "overhead_ms": 15.0,
    "e2e_ms": 100.0,
}


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def read_pcm(path: str) -> bytes:
    with wave.open(path, "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise SystemExit(
                "audio must be 16 kHz mono signed 16-bit WAV for /inference-pcm"
            )
        return wav.readframes(wav.getnframes())


def build_url(base: str) -> str:
    params = {
        "language": "es",
        "temperature": "0",
        "temperature_inc": "0",
        "response_format": "json",
        "no_timestamps": "true",
        "beam_size": "1",
        "best_of": "1",
        "audio_ctx": "256",
        "max_context": "0",
        "max_len": "40",
        "timings": "true",
    }
    separator = "&" if "?" in base else "?"
    return base + separator + urllib.parse.urlencode(params)


def run_once(url: str, pcm: bytes) -> dict[str, float]:
    request = urllib.request.Request(
        url,
        data=pcm,
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(pcm)),
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    client_e2e_ms = (time.perf_counter() - started) * 1000.0
    timings = payload["timings"]
    timings["overhead_ms"] = max(
        0.0,
        float(timings.get("overhead_ms", 0.0))
        or (
            float(timings.get("e2e_ms", 0.0))
            - float(timings.get("encode_ms", 0.0))
            - float(timings.get("decode_ms", 0.0))
        ),
    )
    timings["client_e2e_ms"] = client_e2e_ms
    return {field: float(timings.get(field, 0.0)) for field in FIELDS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=DEFAULT_AUDIO)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    pcm = read_pcm(args.audio)
    url = build_url(args.url)

    for _ in range(args.warmup):
        run_once(url, pcm)

    rows = []
    for index in range(args.runs):
        row = run_once(url, pcm)
        rows.append(row)
        print(
            f"run={index + 1:02d} "
            + " ".join(f"{field}={row[field]:.2f}" for field in FIELDS)
        )

    print("\nsummary")
    failed = False
    for field in FIELDS:
        values = [row[field] for row in rows]
        p50 = statistics.median(values)
        p95 = percentile(values, 95)
        line = f"{field}: p50={p50:.2f} p95={p95:.2f}"
        if field in GATES:
            ok = p95 <= GATES[field]
            failed = failed or not ok
            line += f" gate<={GATES[field]:.2f} {'PASS' if ok else 'FAIL'}"
        print(line)

    return 1 if failed and not args.no_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
