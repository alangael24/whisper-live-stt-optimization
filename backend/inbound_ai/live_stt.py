"""Precision-first browser microphone transcription over WebSocket."""

from __future__ import annotations

import json
import math
import re
import tempfile
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from .audio_stt import transcribe_audio
from .config import Settings


SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
STEP_SECONDS = 0.1
MIN_WINDOW_SECONDS = 1.2
SPEECH_RMS_THRESHOLD = 260
SILENCE_FINALIZE_SECONDS = 0.3
PRE_ROLL_SECONDS = 0.25
MAX_UTTERANCE_SECONDS = 25
TRIM_FRAME_SECONDS = 0.02
TRIM_PADDING_SECONDS = 0.16
TRIM_RMS_THRESHOLD = 120
SPECULATIVE_ENABLED = False
SPECULATIVE_INTERVAL_SECONDS = 0.55
SPECULATIVE_MIN_NEW_AUDIO_SECONDS = 0.35
SPECULATIVE_REUSE_TOLERANCE_SECONDS = 0.45
SPECULATIVE_WAIT_SECONDS = 0.28


class PcmRingBuffer:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.pre_roll_capacity = int(sample_rate * BYTES_PER_SAMPLE * PRE_ROLL_SECONDS)
        self.utterance_capacity = int(
            sample_rate * BYTES_PER_SAMPLE * MAX_UTTERANCE_SECONDS
        )
        self.pre_roll = bytearray()
        self.utterance = bytearray()
        self.lock = threading.Lock()
        self.in_speech = False
        self.last_speech_at = 0.0

    def append(self, payload: bytes) -> None:
        if len(payload) < BYTES_PER_SAMPLE:
            return
        if len(payload) % BYTES_PER_SAMPLE:
            payload = payload[:-1]
        now = time.monotonic()
        speech = pcm_rms(payload) >= SPEECH_RMS_THRESHOLD
        with self.lock:
            self.pre_roll.extend(payload)
            if len(self.pre_roll) > self.pre_roll_capacity:
                del self.pre_roll[: len(self.pre_roll) - self.pre_roll_capacity]
            if speech:
                if not self.in_speech:
                    self.utterance = bytearray(self.pre_roll)
                    self.in_speech = True
                else:
                    self.utterance.extend(payload)
                if len(self.utterance) > self.utterance_capacity:
                    del self.utterance[: len(self.utterance) - self.utterance_capacity]
                self.last_speech_at = now
                return
            if self.in_speech:
                self.utterance.extend(payload)
                if len(self.utterance) > self.utterance_capacity:
                    del self.utterance[: len(self.utterance) - self.utterance_capacity]

    def extract_utterance_if_ready(self, silence_seconds: float, force: bool = False) -> bytes:
        with self.lock:
            if not self.in_speech or not self.utterance:
                return b""
            if not force:
                if not self.last_speech_at:
                    return b""
                if time.monotonic() - self.last_speech_at < silence_seconds:
                    return b""
            utterance = bytes(self.utterance)
            self.utterance = bytearray()
            self.pre_roll = bytearray()
            self.in_speech = False
            self.last_speech_at = 0.0
            return utterance

    def snapshot_utterance(self) -> bytes:
        with self.lock:
            if not self.in_speech or not self.utterance:
                return b""
            return bytes(self.utterance)


@dataclass
class TranscriptState:
    final_text: str = ""

    def commit_final(self, final_hypothesis: str) -> str:
        final_text, _ = self.commit_final_delta(final_hypothesis)
        return final_text

    def commit_final_delta(self, final_hypothesis: str) -> tuple[str, str]:
        new_part = remove_confirmed_overlap(final_hypothesis, self.final_text)
        new_part = remove_duplicate_prefix(self.final_text, new_part)
        if new_part.strip():
            self.final_text = join_words(self.final_text, new_part)
        return self.final_text, new_part


@dataclass
class SttOutcome:
    hypothesis: str
    error: str
    inference_ms: int
    duration_ms: int
    source_bytes: int
    trimmed_bytes: int


@dataclass
class SpeculativeResult:
    hypothesis: str
    inference_ms: int
    duration_ms: int
    source_bytes: int
    trimmed_bytes: int
    completed_at: float


class LiveSttBridge:
    def __init__(self, websocket: Any, settings: Settings):
        self.websocket = websocket
        self.settings = settings
        self.ring = PcmRingBuffer(SAMPLE_RATE)
        self.state = TranscriptState()
        self.closed = threading.Event()
        self.started_at = time.monotonic()
        self.speculative_lock = threading.Lock()
        self.speculative_inflight = False
        self.latest_speculative: Optional[SpeculativeResult] = None
        self.last_speculative_source_bytes = 0
        self.last_speculative_started_at = 0.0
        self.speculative_generation = 0

    def run(self) -> None:
        worker = threading.Thread(target=self._infer_loop, name="live-stt", daemon=True)
        worker.start()
        self.websocket.send_json(
            {
                "type": "ready",
                "mode": "precision_final",
                "sample_rate": SAMPLE_RATE,
                "silence_finalize_ms": int(SILENCE_FINALIZE_SECONDS * 1000),
            }
        )
        try:
            while not self.closed.is_set():
                message = self.websocket.recv_message()
                if message is None:
                    break
                opcode, payload = message
                if opcode == 0x1:
                    self._handle_text(payload)
                elif opcode == 0x2:
                    self.ring.append(payload)
        except (EOFError, OSError, json.JSONDecodeError):
            pass
        finally:
            self.closed.set()
            self.websocket.close()

    def _handle_text(self, payload: bytes) -> None:
        event = json.loads(payload.decode("utf-8"))
        if event.get("type") == "stop":
            utterance = self.ring.extract_utterance_if_ready(
                SILENCE_FINALIZE_SECONDS,
                force=True,
            )
            if utterance:
                self._transcribe_final_utterance(utterance)
            self.closed.set()
        if event.get("type") == "reset":
            self.state = TranscriptState()
            self._clear_speculative_result()

    def _infer_loop(self) -> None:
        while not self.closed.is_set():
            time.sleep(STEP_SECONDS)
            self._maybe_start_speculative_transcription()
            self._maybe_finalize_on_silence()

    def _maybe_finalize_on_silence(self) -> None:
        utterance = self.ring.extract_utterance_if_ready(SILENCE_FINALIZE_SECONDS)
        if not utterance:
            return
        self._transcribe_final_utterance(utterance)

    def _transcribe_final_utterance(self, pcm: bytes) -> None:
        if len(pcm) < int(MIN_WINDOW_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE):
            return
        delivery_started = time.perf_counter()
        trimmed = trim_pcm_for_stt(pcm)
        speculative = self._wait_for_matching_speculative_result(
            len(pcm),
            len(trimmed),
        )
        if speculative:
            delivery_ms = int((time.perf_counter() - delivery_started) * 1000)
            self._send_final_transcript(
                speculative.hypothesis,
                speculative.inference_ms,
                speculative.duration_ms,
                delivery_ms,
                speculative=True,
            )
            self._clear_speculative_result()
            return

        outcome = self._transcribe_pcm(pcm, "final-stt-")
        if outcome.error:
            self.websocket.send_json({"type": "error", "detail": outcome.error})
            return
        delivery_ms = int((time.perf_counter() - delivery_started) * 1000)
        self._send_final_transcript(
            outcome.hypothesis,
            outcome.inference_ms,
            outcome.duration_ms,
            delivery_ms,
            speculative=False,
        )
        self._clear_speculative_result()

    def _send_final_transcript(
        self,
        hypothesis: str,
        inference_ms: int,
        duration_ms: int,
        delivery_ms: int,
        speculative: bool,
    ) -> None:
        hypothesis = hypothesis.strip()
        if not hypothesis:
            return
        final_text, committed_hypothesis = self.state.commit_final_delta(hypothesis)
        if not committed_hypothesis.strip():
            return
        rtf = inference_ms / max(duration_ms, 1)
        self.websocket.send_json(
            {
                "type": "transcript",
                "final_text": final_text,
                "hypothesis": committed_hypothesis,
                "raw_hypothesis": hypothesis,
                "inference_ms": inference_ms,
                "delivery_ms": delivery_ms,
                "window_ms": duration_ms,
                "rtf": round(rtf, 3),
                "final": True,
                "speculative": speculative,
            }
        )

    def _maybe_start_speculative_transcription(self) -> None:
        if not SPECULATIVE_ENABLED:
            return
        pcm = self.ring.snapshot_utterance()
        if len(pcm) < int(MIN_WINDOW_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE):
            return

        now = time.monotonic()
        min_new_bytes = int(
            SPECULATIVE_MIN_NEW_AUDIO_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE
        )
        with self.speculative_lock:
            if len(pcm) < self.last_speculative_source_bytes:
                self.latest_speculative = None
                self.last_speculative_source_bytes = 0
            if self.speculative_inflight:
                return
            if now - self.last_speculative_started_at < SPECULATIVE_INTERVAL_SECONDS:
                return
            if (
                self.latest_speculative
                and len(pcm) - self.last_speculative_source_bytes < min_new_bytes
            ):
                return
            self.speculative_inflight = True
            self.last_speculative_started_at = now
            self.last_speculative_source_bytes = len(pcm)
            generation = self.speculative_generation

        worker = threading.Thread(
            target=self._run_speculative_transcription,
            args=(pcm, generation),
            name="live-stt-speculative",
            daemon=True,
        )
        worker.start()

    def _run_speculative_transcription(self, pcm: bytes, generation: int) -> None:
        outcome = self._transcribe_pcm(pcm, "spec-stt-")
        with self.speculative_lock:
            self.speculative_inflight = False
            if generation != self.speculative_generation:
                return
            if outcome.error or not outcome.hypothesis.strip():
                return
            self.latest_speculative = SpeculativeResult(
                hypothesis=outcome.hypothesis,
                inference_ms=outcome.inference_ms,
                duration_ms=outcome.duration_ms,
                source_bytes=outcome.source_bytes,
                trimmed_bytes=outcome.trimmed_bytes,
                completed_at=time.monotonic(),
            )

    def _wait_for_matching_speculative_result(
        self,
        source_bytes: int,
        trimmed_bytes: int,
    ) -> Optional[SpeculativeResult]:
        if not SPECULATIVE_ENABLED:
            return None
        deadline = time.monotonic() + SPECULATIVE_WAIT_SECONDS
        while True:
            result = self._matching_speculative_result(source_bytes, trimmed_bytes)
            if result:
                return result
            with self.speculative_lock:
                inflight = self.speculative_inflight
            if not inflight or time.monotonic() >= deadline:
                return None
            time.sleep(0.01)

    def _matching_speculative_result(
        self,
        source_bytes: int,
        trimmed_bytes: int,
    ) -> Optional[SpeculativeResult]:
        tolerance = int(
            SPECULATIVE_REUSE_TOLERANCE_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE
        )
        with self.speculative_lock:
            result = self.latest_speculative
            if not result:
                return None
            if source_bytes - result.source_bytes > tolerance:
                return None
            if trimmed_bytes - result.trimmed_bytes > tolerance:
                return None
            return result

    def _clear_speculative_result(self) -> None:
        with self.speculative_lock:
            self.speculative_generation += 1
            self.latest_speculative = None
            self.last_speculative_source_bytes = 0

    def _transcribe_pcm(self, pcm: bytes, prefix: str) -> SttOutcome:
        source_bytes = len(pcm)
        trimmed = trim_pcm_for_stt(pcm)
        audio_file = self._write_temp_wav(trimmed, prefix)
        try:
            started = time.perf_counter()
            result = transcribe_audio(
                audio_file,
                self.settings.stt_final_command or self.settings.stt_command,
                self.settings.stt_final_server_url,
                self.settings.stt_language,
                self.state.final_text[-300:],
            )
            inference_ms = int((time.perf_counter() - started) * 1000)
        finally:
            try:
                Path(audio_file).unlink()
            except OSError:
                pass

        duration_ms = int(len(trimmed) / (SAMPLE_RATE * BYTES_PER_SAMPLE) * 1000)
        return SttOutcome(
            hypothesis=result.transcript.strip(),
            error=result.error,
            inference_ms=inference_ms,
            duration_ms=duration_ms,
            source_bytes=source_bytes,
            trimmed_bytes=len(trimmed),
        )

    def _write_temp_wav(self, pcm: bytes, prefix: str) -> str:
        Path(self.settings.audio_upload_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix=prefix,
            dir=self.settings.audio_upload_dir,
            delete=False,
        ) as tmp:
            audio_file = tmp.name
        write_pcm_wav(audio_file, pcm, SAMPLE_RATE)
        return audio_file


def write_pcm_wav(path: str, pcm: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def pcm_rms(payload: bytes) -> float:
    if len(payload) < BYTES_PER_SAMPLE:
        return 0.0
    total = 0
    count = 0
    for index in range(0, len(payload) - 1, BYTES_PER_SAMPLE):
        sample = int.from_bytes(payload[index : index + 2], "little", signed=True)
        total += sample * sample
        count += 1
    return math.sqrt(total / max(1, count))


def trim_pcm_for_stt(
    pcm: bytes,
    sample_rate: int = SAMPLE_RATE,
    threshold: int = TRIM_RMS_THRESHOLD,
) -> bytes:
    if len(pcm) < BYTES_PER_SAMPLE:
        return pcm
    if len(pcm) % BYTES_PER_SAMPLE:
        pcm = pcm[:-1]
    frame_bytes = max(
        BYTES_PER_SAMPLE,
        int(sample_rate * BYTES_PER_SAMPLE * TRIM_FRAME_SECONDS),
    )
    frame_bytes -= frame_bytes % BYTES_PER_SAMPLE
    speech_frames: List[tuple[int, int]] = []
    for start in range(0, len(pcm), frame_bytes):
        end = min(len(pcm), start + frame_bytes)
        if pcm_rms(pcm[start:end]) >= threshold:
            speech_frames.append((start, end))
    if not speech_frames:
        return pcm
    padding_bytes = int(sample_rate * BYTES_PER_SAMPLE * TRIM_PADDING_SECONDS)
    padding_bytes -= padding_bytes % BYTES_PER_SAMPLE
    start = max(0, speech_frames[0][0] - padding_bytes)
    end = min(len(pcm), speech_frames[-1][1] + padding_bytes)
    return pcm[start:end]


def word_list(text: str) -> List[str]:
    return re.findall(r"\S+", re.sub(r"\s+", " ", text).strip())


def normalized_words(text: str) -> List[str]:
    return [normalize_word(word) for word in word_list(text)]


def normalize_word(word: str) -> str:
    without_marks = "".join(
        char
        for char in unicodedata.normalize("NFKD", word.lower())
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^\w]+", "", without_marks)


def remove_confirmed_overlap(hypothesis: str, final_text: str) -> str:
    hyp_words = word_list(hypothesis)
    final_words = word_list(final_text)
    if not hyp_words or not final_words:
        return " ".join(hyp_words)
    hyp_norm = normalized_words(hypothesis)
    final_norm = normalized_words(final_text)
    max_overlap = min(len(hyp_norm), len(final_norm), 30)
    for size in range(max_overlap, 0, -1):
        if final_norm[-size:] == hyp_norm[:size]:
            return " ".join(hyp_words[size:])
    return " ".join(hyp_words)


def append_without_duplicate_overlap(existing: str, addition: str) -> str:
    return join_words(existing, remove_duplicate_prefix(existing, addition))


def remove_duplicate_prefix(existing: str, addition: str) -> str:
    addition = addition.strip()
    if not addition:
        return ""
    if not existing.strip():
        return addition
    existing_words = word_list(existing)
    addition_words = word_list(addition)
    existing_norm = normalized_words(existing)
    addition_norm = normalized_words(addition)
    max_overlap = min(len(existing_norm), len(addition_norm), 30)
    for size in range(max_overlap, 0, -1):
        if existing_norm[-size:] == addition_norm[:size]:
            return " ".join(addition_words[size:])
    return " ".join(addition_words)


def join_words(existing: str, addition: str) -> str:
    existing = existing.strip()
    addition = addition.strip()
    if existing and addition:
        return f"{existing} {addition}"
    return existing or addition
