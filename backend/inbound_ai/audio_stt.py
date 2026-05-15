"""Server-side audio upload and local speech-to-text helpers."""

from __future__ import annotations

import base64
import json
import re
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from .asterisk_agi import clean_transcript, render_command, run_command, safe_basename


@dataclass
class TranscriptionResult:
    transcript: str
    audio_file: str
    configured: bool
    error: str = ""


def save_audio_upload(
    audio_base64: str,
    mime_type: str,
    upload_dir: str,
    basename: str,
) -> str:
    if not audio_base64:
        raise ValueError("audio_base64 is required")
    if "," in audio_base64 and audio_base64.startswith("data:"):
        audio_base64 = audio_base64.split(",", 1)[1]

    raw = base64.b64decode(audio_base64, validate=True)
    if not raw:
        raise ValueError("audio payload is empty")

    path = Path(upload_dir)
    path.mkdir(parents=True, exist_ok=True)
    extension = extension_for_mime(mime_type)
    audio_path = path / (safe_basename(basename) + extension)
    audio_path.write_bytes(raw)
    return str(audio_path)


def extension_for_mime(mime_type: str) -> str:
    lowered = mime_type.lower()
    if "wav" in lowered or "wave" in lowered:
        return ".wav"
    if "webm" in lowered:
        return ".webm"
    if "ogg" in lowered:
        return ".ogg"
    if "mpeg" in lowered or "mp3" in lowered:
        return ".mp3"
    return ".audio"


def transcribe_with_command(audio_file: str, stt_command: str) -> TranscriptionResult:
    if not stt_command.strip():
        return TranscriptionResult(
            transcript="",
            audio_file=audio_file,
            configured=False,
            error="STT_COMMAND is not configured",
        )

    text_output = str(Path(audio_file).with_suffix(".txt"))
    command = render_command(
        stt_command,
        {
            "audio": audio_file,
            "text_output": text_output,
            "text_output_base": text_output[:-4],
        },
    )
    try:
        completed = run_command(command, timeout=120)
    except Exception as exc:
        return TranscriptionResult(
            transcript="",
            audio_file=audio_file,
            configured=True,
            error=str(exc),
        )

    if Path(text_output).exists():
        text = Path(text_output).read_text(encoding="utf-8")
    else:
        text = completed.stdout
    return TranscriptionResult(
        transcript=clean_transcript(strip_whisper_noise(text)),
        audio_file=audio_file,
        configured=True,
    )


def transcribe_with_server(
    audio_file: str,
    stt_server_url: str,
    language: str = "es",
    prompt: str = "",
    audio_ctx: int = 0,
) -> TranscriptionResult:
    if not stt_server_url.strip():
        return TranscriptionResult(
            transcript="",
            audio_file=audio_file,
            configured=False,
            error="STT_SERVER_URL is not configured",
        )
    if _is_whisper_cpp_pcm_url(stt_server_url):
        return transcribe_with_pcm_server(
            audio_file,
            stt_server_url,
            language,
            prompt,
            audio_ctx,
        )

    boundary = "----ai-inbound-" + uuid.uuid4().hex
    whisper_cpp_server = _is_whisper_cpp_inference_url(stt_server_url)
    fields = (
        {
            "language": language,
            "temperature": "0",
            "temperature_inc": "0",
            "response_format": "json",
            "no_timestamps": "true",
            "beam_size": "1",
            "best_of": "1",
            "max_context": "0",
            "max_len": "40",
            "prompt": prompt[-300:],
        }
        if whisper_cpp_server
        else {
            "model": "local",
            "language": language,
            "temperature": "0",
            "response_format": "json",
            "stream": "false",
            "prompt": prompt[-300:],
        }
    )
    if whisper_cpp_server and audio_ctx > 0:
        fields["audio_ctx"] = str(audio_ctx)
    body = _multipart_body(
        boundary,
        fields=fields,
        file_field="file",
        file_path=audio_file,
    )
    request = Request(
        _transcription_url(stt_server_url),
        data=body,
        headers={
            "Content-Type": "multipart/form-data; boundary=" + boundary,
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except (HTTPError, URLError, OSError) as exc:
        return TranscriptionResult(
            transcript="",
            audio_file=audio_file,
            configured=True,
            error=str(exc),
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"text": raw}
    return TranscriptionResult(
        transcript=clean_transcript(strip_whisper_noise(_server_text(payload))),
        audio_file=audio_file,
        configured=True,
    )


def transcribe_with_pcm_server(
    audio_file: str,
    stt_server_url: str,
    language: str = "es",
    prompt: str = "",
    audio_ctx: int = 0,
) -> TranscriptionResult:
    try:
        with wave.open(audio_file, "rb") as wav:
            if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                return TranscriptionResult(
                    transcript="",
                    audio_file=audio_file,
                    configured=True,
                    error="PCM endpoint requires 16 kHz mono signed 16-bit WAV input",
                )
            body = wav.readframes(wav.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        return TranscriptionResult(
            transcript="",
            audio_file=audio_file,
            configured=True,
            error=str(exc),
        )

    fields = {
        "language": language,
        "temperature": "0",
        "temperature_inc": "0",
        "response_format": "json",
        "no_timestamps": "true",
        "beam_size": "1",
        "best_of": "1",
        "max_context": "0",
        "max_len": "40",
        "prompt": prompt[-300:],
    }
    if audio_ctx > 0:
        fields["audio_ctx"] = str(audio_ctx)
    query = urlencode({key: value for key, value in fields.items() if value != ""})
    url = _transcription_url(stt_server_url)
    if query:
        url += ("&" if "?" in url else "?") + query
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except (HTTPError, URLError, OSError) as exc:
        return TranscriptionResult(
            transcript="",
            audio_file=audio_file,
            configured=True,
            error=str(exc),
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"text": raw}
    return TranscriptionResult(
        transcript=clean_transcript(strip_whisper_noise(_server_text(payload))),
        audio_file=audio_file,
        configured=True,
    )


def transcribe_audio(
    audio_file: str,
    stt_command: str,
    stt_server_url: str = "",
    language: str = "es",
    prompt: str = "",
) -> TranscriptionResult:
    if stt_server_url.strip():
        audio_ctx = whisper_cpp_audio_ctx_for_file(audio_file, stt_server_url)
        duration_seconds = wav_duration_seconds(audio_file) if audio_ctx > 0 else 0.0
        result = transcribe_with_server(
            audio_file,
            stt_server_url,
            language,
            prompt,
            audio_ctx=audio_ctx,
        )
        if audio_ctx > 0 and _should_retry_full_context(
            result.transcript,
            duration_seconds,
        ):
            result = transcribe_with_server(audio_file, stt_server_url, language, prompt)
        if not result.error or not stt_command.strip():
            return result
    return transcribe_with_command(audio_file, stt_command)


def whisper_cpp_audio_ctx_for_file(audio_file: str, stt_server_url: str) -> int:
    if not _is_whisper_cpp_url(stt_server_url):
        return 0
    duration = wav_duration_seconds(audio_file)
    if duration < 1.0:
        return 0
    if duration <= 4.0:
        return 256
    if duration <= 8.0:
        return 1024
    return 0


def wav_duration_seconds(audio_file: str) -> float:
    try:
        with wave.open(audio_file, "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        return 0.0


def _looks_degenerate_transcript(text: str) -> bool:
    if re.search(r"(\w{3,})\1{3,}", text.lower()):
        return True
    words = re.findall(r"[\wáéíóúüñ]+", text.lower())
    if len(words) < 8:
        return False
    repeated = 1
    for index in range(1, len(words)):
        if words[index] == words[index - 1]:
            repeated += 1
            if repeated >= 4:
                return True
        else:
            repeated = 1
    joined = " ".join(words)
    for size in range(3, 12):
        for start in range(0, max(0, len(words) - size * 3 + 1)):
            phrase = " ".join(words[start : start + size])
            if phrase and joined.count(phrase) >= 4:
                return True
    return False


def _should_retry_full_context(text: str, duration_seconds: float) -> bool:
    words = re.findall(r"[\wáéíóúüñ]+", text.lower())
    if duration_seconds >= 2.0 and len(words) <= 1:
        return True
    if duration_seconds >= 3.0 and len(words) <= 2:
        return True
    return _looks_degenerate_transcript(text)


def _transcription_url(stt_server_url: str) -> str:
    trimmed = stt_server_url.rstrip("/")
    if _is_whisper_cpp_url(trimmed):
        return trimmed
    if trimmed.endswith("/v1/audio/transcriptions"):
        return trimmed
    if trimmed.endswith("/v1"):
        return trimmed + "/audio/transcriptions"
    return urljoin(trimmed + "/", "v1/audio/transcriptions")


def _is_whisper_cpp_inference_url(stt_server_url: str) -> bool:
    path = urlparse(stt_server_url.strip()).path.rstrip("/")
    return path.endswith("/inference")


def _is_whisper_cpp_pcm_url(stt_server_url: str) -> bool:
    path = urlparse(stt_server_url.strip()).path.rstrip("/")
    return path.endswith("/inference-pcm")


def _is_whisper_cpp_url(stt_server_url: str) -> bool:
    return _is_whisper_cpp_inference_url(stt_server_url) or _is_whisper_cpp_pcm_url(
        stt_server_url
    )


def _multipart_body(
    boundary: str,
    fields: Dict[str, str],
    file_field: str,
    file_path: str,
) -> bytes:
    chunks = []
    for name, value in fields.items():
        if value == "":
            continue
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(
                    "utf-8"
                ),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    filename = Path(file_path).name
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: audio/wav\r\n\r\n",
            Path(file_path).read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks)


def _server_text(payload: Dict[str, object]) -> str:
    text = payload.get("text")
    if isinstance(text, str):
        return text
    segments = payload.get("segments")
    if isinstance(segments, list):
        parts = []
        for segment in segments:
            if isinstance(segment, dict) and isinstance(segment.get("text"), str):
                parts.append(segment["text"])
        return " ".join(parts)
    return ""


def strip_whisper_noise(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if line.lower().startswith("whisper"):
            continue
        cleaned.append(line)
    return " ".join(cleaned)
