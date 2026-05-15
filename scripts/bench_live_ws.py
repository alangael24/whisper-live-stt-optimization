import base64
import json
import os
import socket
import struct
import time
import wave


HOST = "127.0.0.1"
PORT = 8080
PATH = "/api/transcribe/live"
AUDIO = "/Users/alan/ai-inbound-backend/data/audio/TRANSCRIBE_TEST-986b6dc2.wav"
FRAME_SAMPLES = 800


def recv_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError("socket closed")
        data.extend(chunk)
    return bytes(data)


def send_frame(sock, opcode, payload):
    payload = payload if isinstance(payload, bytes) else payload.encode("utf-8")
    key = os.urandom(4)
    first = 0x80 | opcode
    length = len(payload)
    if length < 126:
        header = struct.pack("!BB", first, 0x80 | length)
    elif length < (1 << 16):
        header = struct.pack("!BBH", first, 0x80 | 126, length)
    else:
        header = struct.pack("!BBQ", first, 0x80 | 127, length)
    masked = bytes(byte ^ key[i % 4] for i, byte in enumerate(payload))
    sock.sendall(header + key + masked)


def recv_frame(sock):
    b1, b2 = recv_exact(sock, 2)
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if length == 126:
        (length,) = struct.unpack("!H", recv_exact(sock, 2))
    elif length == 127:
        (length,) = struct.unpack("!Q", recv_exact(sock, 8))
    mask = recv_exact(sock, 4) if masked else b""
    payload = recv_exact(sock, length) if length else b""
    if masked:
        payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    return opcode, payload


def connect():
    sock = socket.create_connection((HOST, PORT), timeout=10)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET {PATH} HTTP/1.1\r\n"
        f"Host: {HOST}:{PORT}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )
    sock.sendall(request.encode("ascii"))
    response = b""
    while b"\r\n\r\n" not in response:
        response += sock.recv(4096)
    if b" 101 " not in response.split(b"\r\n", 1)[0]:
        raise RuntimeError(response.decode("latin1", errors="replace"))
    sock.settimeout(20)
    return sock


def main():
    with wave.open(AUDIO, "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if rate != 16000 or channels != 1 or width != 2:
        raise RuntimeError(f"unexpected WAV format: rate={rate}, channels={channels}, width={width}")

    frame_bytes = FRAME_SAMPLES * width
    audio_ms = int(len(frames) / width / rate * 1000)

    sock = connect()
    started = time.perf_counter()
    first_ready = None
    transcript = None
    try:
        while first_ready is None:
            opcode, payload = recv_frame(sock)
            if opcode == 1:
                event = json.loads(payload.decode("utf-8"))
                if event.get("type") == "ready":
                    first_ready = time.perf_counter()

        offset = 0
        while offset < len(frames):
            chunk = frames[offset : offset + frame_bytes]
            send_frame(sock, 2, chunk)
            offset += len(chunk)
            time.sleep((len(chunk) / width) / rate)
        audio_done = time.perf_counter()
        send_frame(sock, 1, json.dumps({"type": "stop"}))

        while transcript is None:
            opcode, payload = recv_frame(sock)
            if opcode == 8:
                break
            if opcode != 1:
                continue
            event = json.loads(payload.decode("utf-8"))
            if event.get("type") == "transcript":
                transcript = event
                received = time.perf_counter()
                break
    finally:
        sock.close()

    if not transcript:
        raise RuntimeError("no transcript received")

    result = {
        "text": transcript.get("hypothesis") or transcript.get("final_text"),
        "audio_ms": audio_ms,
        "since_start_ms": int((received - started) * 1000),
        "after_audio_ms": int((received - audio_done) * 1000),
        "inference_ms": transcript.get("inference_ms"),
        "delivery_ms": transcript.get("delivery_ms"),
        "rtf": transcript.get("rtf"),
        "speculative": transcript.get("speculative"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
