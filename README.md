# Whisper Live STT Optimization

Repositorio separado para aislar el trabajo de latencia y estabilidad del flujo
live STT con `whisper.cpp` + backend local.

## Estado actual

- Backend local: `127.0.0.1:8080`
- Whisper server local: `127.0.0.1:50061`
- Modelo: `ggml-large-v3-turbo.bin`
- Binario activo: `/Users/alan/whisper.cpp/build-metal-current/bin/whisper-server`
- Test page: `http://127.0.0.1:8080/test/transcribe`
- Endpoint rapido C++ PCM: `http://127.0.0.1:50061/inference-pcm`

Ultima medicion registrada:

```json
{
  "text": "Me llamo Alan y necesito reparar una fuga hoy.",
  "audio_ms": 2828,
  "since_start_ms": 3569,
  "after_audio_ms": 506,
  "inference_ms": 460,
  "delivery_ms": 504,
  "rtf": 0.163,
  "speculative": false
}
```

## Layout

```text
backend/
  inbound_ai/
    audio_stt.py
    live_stt.py
  static/
    transcribe-test.html
  tests/
    test_audio_stt.py

whisper.cpp/
  examples/server/server.cpp
  ggml/src/ggml-metal/ggml-metal-device.cpp
  ggml/src/ggml-metal/ggml-metal.metal
  src/coreml/whisper-encoder.mm

patches/
  ai-inbound-backend-whisper-live.patch
  whisper-cpp-metal-server.patch

scripts/
  bench_live_ws.py
```

## Aplicar patches

Backend:

```sh
git -C /Users/alan/ai-inbound-backend apply /Users/alan/Documents/Codex/2026-05-14/we-got-whisper-en-esta-computadora/whisper-live-stt-optimization/patches/ai-inbound-backend-whisper-live.patch
```

Whisper.cpp:

```sh
git -C /Users/alan/whisper.cpp apply /Users/alan/Documents/Codex/2026-05-14/we-got-whisper-en-esta-computadora/whisper-live-stt-optimization/patches/whisper-cpp-metal-server.patch
```

## Correr servidores

Whisper:

```sh
/Users/alan/whisper.cpp/build-metal-current/bin/whisper-server \
  --host 127.0.0.1 \
  --port 50061 \
  -m /Users/alan/ai-inbound-backend/models/ggml-large-v3-turbo.bin \
  -l es \
  -nt \
  -bs 1 \
  -bo 1 \
  -nf \
  -t 6
```

Backend:

```sh
cd /Users/alan/ai-inbound-backend
STT_SERVER_URL=http://127.0.0.1:50061/inference-pcm \
STT_FINAL_SERVER_URL=http://127.0.0.1:50061/inference-pcm \
PUBLIC_DEMO_ONLY=false \
KEEPALIVE_ENABLED=false \
.venv/bin/python server.py
```

## Benchmark

Directo a Whisper con multipart WAV:

```sh
curl -sS -w '\nTIME:%{time_total}\n' http://127.0.0.1:50061/inference \
  -H 'Expect:' \
  -F file=@/Users/alan/ai-inbound-backend/data/audio/TRANSCRIBE_TEST-986b6dc2.wav \
  -F language=es \
  -F temperature=0 \
  -F temperature_inc=0 \
  -F response_format=json \
  -F no_timestamps=true \
  -F beam_size=1 \
  -F best_of=1 \
  -F audio_ctx=256 \
  -F max_context=0 \
  -F max_len=40
```

Directo a Whisper con PCM crudo:

```sh
python -c 'import wave; src="/Users/alan/ai-inbound-backend/data/audio/TRANSCRIBE_TEST-986b6dc2.wav"; dst="/tmp/transcribe-test.pcm"; w=wave.open(src,"rb"); open(dst,"wb").write(w.readframes(w.getnframes()))'

curl -sS -w '\nTIME:%{time_total}\n' \
  'http://127.0.0.1:50061/inference-pcm?language=es&temperature=0&temperature_inc=0&response_format=json&no_timestamps=true&beam_size=1&best_of=1&audio_ctx=256&max_context=0&max_len=40' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @/tmp/transcribe-test.pcm
```

Flujo live WebSocket:

```sh
/Users/alan/ai-inbound-backend/.venv/bin/python scripts/bench_live_ws.py
```

## Tests

```sh
cd /Users/alan/ai-inbound-backend
.venv/bin/python -m unittest tests.test_audio_stt tests.test_voice_routes
```

Ultimo resultado:

```text
Ran 41 tests in 0.433s
OK
```
