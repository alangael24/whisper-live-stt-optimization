# Whisper Live STT Optimization

Repositorio separado para aislar el trabajo de latencia y estabilidad del flujo
live STT con `whisper.cpp` + backend local.

## Estado actual

- Backend local: `127.0.0.1:8080`
- Whisper server local: `127.0.0.1:50061`
- Modelo: `ggml-large-v3-turbo.bin`
- Binario activo: `/Users/alan/whisper.cpp/build-metal-current/bin/whisper-server`
- Test page: `http://127.0.0.1:8080/test/transcribe`

Ultima medicion registrada:

```json
{
  "text": "Me llamo Alan y necesito reparar una fuga hoy.",
  "audio_ms": 2828,
  "since_start_ms": 3593,
  "after_audio_ms": 492,
  "inference_ms": 455,
  "delivery_ms": 490,
  "rtf": 0.161,
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
PUBLIC_DEMO_ONLY=false KEEPALIVE_ENABLED=false .venv/bin/python server.py
```

## Benchmark

Directo a Whisper:

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
Ran 36 tests in 0.424s
OK
```
