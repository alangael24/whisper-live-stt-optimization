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
- Gate de encoder/decode: `scripts/bench_pcm_gate.py`

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

Ultimo gate directo a `/inference-pcm?timings=true` con `audio_ctx=256`,
20 runs y 2 warmups:

```text
mel_ms: p50=6.71 p95=7.71
encode_ms: p50=265.07 p95=272.74 gate<=70.00 FAIL
decode_ms: p50=107.23 p95=110.74 gate<=15.00 FAIL
sample_ms: p50=15.75 p95=17.23
batchd_ms: p50=8.68 p95=9.78
prompt_ms: p50=0.00 p95=0.00
postprocess_ms: p50=0.14 p95=0.32
overhead_ms: p50=33.85 p95=37.73 gate<=15.00 FAIL
e2e_ms: p50=407.50 p95=415.33 gate<=100.00 FAIL
client_e2e_ms: p50=408.86 p95=416.28
```

Experimento `GGML_METAL_MATMUL_N64=1` sobre el hot path `f16 x f32 -> f32`
del encoder. El kernel queda opt-in porque no sostuvo una mejora real frente
al path normal.

```text
normal, 5 runs: encode_ms p50=261.06 p95=279.68; e2e_ms p50=377.89 p95=396.55
N64,    5 runs: encode_ms p50=288.85 p95=296.77; e2e_ms p50=439.54 p95=460.08
```

Transcripcion de verificacion con el kernel experimental:

```text
Me llamo Alan y necesito reparar una fuga hoy.
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
  include/whisper.h
  src/coreml/whisper-encoder.mm
  src/whisper.cpp

patches/
  ai-inbound-backend-whisper-live.patch
  whisper-cpp-metal-server.patch

scripts/
  bench_live_ws.py
  bench_pcm_gate.py
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

`GGML_METAL_MATMUL_N64=1` activa el kernel experimental `64x64` para las
GEMM `f16 x f32 -> f32` alineadas del encoder. Se mantiene apagado por defecto
porque en esta maquina fue igual o peor que el kernel Metal upstream. Para
trazar que kernel toma cada `mul_mat`:

```sh
GGML_METAL_MATMUL_DEBUG=1 /Users/alan/whisper.cpp/build-metal-current/bin/whisper-server ...
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

Gate de encoder/decode/e2e usando timings internos del endpoint PCM:

```sh
/Users/alan/ai-inbound-backend/.venv/bin/python scripts/bench_pcm_gate.py --runs 20 --warmup 2
```

El gate falla si:

```text
encode_p95_ms > 70
decode_p95_ms > 15
overhead_p95_ms > 15
e2e_p95_ms > 100
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
