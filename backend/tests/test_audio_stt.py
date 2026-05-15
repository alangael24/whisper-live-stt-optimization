import base64
import tempfile
import unittest
import wave

from inbound_ai.audio_stt import (
    _looks_degenerate_transcript,
    extension_for_mime,
    save_audio_upload,
    transcribe_with_command,
    whisper_cpp_audio_ctx_for_file,
)
from inbound_ai.live_stt import PcmRingBuffer, SAMPLE_RATE, TranscriptState, trim_pcm_for_stt


class AudioSttTests(unittest.TestCase):
    def test_extension_for_mime(self):
        self.assertEqual(extension_for_mime("audio/wav"), ".wav")
        self.assertEqual(extension_for_mime("audio/webm;codecs=opus"), ".webm")

    def test_save_audio_upload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = base64.b64encode(b"RIFFdemo").decode("ascii")
            path = save_audio_upload(payload, "audio/wav", tmpdir, "call/1")
            self.assertTrue(path.endswith("call-1.wav"))

    def test_transcribe_without_command(self):
        result = transcribe_with_command("/tmp/test.wav", "")
        self.assertFalse(result.configured)
        self.assertEqual(result.transcript, "")

    def test_whisper_cpp_audio_ctx_for_short_precise_windows(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            with wave.open(tmp.name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * int(16000 * 2.8))

            audio_ctx = whisper_cpp_audio_ctx_for_file(
                tmp.name,
                "http://127.0.0.1:50061/inference",
            )

        self.assertEqual(audio_ctx, 256)

    def test_whisper_cpp_audio_ctx_uses_short_context_for_live_windows(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            with wave.open(tmp.name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * int(16000 * 1.5))

            audio_ctx = whisper_cpp_audio_ctx_for_file(
                tmp.name,
                "http://127.0.0.1:50061/inference",
            )

        self.assertEqual(audio_ctx, 256)

    def test_whisper_cpp_audio_ctx_skips_subsecond_audio(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            with wave.open(tmp.name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * int(16000 * 0.8))

            audio_ctx = whisper_cpp_audio_ctx_for_file(
                tmp.name,
                "http://127.0.0.1:50061/inference",
            )

        self.assertEqual(audio_ctx, 0)

    def test_degenerate_transcript_detection(self):
        self.assertTrue(
            _looks_degenerate_transcript(
                "me llamo Alan y necesito repar reparararararararararara "
                "reparararararararararara"
            )
        )
        self.assertFalse(
            _looks_degenerate_transcript(
                "Me llamo Alan y necesito reparar una fuga hoy."
            )
        )

    def test_live_state_commits_only_final_text(self):
        state = TranscriptState()
        final_text = state.commit_final("me llamo Alan")
        self.assertEqual(final_text, "me llamo Alan")

    def test_live_ring_does_not_duplicate_first_speech_chunk(self):
        ring = PcmRingBuffer(SAMPLE_RATE)
        silence = b"\0\0" * 800
        speech = (1000).to_bytes(2, "little", signed=True) * 800

        ring.append(silence)
        ring.append(speech)
        utterance = ring.snapshot_utterance()

        self.assertEqual(utterance, silence + speech)

    def test_live_ring_clears_pre_roll_after_final_utterance(self):
        ring = PcmRingBuffer(SAMPLE_RATE)
        silence = b"\0\0" * 800
        first_speech = (1000).to_bytes(2, "little", signed=True) * 800
        next_speech = (1200).to_bytes(2, "little", signed=True) * 800

        ring.append(silence)
        ring.append(first_speech)
        utterance = ring.extract_utterance_if_ready(0, force=True)
        ring.append(next_speech)

        self.assertEqual(utterance, silence + first_speech)
        self.assertEqual(ring.snapshot_utterance(), next_speech)

    def test_live_state_dedupes_accented_overlap(self):
        state = TranscriptState()
        state.commit_final("Me llamo Alan")

        final_text = state.commit_final("Me llamo Alán y necesito reparar")

        self.assertEqual(final_text, "Me llamo Alan y necesito reparar")

    def test_live_state_returns_empty_delta_for_duplicate(self):
        state = TranscriptState("Me llamo Alan")

        final_text, delta = state.commit_final_delta("Me llamo Alán")

        self.assertEqual(final_text, "Me llamo Alan")
        self.assertEqual(delta, "")

    def test_live_state_dedupes_confirmed_suffix_after_filler(self):
        state = TranscriptState("Me llamo Alan y necesito reparar")

        final_text, delta = state.commit_final_delta("ok necesito reparar una fuga hoy")

        self.assertEqual(final_text, "Me llamo Alan y necesito reparar una fuga hoy")
        self.assertEqual(delta, "una fuga hoy")

    def test_live_state_dedupes_ordered_overlap_with_missing_word(self):
        state = TranscriptState("Me llamo Alan y necesito reparar")

        final_text, delta = state.commit_final_delta(
            "Me llamo Alan necesito reparar una fuga"
        )

        self.assertEqual(final_text, "Me llamo Alan y necesito reparar una fuga")
        self.assertEqual(delta, "una fuga")

    def test_trim_pcm_for_stt_removes_outer_silence(self):
        silence_before = b"\0\0" * int(SAMPLE_RATE * 0.5)
        speech = (1000).to_bytes(2, "little", signed=True) * int(SAMPLE_RATE * 0.4)
        silence_after = b"\0\0" * int(SAMPLE_RATE * 0.6)
        pcm = silence_before + speech + silence_after

        trimmed = trim_pcm_for_stt(pcm)

        self.assertLess(len(trimmed), len(pcm))
        self.assertGreater(len(trimmed), len(speech))
        self.assertLess(len(trimmed), len(speech) + len(silence_before))

    def test_trim_pcm_for_stt_keeps_all_silence(self):
        pcm = b"\0\0" * int(SAMPLE_RATE * 0.5)

        self.assertEqual(trim_pcm_for_stt(pcm), pcm)


if __name__ == "__main__":
    unittest.main()
