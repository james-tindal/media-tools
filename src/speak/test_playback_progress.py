"""Duration prediction and whole-document playback progress."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from playback_progress import PlaybackProgress, predict_durations
import synthesise_audio as playback


class ProgressTests(unittest.TestCase):
    def test_progress_weights_duration_instead_of_chunk_count(self):
        progress = PlaybackProgress([1.0, 5.0, 2.0], 0)
        progress.elapsed = 1
        self.assertEqual(progress.fraction(), 1 / 8)
        progress.chunk_index = 1
        progress.elapsed = 2
        self.assertEqual(progress.fraction(), 3 / 8)
        progress.chunk_index = 2
        progress.elapsed = 2
        self.assertEqual(progress.fraction(), 1)

    def test_actual_duration_replaces_prediction_in_numerator_and_total(self):
        progress = PlaybackProgress([10.0, 20.0], 0, chunk_index=1, elapsed=5)
        self.assertEqual(progress.fraction(), 0.5)
        progress.durations[0] = 5
        self.assertEqual(progress.fraction(), 10 / 25)

    def test_silence_progress_is_continuous_between_chunks(self):
        progress = PlaybackProgress([1.0, 5.0], 2, elapsed=1)
        self.assertEqual(progress.fraction(), 1 / 8)
        progress.chunk_index = 1
        progress.in_silence = True
        progress.elapsed = 0
        self.assertEqual(progress.fraction(), 1 / 8)
        progress.elapsed = 1
        self.assertEqual(progress.fraction(), 2 / 8)
        progress.elapsed = 2
        self.assertEqual(progress.fraction(), 3 / 8)
        progress.in_silence = False
        progress.elapsed = 0
        self.assertEqual(progress.fraction(), 3 / 8)
        progress.elapsed = 5
        self.assertEqual(progress.fraction(), 1)

    def test_start_chunk_retains_whole_document_position(self):
        progress = PlaybackProgress([1.0, 5.0, 2.0], 1, chunk_index=2)
        self.assertEqual(progress.fraction(), 8 / 10)

    def test_single_chunk_and_elapsed_clamping(self):
        progress = PlaybackProgress([2.0], 1)
        progress.elapsed = -1
        self.assertEqual(progress.fraction(), 0)
        progress.elapsed = 3
        self.assertEqual(progress.fraction(), 1)

    def test_predictor_uses_voice_language_speed_and_internal_chunking(self):
        model = SimpleNamespace(text_processor=Mock(return_value=("ids", "mask")),
                                dp_ort=Mock())
        # ONNX returns a list of output arrays.
        model.dp_ort.run.side_effect = [[np.array([4.0])], [np.array([6.0])], [np.array([2.0])]]
        tts = SimpleNamespace(model=model, is_multilingual=True)
        voice = SimpleNamespace(dp="voice style")
        args = SimpleNamespace(lang="en", speed=2)
        with patch("playback_progress.chunk_text", side_effect=[["a", "b"], ["c"]]):
            self.assertEqual(predict_durations(["first", "second"], tts, voice, args), [5, 1])
        model.text_processor.assert_any_call(["a"], "en")
        model.dp_ort.run.assert_called_with(None, {
            "text_ids": "ids", "style_dp": "voice style", "text_mask": "mask",
        })

    def test_english_model_omits_language_tokens_and_rejects_invalid_prediction(self):
        model = SimpleNamespace(text_processor=Mock(return_value=("ids", "mask")),
                                dp_ort=SimpleNamespace(run=Mock(return_value=[np.array([float("nan")])])))
        tts = SimpleNamespace(model=model, is_multilingual=False)
        with self.assertRaisesRegex(ValueError, "invalid chunk duration"):
            predict_durations(["hello"], tts, SimpleNamespace(dp="voice"),
                              SimpleNamespace(lang=None, speed=1))
        model.text_processor.assert_called_once_with(["hello"], None)

    def test_producer_records_actual_audio_duration_with_start_offset(self):
        progress = PlaybackProgress([10.0, 20.0, 30.0], 0)
        audio = playback.AudioQueue()
        tts = SimpleNamespace(sample_rate=100, synthesize=Mock(return_value=(np.ones(250), None)))
        args = SimpleNamespace(lang=None, speed=1.05, steps=8)
        playback.produce_chunks(["second"], audio, tts, "voice", args, False, progress, 1)
        self.assertEqual(progress.durations, [10, 2.5, 30])
        self.assertEqual(len(audio.get()), 250)
        self.assertIsNone(audio.get())

    def test_display_uses_weighted_progress(self):
        progress = PlaybackProgress([1.0, 5.0, 2.0], 0, chunk_index=1)
        controls = playback.PlaybackControls(chunk=2, total_chunks=3, progress=progress)
        with patch.object(playback.select, "select", return_value=([], [], [])), \
                patch("builtins.print") as output:
            playback.update_playback(7, controls, playback.AudioQueue(), 100)
        self.assertIn("12%", output.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
