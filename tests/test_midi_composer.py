"""Testes para midi_composer."""
import unittest
import tempfile
from pathlib import Path
from midi_composer import MidiComposer


class TestMidiComposer(unittest.TestCase):
    def setUp(self):
        self.composer = MidiComposer(bpm=120, key="C", scale="major")

    def test_create_composer(self):
        self.assertEqual(self.composer.bpm, 120)

    def test_add_track(self):
        idx = self.composer.add_track("piano")
        self.assertEqual(idx, 0)

    def test_add_invalid_track(self):
        with self.assertRaises(ValueError):
            self.composer.add_track("instrumento_invalido")

    def test_add_note(self):
        self.composer.add_track("piano")
        self.composer.add_note(0, start=0.0, duration=0.5, pitch=60, velocity=90)
        self.assertEqual(len(self.composer.tracks[0]["notes"]), 1)

    def test_add_invalid_pitch(self):
        self.composer.add_track("piano")
        with self.assertRaises(ValueError):
            self.composer.add_note(0, start=0.0, duration=0.5, pitch=200)

    def test_scale_pitches(self):
        self.assertEqual(len(self.composer.scale_pitches()), 7)

    def test_chord_progression(self):
        self.composer.add_track("piano")
        self.composer.add_chord_progression(0, bars=2)
        self.assertGreater(len(self.composer.tracks[0]["notes"]), 0)

    def test_to_dict(self):
        self.composer.add_track("piano")
        data = self.composer.to_dict()
        self.assertEqual(data["bpm"], 120)

    def test_save_json(self):
        self.composer.add_track("piano")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.composer.save_json(tmp_path)
            self.assertTrue(Path(tmp_path).is_file())
        finally:
            Path(tmp_path).unlink()

    def test_save_midi(self):
        self.composer.add_track("piano")
        self.composer.add_note(0, start=0.0, duration=0.5, pitch=60)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = self.composer.save_midi(tmp_path)
            if result:
                self.assertTrue(Path(tmp_path).is_file())
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()


if __name__ == "__main__":
    unittest.main()
