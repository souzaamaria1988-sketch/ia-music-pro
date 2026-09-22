"""Testes para soundfont_renderer."""
import unittest
import tempfile
from pathlib import Path
from soundfont_renderer import SoundFontRenderer


class TestSoundFontRenderer(unittest.TestCase):
    def test_renderer_creation(self):
        renderer = SoundFontRenderer()
        self.assertIsNotNone(renderer)

    def test_missing_soundfont(self):
        renderer = SoundFontRenderer(soundfont_path="/inexistente.sf2")
        self.assertFalse(renderer.is_soundfont_available())

    def test_render_missing_soundfont(self):
        renderer = SoundFontRenderer(soundfont_path="/inexistente.sf2")
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = renderer.render_midi(tmp_path, "/tmp/test.wav")
            self.assertFalse(result)
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_status_structure(self):
        renderer = SoundFontRenderer()
        status = renderer.status()
        for key in ["fluidsynth_installed", "fluidsynth_binary",
                    "soundfont_path", "soundfont_exists", "ready"]:
            self.assertIn(key, status)


class TestSoundFontRendererFallback(unittest.TestCase):
    def test_fallback_message(self):
        renderer = SoundFontRenderer(soundfont_path=None)
        self.assertFalse(renderer.is_available())


if __name__ == "__main__":
    unittest.main()
