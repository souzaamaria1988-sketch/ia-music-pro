"""Testes para real_instruments."""
import unittest
from real_instruments import (
    INSTRUMENTS, normalize_instrument_name, resolve_instrument_alias,
    list_families, BRAZILIAN_FALLBACK_ONLY
)


class TestInstrumentNormalization(unittest.TestCase):
    def test_normalize_basic(self):
        self.assertEqual(normalize_instrument_name("piano"), "piano")
    def test_normalize_accent(self):
        self.assertEqual(normalize_instrument_name("violão"), "violao")
    def test_normalize_uppercase(self):
        self.assertEqual(normalize_instrument_name("PIANO"), "piano")
    def test_normalize_hyphen(self):
        self.assertEqual(normalize_instrument_name("reco-reco"), "reco reco")
    def test_normalize_plural(self):
        self.assertEqual(normalize_instrument_name("pianos"), "piano")
    def test_normalize_spaces(self):
        self.assertEqual(normalize_instrument_name("  piano  "), "piano")


class TestInstrumentResolution(unittest.TestCase):
    def test_resolve_direct(self):
        self.assertEqual(resolve_instrument_alias("piano"), "piano")
    def test_resolve_portuguese(self):
        self.assertEqual(resolve_instrument_alias("violão"), "acoustic_guitar")
    def test_resolve_alias(self):
        self.assertEqual(resolve_instrument_alias("sanfona"), "accordion")
    def test_resolve_nonexistent(self):
        self.assertIsNone(resolve_instrument_alias("instrumento_inexistente"))
    def test_resolve_drum_kit(self):
        self.assertEqual(resolve_instrument_alias("bumbo"), "kick")


class TestInstrumentCatalog(unittest.TestCase):
    def test_catalog_not_empty(self):
        self.assertGreater(len(INSTRUMENTS), 0)
    def test_required_instruments_present(self):
        required = ["piano", "violin", "acoustic_guitar", "pandeiro", "cavaquinho"]
        for inst in required:
            self.assertIn(inst, INSTRUMENTS)
    def test_brazilian_fallback(self):
        for inst in BRAZILIAN_FALLBACK_ONLY:
            self.assertIn(inst, INSTRUMENTS)
            self.assertFalse(INSTRUMENTS[inst]["realistic"])
    def test_families_not_empty(self):
        families = list_families()
        self.assertGreater(len(families), 0)
    def test_percussion_instruments(self):
        for name, data in INSTRUMENTS.items():
            if data["is_percussion"]:
                self.assertEqual(data["channel"], 9)


if __name__ == "__main__":
    unittest.main()
