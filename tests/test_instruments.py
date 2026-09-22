import unittest
from real_instruments import INSTRUMENTS, normalize_instrument_name, resolve_instrument_alias


class TestInstruments(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_instrument_name("Violão"), "violao")

    def test_resolve(self):
        self.assertEqual(resolve_instrument_alias("sanfona"), "accordion")

    def test_not_found(self):
        self.assertIsNone(resolve_instrument_alias("instrumento_inexistente"))


if __name__ == "__main__":
    unittest.main()
