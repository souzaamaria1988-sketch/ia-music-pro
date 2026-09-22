"""Renderiza MIDI para WAV via FluidSynth + SoundFont."""
from __future__ import annotations
import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class SoundFontRenderer:
    def __init__(self, soundfont_path=None, sample_rate: int = 44100, gain: float = 0.8):
        self.soundfont_path = Path(soundfont_path) if soundfont_path else None
        self.sample_rate = sample_rate
        self.gain = gain
        self._fluidsynth_bin = shutil.which("fluidsynth")

    def is_fluidsynth_installed(self) -> bool:
        return self._fluidsynth_bin is not None

    def is_soundfont_available(self) -> bool:
        return bool(self.soundfont_path) and Path(self.soundfont_path).is_file()

    def is_available(self) -> bool:
        return self.is_fluidsynth_installed() and self.is_soundfont_available()

    def status(self) -> dict:
        return {
            "fluidsynth_installed": self.is_fluidsynth_installed(),
            "fluidsynth_binary": self._fluidsynth_bin,
            "soundfont_path": str(self.soundfont_path) if self.soundfont_path else None,
            "soundfont_exists": self.is_soundfont_available(),
            "ready": self.is_available(),
        }

    def render_midi(self, midi_path, wav_path) -> bool:
        if not self.is_available():
            missing = []
            if not self.is_fluidsynth_installed():
                missing.append("fluidsynth (binário não encontrado no PATH)")
            if not self.is_soundfont_available():
                missing.append(f"SoundFont em '{self.soundfont_path}'")
            log.warning("SoundFont indisponível: %s. Ativando fallback sintético.", "; ".join(missing))
            return False

        cmd = [
            self._fluidsynth_bin, "-ni",
            "-r", str(self.sample_rate),
            "-g", str(self.gain),
            str(self.soundfont_path),
            str(midi_path),
            "-F", str(wav_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as e:
            log.error("Falha ao executar FluidSynth: %s", e)
            return False

        if result.returncode != 0 or not Path(wav_path).is_file():
            log.error("FluidSynth retornou código %s: %s", result.returncode, result.stderr[:500])
            return False

        log.info("WAV renderizado com sucesso: %s", wav_path)
        return True

    def render_notes(self, notes, wav_path, instrument_name: str = "piano", bpm: int = 120) -> bool:
        from midi_composer import MidiComposer
        comp = MidiComposer(bpm=bpm)
        comp.add_track(instrument_name)
        for n in notes:
            comp.add_note(0, start=n["start"], duration=n["duration"],
                          pitch=n["pitch"], velocity=n.get("velocity", 90))
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp_path = tmp.name
        if not comp.save_midi(tmp_path):
            return False
        try:
            return self.render_midi(tmp_path, wav_path)
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
