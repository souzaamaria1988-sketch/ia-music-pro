"""Compositor de MIDI independente da renderização de áudio."""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

try:
    from mido import Message, MetaMessage, MidiFile, MidiTrack
    from mido import bpm2tempo
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False

from real_instruments import INSTRUMENTS


class MidiComposer:
    """Compõe estruturas MIDI e salva em .mid."""

    SCALES = {
        "major":      [0, 2, 4, 5, 7, 9, 11],
        "minor":      [0, 2, 3, 5, 7, 8, 10],
        "pentatonic": [0, 3, 5, 7, 10],
        "blues":      [0, 3, 5, 6, 7, 10],
        "dorian":     [0, 2, 3, 5, 7, 9, 10],
    }
    KEYS = {
        "C": 0, "G": 7, "D": 2, "A": 9, "E": 4, "B": 11,
        "F": 5, "Bb": 10, "Eb": 3, "Ab": 8, "Db": 1, "F#": 6,
    }

    def __init__(self, title: str = "Música gerada", bpm: int = 120,
                 key: str = "C", scale: str = "major", style: str = "generic"):
        self.title = title
        self.bpm = bpm
        self.key = key
        self.scale = scale
        self.style = style
        self.tracks: List[Dict[str, Any]] = []

    def add_track(self, instrument_id: str, name=None) -> int:
        if instrument_id not in INSTRUMENTS:
            raise ValueError(f"Instrumento desconhecido: {instrument_id}")
        idx = len(self.tracks)
        self.tracks.append({
            "index": idx,
            "instrument": instrument_id,
            "name": name or INSTRUMENTS[instrument_id]["display_name"],
            "notes": [],
        })
        return idx

    def add_note(self, track_idx: int, *, start: float, duration: float,
                 pitch: int, velocity: int = 90) -> None:
        if track_idx < 0 or track_idx >= len(self.tracks):
            raise IndexError("track_idx fora do intervalo")
        if not (0 <= pitch <= 127):
            raise ValueError(f"pitch inválido: {pitch}")
        if not (1 <= velocity <= 127):
            raise ValueError(f"velocity inválido: {velocity}")
        if duration <= 0:
            raise ValueError("duração deve ser positiva")
        self.tracks[track_idx]["notes"].append({
            "start": float(start),
            "duration": float(duration),
            "pitch": int(pitch),
            "velocity": int(velocity),
            "instrument": self.tracks[track_idx]["instrument"],
        })

    def scale_pitches(self, root_octave: int = 4) -> list:
        root = self.KEYS.get(self.key, 0)
        intervals = self.SCALES.get(self.scale, self.SCALES["major"])
        base = 12 * root_octave + root
        return [base + i for i in intervals]

    def add_chord_progression(self, track_idx: int, *, bars: int = 8,
                              beats_per_bar: int = 4, progressions=None):
        pitches = self.scale_pitches(root_octave=4)
        prog = progressions or [[0, 2, 4], [5, 0, 2], [3, 5, 0], [4, 0, 2]]
        beat = 60.0 / self.bpm
        for bar in range(bars):
            chord_idx = prog[bar % len(prog)]
            chord = [pitches[i % len(pitches)] + (12 if i >= len(pitches) else 0)
                     for i in chord_idx]
            for beat_in_bar in range(beats_per_bar):
                t = (bar * beats_per_bar + beat_in_bar) * beat
                for p in chord:
                    self.add_note(track_idx, start=t, duration=beat,
                                  pitch=p % 128, velocity=80)

    def add_drum_pattern(self, track_idx: int, *, bars: int = 8,
                         beats_per_bar: int = 4, style: str = "rock"):
        beat = 60.0 / self.bpm
        patterns = {
            "rock":     {"kick": [0.0, 2.0], "snare": [1.0, 3.0],
                         "hihat_closed": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]},
            "samba":    {"kick": [0.0, 2.0], "snare": [0.75, 2.75],
                         "hihat_closed": [i * 0.5 for i in range(8)]},
            "electronic": {"kick": [0.0, 1.0, 2.0, 3.0],
                           "hihat_closed": [i * 0.25 for i in range(16)]},
        }
        pat = patterns.get(style, patterns["rock"])
        for bar in range(bars):
            base = bar * beats_per_bar * beat
            for drum_id, positions in pat.items():
                if drum_id not in INSTRUMENTS:
                    continue
                for pos in positions:
                    t = base + pos * beat
                    note = INSTRUMENTS[drum_id].get("midi_note")
                    if note is None:
                        continue
                    self.add_note(track_idx, start=t, duration=0.1,
                                  pitch=note, velocity=100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "style": self.style,
            "bpm": self.bpm,
            "key": self.key,
            "scale": self.scale,
            "tracks": [
                {"name": t["name"], "instrument": t["instrument"],
                 "note_count": len(t["notes"])} for t in self.tracks
            ],
        }

    def save_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    def save_midi(self, path) -> bool:
        if not HAS_MIDO:
            return False
        mid = MidiFile(ticks_per_beat=480)
        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(MetaMessage("track_name", name="Tempo", time=0))
        tempo_track.append(MetaMessage("set_tempo", tempo=bpm2tempo(self.bpm), time=0))
        tempo_track.append(MetaMessage("end_of_track", time=0))

        beat = 60.0 / self.bpm
        tpb = mid.ticks_per_beat

        for tdata in self.tracks:
            track = MidiTrack()
            mid.tracks.append(track)
            inst = INSTRUMENTS[tdata["instrument"]]
            track.append(MetaMessage("track_name", name=tdata["name"], time=0))
            if not inst["is_percussion"] and "midi_program" in inst:
                track.append(Message("program_change", channel=inst["channel"],
                                     program=inst["midi_program"], time=0))

            events = []
            for n in tdata["notes"]:
                start_tick = int(n["start"] / beat * tpb)
                end_tick = int((n["start"] + n["duration"]) / beat * tpb)
                ch = inst["channel"]
                events.append((start_tick, "on", n["pitch"], n["velocity"], ch))
                events.append((end_tick, "off", n["pitch"], 0, ch))
            events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))

            last_tick = 0
            for tick, kind, pitch, vel, ch in events:
                dt = tick - last_tick
                track.append(Message("note_on" if kind == "on" else "note_off",
                                     channel=ch, note=pitch, velocity=vel, time=dt))
                last_tick = tick
            track.append(MetaMessage("end_of_track", time=0))

        mid.save(str(path))
        return True
