#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
music_generator.py — Gerador principal do IA Music Pro (v9).

v9 — OS SEUS SOUNDFONTS COMO SOM PRINCIPAL + ANTI-REPETIÇÃO:
    1. SOUNDFONT ENGINE: os 15 bancos de soundfonts/ são a fonte primária de
       timbre. Cada trilha é renderizada com o banco certo (lead ← Juno106/
       Pro53/FlangerSaw/TJ, bass ← Moog/Bandpass, pads ← HS/UK, drums ←
       JD Rockset/Giant/Darbuka, cordas ← Crunk String). Pareamento por
       NOME PARECIDO + SOUNDFONT_CORE (editável) + notas aprendidas.
    2. AUDIÇÃO/TREINO DOS BANCOS (--audition): renderiza sondas cromáticas e
       de bateria com CADA banco, mede RMS/brilho/graves, detecta bancos
       mudos e salva o aprendizado em memory/sf2_scores.json — o gerador
       passa a preferir os bancos que soam melhor em cada papel.
    3. ROTAÇÃO: dentro de cada papel os melhores bancos alternam entre
       gerações → os 15 bancos são usados, não só um.
    4. ANTI-REPETIÇÃO: modulação de tom por seção (relativa na ponte, +2
       semitons no refrão final — "truck driver"), melodia com
       DESENVOLVIMENTO temático (inversão/retrogrado/transposição do motivo),
       breakdown antes do refrão final, dropout de bateria, arco dinâmico
       crescente ao longo da música e fade no outro.
    5. --gm-instruments: usa timbres GM para acústicos (piano/órgão) se você
       preferir o som acústico; padrão = seus bancos para tudo que eles
       cobrem (conforme pedido).

v8: render por trilha, --reference, --render-midi. v7: FluidSynth CLI + IA
    (knowledge_base, personalidade, feedback). v6.1: auditoria com offset.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as _np
except ImportError:
    _np = None

try:
    from real_instruments import INSTRUMENTS
    _RI_ERR: Optional[str] = None
except ImportError as _exc:
    _RI_ERR = str(_exc)
    INSTRUMENTS = {
        "piano":  {"display_name": "Piano", "channel": 0, "midi_program": 0,
                   "is_percussion": False},
        "violin": {"display_name": "Violino", "channel": 7, "midi_program": 40,
                   "is_percussion": False},
        "cello":  {"display_name": "Violoncelo", "channel": 8, "midi_program": 42,
                   "is_percussion": False},
    }

_ESSENTIAL_INSTRUMENTS: Dict[str, Dict[str, Any]] = {
    "bass":            {"display_name": "Baixo", "channel": 3, "midi_program": 33,
                        "is_percussion": False, "aliases": ["baixo", "contrabaixo"]},
    "acoustic_guitar": {"display_name": "Violão", "channel": 1, "midi_program": 24,
                        "is_percussion": False, "aliases": ["violao"]},
    "electric_guitar": {"display_name": "Guitarra", "channel": 2, "midi_program": 29,
                        "is_percussion": False, "aliases": ["guitarra"]},
    "electric_piano":  {"display_name": "Piano Elétrico", "channel": 5, "midi_program": 4,
                        "is_percussion": False, "aliases": ["rhodes"]},
    "organ":           {"display_name": "Órgão", "channel": 4, "midi_program": 16,
                        "is_percussion": False, "aliases": ["orgao"]},
    "strings":         {"display_name": "Cordas", "channel": 6, "midi_program": 48,
                        "is_percussion": False, "aliases": ["cordas"]},
    "accordion":       {"display_name": "Sanfona", "channel": 7, "midi_program": 21,
                        "is_percussion": False, "aliases": ["acordeon", "acordeao"]},
    "cavaquinho":      {"display_name": "Cavaquinho", "channel": 8, "midi_program": 105,
                        "is_percussion": False, "aliases": ["cavaco"]},
    "banjo":           {"display_name": "Banjo", "channel": 14, "midi_program": 105,
                        "is_percussion": False},
    "synth_pad":       {"display_name": "Synth Pad", "channel": 12, "midi_program": 89,
                        "is_percussion": False, "aliases": ["pad"]},
    "synth_lead":      {"display_name": "Synth Lead", "channel": 13, "midi_program": 81,
                        "is_percussion": False, "aliases": ["sintetizador"]},
    "flute":           {"display_name": "Flauta", "channel": 10, "midi_program": 73,
                        "is_percussion": False, "aliases": ["flauta"]},
    "trumpet":         {"display_name": "Trompete", "channel": 11, "midi_program": 56,
                        "is_percussion": False, "aliases": ["trompete"]},
    "saxophone":       {"display_name": "Saxofone", "channel": 15, "midi_program": 65,
                        "is_percussion": False, "aliases": ["sax", "saxofone"]},
    "drums":     {"display_name": "Bateria", "channel": 9, "is_percussion": True,
                  "aliases": ["bateria", "tambor"]},
    "pandeiro":  {"display_name": "Pandeiro", "channel": 9, "is_percussion": True},
    "surdo":     {"display_name": "Surdo", "channel": 9, "is_percussion": True,
                  "aliases": ["zabumba"]},
    "tamborim":  {"display_name": "Tamborim", "channel": 9, "is_percussion": True},
    "agogo":     {"display_name": "Agogô", "channel": 9, "is_percussion": True},
    "cuica":     {"display_name": "Cuíca", "channel": 9, "is_percussion": True},
    "reco_reco": {"display_name": "Reco-reco", "channel": 9, "is_percussion": True,
                  "aliases": ["reco"]},
    "chocalho":  {"display_name": "Chocalho", "channel": 9, "is_percussion": True,
                  "aliases": ["ganza", "maracas"]},
    "berimbau":  {"display_name": "Berimbau", "channel": 9, "is_percussion": True},
    "conga":     {"display_name": "Conga", "channel": 9, "is_percussion": True,
                  "aliases": ["congas"]},
    "shaker":    {"display_name": "Shaker", "channel": 9, "is_percussion": True},
}
for _id, _entry in _ESSENTIAL_INSTRUMENTS.items():
    INSTRUMENTS.setdefault(_id, _entry)
INSTRUMENTS.setdefault("square_lead",
    {"display_name": "Square Lead", "channel": 0, "midi_program": 80,
     "is_percussion": False, "aliases": ["chiptune lead", "8bit lead"]})
INSTRUMENTS.setdefault("synth_bass",
    {"display_name": "Synth Bass", "channel": 3, "midi_program": 38,
     "is_percussion": False, "aliases": ["reese", "808 bass"]})

try:
    from midi_composer import MidiComposer
except ImportError as _exc2:
    MidiComposer = None
    _MIDI_COMPOSER_ERR = str(_exc2)

_RENDERER = None
try:
    import soundfont_renderer as _sr
    for _n in ("render_midi_to_wav", "render", "render_wav", "midi_to_wav", "render_audio"):
        if hasattr(_sr, _n):
            _RENDERER = getattr(_sr, _n)
            break
except ImportError:
    pass

LOG = logging.getLogger("ia_music_pro.generator")
_GM_INSTRUMENTS = False  # --gm-instruments: acústicos com timbre GM


def _setup_logging(verbose: bool = False) -> logging.Logger:
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not LOG.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        LOG.addHandler(h)
    LOG.propagate = False
    return LOG


# =============================================================================
# PERCUSSÃO GM
# =============================================================================

GM: Dict[str, int] = {
    "kick": 36, "snare": 38, "rim": 37, "clap": 39, "hh_closed": 42,
    "hh_pedal": 44, "hh_open": 46, "crash": 49, "ride": 51, "ride_bell": 53,
    "tambourine": 54, "cowbell": 56, "tom_low": 41, "tom_mid": 47, "tom_high": 50,
    "conga_slap": 62, "conga_open": 63, "conga_low": 64, "timbale": 65,
    "agogo_hi": 67, "agogo_lo": 68, "cabasa": 69, "shaker": 70, "maracas": 70,
    "guiro": 74, "claves": 75, "wood_hi": 76, "wood_lo": 77, "cuica": 78,
    "triangle": 80, "timpani": 41, "whistle": 72,
    "surdo": 36, "tamborim": 76, "pandeiro": 54, "reco": 74, "reco_reco": 74,
    "chocalho": 70, "berimbau": 47, "agogo": 67, "conga": 63,
}
DRUM_DUR: Dict[str, float] = {
    "crash": 0.9, "ride": 0.7, "hh_open": 0.3, "timpani": 0.55,
    "guiro": 0.2, "reco": 0.2, "cuica": 0.25, "triangle": 0.2,
    "agogo_hi": 0.15, "agogo_lo": 0.15,
}
DEFAULT_DRUM_DUR = 0.12


# =============================================================================
# TEXTO / NORMALIZAÇÃO
# =============================================================================

def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", _strip_accents(str(text).lower())).strip()


def _build_catalog_index() -> Dict[str, str]:
    index: Dict[str, str] = {}
    for key, entry in INSTRUMENTS.items():
        names = [key, str(entry.get("display_name") or entry.get("name") or "")]
        names += [str(a) for a in (entry.get("aliases") or [])]
        for nm in names:
            if nm:
                index.setdefault(_norm(nm).replace("_", " "), key)
    return index


_CAT_INDEX = _build_catalog_index()

_CONCEPTS: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = [
    (("bateria", "tambor", "drums", "drum", "kit", "baterista", "percussao"), ("drums",)),
    (("violao", "nylon guitar", "acoustic guitar", "guitarra acustica"),
     ("acoustic_guitar", "violao", "nylon_guitar")),
    (("guitarra", "guitarra eletrica", "electric guitar", "guitar"),
     ("electric_guitar", "guitarra eletrica", "overdriven_guitar", "guitarra")),
    (("piano eletrico", "rhodes", "electric piano"), ("electric_piano", "rhodes")),
    (("piano", "teclado", "keyboard"), ("piano", "acoustic_piano", "grand_piano")),
    (("sanfona", "acordeon", "acordeao", "gaita", "accordion"), ("accordion", "sanfona")),
    (("baixo", "contrabaixo", "bass", "electric bass"),
     ("bass", "baixo", "electric_bass", "fingered_bass", "acoustic_bass")),
    (("cavaquinho", "cavaco"), ("cavaquinho", "cavaco", "banjo", "acoustic_guitar")),
    (("bandolim", "mandolin"), ("mandolin",)),
    (("flauta", "flute"), ("flute", "flauta")),
    (("sax", "saxofone", "saxophone"), ("saxophone", "saxofone", "alto_sax")),
    (("trompete", "trompeta", "trumpet"), ("trumpet", "trompete")),
    (("trombone",), ("trombone",)),
    (("clarinete", "clarinet"), ("clarinet",)),
    (("violino", "violin"), ("violin", "violino")),
    (("violoncelo", "cello"), ("cello",)),
    (("orgao", "organ", "hammond"), ("organ", "orgao")),
    (("sintetizador", "synth"), ("synth_lead", "sintetizador", "synth")),
    (("pad", "pads"), ("synth_pad", "pad")),
    (("cordas", "strings", "orquestra"), ("strings", "cordas")),
    (("pandeiro",), ("pandeiro",)),
    (("surdo", "zabumba"), ("surdo",)),
    (("tamborim",), ("tamborim",)),
    (("agogo",), ("agogo",)),
    (("cuica",), ("cuica",)),
    (("reco reco", "recoreco", "reco", "guiro"), ("reco_reco", "reco", "guiro")),
    (("chocalho", "ganza", "maracas"), ("chocalho", "shaker", "maracas")),
    (("berimbau",), ("berimbau",)),
    (("conga", "congas", "tumbadora"), ("conga", "congas")),
    (("shaker",), ("shaker", "chocalho")),
    (("ukelele", "ukulele"), ("ukulele",)),
    (("harpa", "harp"), ("harp",)),
]


def _resolve_id(candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in INSTRUMENTS:
            return c
    for c in candidates:
        hit = _CAT_INDEX.get(_norm(c))
        if hit:
            return hit
    return None


def find_instrument(word: str) -> Optional[str]:
    w_raw = _norm(word)
    if not w_raw or len(w_raw) < 2:
        return None
    if w_raw.replace(" ", "_") in INSTRUMENTS:
        return w_raw.replace(" ", "_")
    for words, candidates in _CONCEPTS:
        if w_raw in words:
            rid = _resolve_id(candidates)
            if rid:
                return rid
    hit = _CAT_INDEX.get(w_raw) or _CAT_INDEX.get(w_raw.replace(" ", "_"))
    if hit:
        return hit
    close = difflib.get_close_matches(w_raw, list(_CAT_INDEX), n=1, cutoff=0.8)
    return _CAT_INDEX[close[0]] if close else None


def parse_instrument_arg(arg: str) -> List[str]:
    ids: List[str] = []
    for part in re.split(r"[,+;/]|\s+e\s+", arg.strip().lower()):
        part = part.strip()
        if not part:
            continue
        rid = find_instrument(part)
        if rid and rid not in ids:
            ids.append(rid)
        elif not rid:
            LOG.warning("Instrumento não reconhecido (ignorado): %r", part)
    return ids


def parse_instruments_from_text(text: str) -> List[str]:
    tokens = _norm(text).split()
    bigrams = [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
    found: List[str] = []
    for phrase in bigrams + tokens:
        rid = find_instrument(phrase)
        if rid and rid not in found:
            found.append(rid)
    return found


# =============================================================================
# TEORIA MUSICAL
# =============================================================================

SCALES: Dict[str, List[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10], "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "pentatonic_major": [0, 2, 4, 7, 9], "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
}
NOTE_NAMES_PT = {"do": 0, "re": 2, "mi": 4, "fa": 5, "sol": 7, "la": 9, "si": 11}
NOTE_NAMES_EN = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
PC_TO_NAME = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

STYLES: Dict[str, Dict[str, Any]] = {
    "rock":       dict(bpm=(100, 150), scale="minor", swing=0.0,
                       instruments=[("electric_guitar", "guitarra"), ("bass", "baixo"), ("drums",)]),
    "pop":        dict(bpm=(95, 130), scale="major", swing=0.0,
                       instruments=[("piano",), ("bass", "baixo"), ("drums",)]),
    "samba":      dict(bpm=(85, 112), scale="major", swing=0.0,
                       instruments=[("cavaquinho", "cavaco", "banjo", "acoustic_guitar", "violao"),
                                    ("acoustic_guitar", "violao"), ("bass", "baixo"),
                                    ("surdo",), ("pandeiro",), ("agogo", "tamborim")]),
    "bossa":      dict(bpm=(68, 100), scale="major", swing=0.0, sevenths=True,
                       instruments=[("acoustic_guitar", "violao"), ("piano",),
                                    ("bass", "baixo"), ("drums",)]),
    "funk":       dict(bpm=(120, 140), scale="minor", swing=0.0,
                       instruments=[("synth_lead", "sintetizador", "piano"),
                                    ("bass", "baixo"), ("drums",)]),
    "reggae":     dict(bpm=(72, 92), scale="minor", swing=0.0,
                       instruments=[("electric_guitar", "guitarra", "violao"), ("organ", "orgao"),
                                    ("bass", "baixo"), ("drums",)]),
    "trap":       dict(bpm=(130, 150), scale="minor", swing=0.0,
                       instruments=[("synth_lead", "sintetizador", "flute", "flauta", "piano"),
                                    ("bass", "baixo"), ("drums",)]),
    "electronic": dict(bpm=(120, 132), scale="minor", swing=0.0,
                       instruments=[("synth_lead", "sintetizador", "piano"),
                                    ("synth_pad", "pad", "strings", "cordas"),
                                    ("bass", "baixo"), ("drums",)]),
    "hiphop":     dict(bpm=(82, 96), scale="minor", swing=0.10,
                       instruments=[("electric_piano", "rhodes", "piano", "teclado"),
                                    ("bass", "baixo"), ("drums",)]),
    "jazz":       dict(bpm=(100, 150), scale="major", swing=0.14, sevenths=True,
                       instruments=[("saxophone", "saxofone", "trumpet", "trompete", "piano"),
                                    ("piano",), ("bass", "contrabaixo", "baixo"), ("drums",)]),
    "ambient":    dict(bpm=(55, 85), scale="major", swing=0.0,
                       instruments=[("synth_pad", "pad", "strings", "cordas", "piano"), ("piano",)]),
    "cinematic":  dict(bpm=(70, 100), scale="minor", swing=0.0,
                       instruments=[("strings", "cordas"), ("piano",), ("drums",)]),
    "classical":  dict(bpm=(70, 120), scale="major", swing=0.0,
                       instruments=[("piano",), ("strings", "cordas", "violin", "violino")]),
    "folk":       dict(bpm=(90, 125), scale="major", swing=0.0,
                       instruments=[("acoustic_guitar", "violao", "viola"),
                                    ("bass", "baixo"), ("drums",)]),
    "latin":      dict(bpm=(95, 125), scale="major", swing=0.0, sevenths=True,
                       instruments=[("piano",), ("bass", "baixo"), ("drums",),
                                    ("trumpet", "trompete", "saxophone", "saxofone")]),
    "forro":      dict(bpm=(125, 150), scale="mixolydian", swing=0.0,
                       instruments=[("accordion", "sanfona", "acordeon", "acordeao"),
                                    ("bass", "baixo"), ("drums",)]),
    "bossfight":  dict(bpm=(142, 165), scale="harmonic_minor", swing=0.0,
                       instruments=[("strings", "cordas"), ("trumpet", "trompete"),
                                    ("piano",), ("drums",)]),
    "chiptune":   dict(bpm=(130, 165), scale="major", swing=0.0,
                       instruments=[("square_lead", "chiptune lead"),
                                    ("synth_bass", "synth bass"), ("drums",)]),
    "dungeon":    dict(bpm=(50, 70), scale="harmonic_minor", swing=0.0,
                       instruments=[("strings", "cordas"), ("piano",)]),
    "breakcore":  dict(bpm=(170, 200), scale="minor", swing=0.0,
                       instruments=[("synth_lead", "sintetizador"), ("synth_pad", "pad"),
                                    ("bass", "baixo"), ("drums",)]),
    "dnb":        dict(bpm=(168, 178), scale="minor", swing=0.0,
                       instruments=[("synth_lead", "sintetizador", "piano"),
                                    ("synth_pad", "pad", "strings", "cordas"),
                                    ("bass", "baixo"), ("drums",)]),
    "synthwave":  dict(bpm=(100, 116), scale="minor", swing=0.0,
                       instruments=[("synth_lead", "sintetizador"), ("synth_pad", "pad"),
                                    ("bass", "baixo"), ("drums",)]),
    "disco":      dict(bpm=(112, 126), scale="major", swing=0.0,
                       instruments=[("electric_piano", "piano"),
                                    ("synth_pad", "pad", "strings", "cordas"),
                                    ("bass", "baixo"), ("drums",)]),
    "blues":      dict(bpm=(72, 100), scale="blues", swing=0.12, sevenths=True,
                       instruments=[("electric_guitar", "guitarra", "violao"), ("organ", "orgao"),
                                    ("bass", "baixo"), ("drums",)]),
    "metal":      dict(bpm=(140, 180), scale="harmonic_minor", swing=0.0,
                       instruments=[("electric_guitar", "guitarra"), ("bass", "baixo"), ("drums",)]),
    "choro":      dict(bpm=(110, 150), scale="major", swing=0.0, sevenths=True,
                       instruments=[("cavaquinho", "cavaco"), ("flute", "flauta"),
                                    ("pandeiro",), ("drums",)]),
    "capoeira":   dict(bpm=(122, 138), scale="mixolydian", swing=0.0,
                       instruments=[("acoustic_guitar", "violao"), ("berimbau",),
                                    ("conga", "congas"), ("agogo",), ("pandeiro",), ("drums",)]),
}
NO_DRUMS_STYLES = {"ambient", "classical", "dungeon"}
AUTO_BASS_STYLES = {"pop", "rock", "funk", "reggae", "trap", "electronic", "hiphop",
                    "jazz", "bossa", "latin", "forro", "samba", "cinematic",
                    "bossfight", "chiptune", "breakcore", "dnb", "synthwave",
                    "disco", "blues", "metal", "choro", "dungeon"}
CRASH_STYLES = {"rock", "pop", "electronic", "funk", "trap", "cinematic", "latin",
                "bossfight", "metal", "breakcore", "disco"}

# v9: pools maiores de progressão (mais variedade entre gerações)
PROGRESSIONS: Dict[str, List[List[int]]] = {
    "pop": [[0, 4, 5, 3], [0, 5, 3, 4], [5, 3, 0, 4], [0, 4, 5, 4],
            [0, 5, 1, 4], [3, 4, 0, 0]],
    "rock": [[0, 3, 4, 3], [0, 5, 3, 4], [0, 0, 3, 4], [5, 3, 0, 4],
             [0, 3, 0, 4], [0, 6, 3, 4]],
    "samba": [[1, 4, 0, 0], [2, 5, 1, 4], [1, 4, 0, 3], [0, 4, 5, 3],
              [1, 4, 0, 0], [0, 3, 4, 0]],
    "bossa": [[1, 4, 0, 0], [0, 5, 1, 4], [1, 4, 2, 5], [2, 5, 1, 4],
              [0, 4, 1, 4]],
    "funk": [[0, 0, 3, 3], [0, 3, 0, 3], [0, 3, 4, 3], [0, 0, 5, 4]],
    "reggae": [[0, 3, 4, 3], [0, 5, 3, 4], [0, 0, 3, 3], [0, 4, 3, 3]],
    "trap": [[0, 5, 3, 4], [5, 3, 0, 4], [0, 5, 0, 4], [0, 3, 5, 4]],
    "electronic": [[0, 5, 3, 4], [5, 3, 0, 4], [0, 3, 5, 4], [0, 5, 0, 3],
                   [3, 0, 4, 0]],
    "hiphop": [[0, 3, 4, 3], [0, 5, 1, 4], [1, 4, 0, 0], [0, 3, 0, 3]],
    "jazz": [[1, 4, 0, 0], [1, 4, 0, 3], [0, 3, 6, 2], [1, 4, 1, 4], [0, 5, 1, 4]],
    "ambient": [[0, 3, 0, 4], [0, 4, 3, 0], [0, 3, 5, 4], [0, 5, 3, 0]],
    "cinematic": [[0, 5, 3, 4], [0, 3, 5, 4], [0, 5, 1, 4], [0, 6, 3, 4]],
    "classical": [[0, 4, 5, 3], [0, 3, 4, 0], [0, 4, 0, 4], [0, 5, 3, 4]],
    "folk": [[0, 3, 4, 3], [0, 4, 5, 3], [0, 3, 0, 4], [0, 4, 0, 3]],
    "latin": [[0, 3, 4, 3], [0, 4, 3, 4], [1, 4, 0, 0], [0, 5, 4, 3]],
    "forro": [[0, 3, 0, 4], [0, 4, 3, 4], [0, 3, 4, 0], [0, 0, 3, 4]],
    "bossfight": [[0, 5, 3, 4], [0, 1, 4, 0], [0, 3, 0, 4], [0, 6, 5, 4]],
    "chiptune": [[0, 4, 5, 3], [0, 3, 4, 4], [0, 5, 3, 4], [5, 3, 0, 4],
                 [0, 4, 0, 4]],
    "dungeon": [[0, 3, 5, 4], [0, 5, 3, 0], [0, 1, 0, 4], [0, 3, 0, 3]],
    "breakcore": [[0, 5, 3, 4], [0, 3, 4, 3], [5, 3, 0, 4], [0, 0, 5, 4]],
    "dnb": [[0, 5, 3, 4], [0, 3, 0, 4], [5, 3, 0, 4], [0, 5, 4, 3]],
    "synthwave": [[0, 5, 3, 4], [5, 3, 0, 4], [0, 3, 5, 4], [0, 4, 5, 3]],
    "disco": [[0, 5, 3, 4], [0, 3, 4, 3], [0, 4, 5, 3], [0, 5, 1, 4]],
    "blues": [[0, 0, 0, 0, 3, 3, 0, 0, 4, 3, 0, 4],
              [0, 0, 3, 0, 3, 3, 0, 4, 0, 0, 3, 0]],
    "metal": [[0, 3, 4, 3], [0, 5, 3, 4], [0, 0, 5, 4], [0, 1, 0, 4]],
    "choro": [[1, 4, 0, 0], [2, 5, 1, 4], [0, 4, 1, 4], [1, 4, 2, 5]],
    "capoeira": [[0, 3, 0, 4], [0, 4, 3, 4], [0, 0, 3, 4], [0, 3, 4, 0]],
}

BASS_PATTERNS: Dict[str, List[Tuple[float, str, float]]] = {
    "rock": [(i * 0.5, "root" if i != 5 else "fifth", 0.45) for i in range(8)],
    "pop": [(0.0, "root", 0.9), (1.5, "root", 0.4), (2.0, "fifth", 0.9), (3.5, "root", 0.4)],
    "samba": [(0.0, "root", 0.7), (1.5, "root", 0.35), (2.0, "root", 0.7), (3.5, "fifth", 0.35)],
    "bossa": [(0.0, "root", 1.4), (1.5, "root", 0.4), (2.0, "fifth", 1.4), (3.5, "fifth", 0.4)],
    "jazz": [(0.0, "root", 0.9), (1.0, "third", 0.9), (2.0, "fifth", 0.9), (3.0, "approach", 0.9)],
    "funk": [(0.0, "root", 0.4), (0.75, "root", 0.2), (1.5, "octave", 0.4),
             (2.0, "root", 0.4), (2.75, "fifth", 0.2), (3.5, "root", 0.4)],
    "reggae": [(2.0, "root", 0.7), (3.5, "fifth", 0.35)],
    "trap": [(0.0, "root", 1.4), (2.5, "root", 1.3)],
    "electronic": [(i * 0.5, "root", 0.45) for i in range(8)],
    "hiphop": [(0.0, "root", 0.7), (2.0, "fifth", 0.65), (3.25, "root", 0.35)],
    "ambient": [(0.0, "root", 3.8)], "cinematic": [(0.0, "root", 3.8)],
    "classical": [(0.0, "root", 1.9), (2.0, "fifth", 1.9)],
    "folk": [(0.0, "root", 1.9), (2.0, "fifth", 1.9)],
    "latin": [(0.0, "root", 0.9), (1.5, "root", 0.4), (2.5, "root", 0.9), (3.5, "fifth", 0.4)],
    "forro": [(0.0, "root", 0.7), (1.75, "root", 0.3), (2.0, "root", 0.7), (3.5, "fifth", 0.35)],
    "bossfight": [(i * 0.5, "root" if i not in (3, 7) else "fifth", 0.45) for i in range(8)],
    "chiptune": [(i * 0.5, "root" if i % 2 == 0 else "octave", 0.4) for i in range(8)],
    "dungeon": [(0.0, "root", 3.8)],
    "breakcore": [(0.0, "root", 0.2), (0.75, "root", 0.2), (1.5, "octave", 0.2),
                  (2.25, "root", 0.2), (3.0, "fifth", 0.2), (3.5, "octave", 0.2)],
    "dnb": [(i * 0.5, "root" if i % 2 == 0 else "octave", 0.45) for i in range(8)],
    "synthwave": [(i * 0.5, "root" if i % 2 == 0 else "octave", 0.42) for i in range(8)],
    "disco": [(i * 0.25, "root" if i % 2 == 0 else "octave", 0.22) for i in range(16)],
    "blues": [(0.0, "root", 0.9), (1.0, "fifth", 0.9), (2.0, "root", 0.9), (3.0, "approach", 0.9)],
    "metal": [(i * 0.5, "root", 0.45) for i in range(8)],
    "choro": [(0.0, "root", 0.7), (1.5, "root", 0.35), (2.0, "root", 0.7), (3.5, "fifth", 0.35)],
    "capoeira": [(0.0, "root", 0.7), (1.5, "root", 0.35), (2.0, "fifth", 0.7), (3.5, "root", 0.35)],
}

COMP_RHYTHMS: Dict[str, List[List[float]]] = {
    "pop": [[0.0, 2.0], [0.0, 1.5, 2.0, 3.0], [0.0, 2.0, 3.5], [0.0, 1.0, 2.0, 3.0]],
    "rock": [[0.0, 2.0], [0.0, 2.0, 3.5], [0.0, 1.5, 2.0, 3.0], [0.0, 1.5, 2.5, 3.5]],
    "samba": [[0.0, 1.75, 2.5, 3.25], [0.75, 1.5, 2.25, 3.0, 3.75], [0.0, 0.75, 2.0, 2.75]],
    "bossa": [[0.0, 0.75, 2.0, 2.75], [0.0, 1.5, 2.5], [0.75, 1.5, 2.0, 2.75, 3.5]],
    "jazz": [[0.0, 1.5], [1.5], [0.0], [2.5, 3.5]],
    "funk": [[0.0, 0.75, 1.5, 2.25, 3.0], [0.0, 1.5, 2.5, 3.25]],
    "reggae": [[0.5, 1.5, 2.5, 3.5]],
    "trap": [[0.0], [0.0, 2.5]],
    "electronic": [[0.0], [0.0, 1.5, 3.5], [0.0, 2.5]],
    "hiphop": [[0.0, 2.5], [0.0, 1.75, 3.25]],
    "ambient": [[0.0]], "cinematic": [[0.0]],
    "classical": [[0.0, 2.0]],
    "folk": [[0.0, 2.0], [0.0, 1.0, 2.0, 3.0]],
    "latin": [[0.0, 0.75, 1.5, 2.0, 2.75, 3.5], [0.0, 0.75, 1.5, 2.25, 3.0, 3.75]],
    "forro": [[0.0, 0.75, 2.0, 2.75], [0.0, 1.5, 2.0, 3.5]],
    "bossfight": [[0.0, 1.5, 2.0, 3.5], [0.0, 0.75, 2.0, 2.75]],
    "chiptune": [[i * 0.25 for i in range(16)]],
    "dungeon": [[0.0]],
    "breakcore": [[0.0, 0.75, 1.5, 2.25, 3.0, 3.75]],
    "dnb": [[0.0, 2.5], [0.0, 1.75, 3.25]],
    "synthwave": [[0.0, 1.5, 2.5], [0.0, 0.75, 2.0, 2.75]],
    "disco": [[0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.0, 3.5]],
    "blues": [[0.0, 1.0, 2.0, 3.0]],
    "metal": [[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]],
    "choro": [[0.0, 0.75, 1.5, 2.0, 2.75, 3.5], [0.75, 1.5, 2.25, 3.0, 3.75]],
    "capoeira": [[0.0, 0.75, 1.5, 2.25, 3.0], [0.0, 1.5, 2.0, 3.5]],
}

STRUM_FAMILIES = ("guitar", "viola", "cavaquinho", "banjo", "ukulele", "cavaco")

SECTION_PROFILES: Dict[str, Dict[str, Any]] = {
    "intro":  dict(vel=64, drums=False, bass=True, comp=True, melody=False, dense=False),
    "verso":  dict(vel=76, drums=True, bass=True, comp=True, melody=True, dense=False),
    "refrao": dict(vel=92, drums=True, bass=True, comp=True, melody=True, dense=True),
    "ponte":  dict(vel=72, drums=True, bass=True, comp=True, melody=True, dense=False),
    "solo":   dict(vel=88, drums=True, bass=True, comp=True, melody=True, dense=True),
    "outro":  dict(vel=68, drums=True, bass=True, comp=True, melody=False, dense=False),
}

SONG_FORMS: Dict[str, List[Tuple[str, int]]] = {
    "pop": [("intro", 2), ("verso", 4), ("refrao", 4), ("verso", 4),
            ("refrao", 4), ("ponte", 2), ("refrao", 4), ("outro", 2)],
    "rock": [("intro", 2), ("verso", 4), ("refrao", 4), ("verso", 4),
             ("refrao", 4), ("solo", 4), ("refrao", 4)],
    "samba": [("intro", 2), ("verso", 4), ("refrao", 4), ("verso", 4),
              ("refrao", 4), ("outro", 2)],
    "bossa": [("intro", 2), ("verso", 4), ("refrao", 4), ("verso", 4), ("refrao", 4)],
    "jazz": [("intro", 1), ("verso", 4), ("solo", 4), ("verso", 4), ("outro", 1)],
    "cinematic": [("intro", 2), ("verso", 4), ("refrao", 4), ("verso", 4),
                  ("refrao", 4), ("ponte", 2), ("refrao", 4), ("outro", 2)],
    "ambient": [("verso", 4), ("verso", 4), ("verso", 4)],
    "bossfight": [("intro", 2), ("verso", 4), ("refrao", 4), ("ponte", 2),
                  ("solo", 4), ("refrao", 4)],
    "chiptune": [("intro", 2), ("refrao", 4), ("verso", 4), ("refrao", 4),
                 ("ponte", 2), ("refrao", 4)],
    "breakcore": [("intro", 2), ("verso", 4), ("refrao", 4), ("ponte", 2),
                  ("refrao", 4), ("refrao", 4)],
}
_DEFAULT_FORM = [("intro", 2), ("verso", 4), ("refrao", 4), ("verso", 4), ("refrao", 4)]

RHYTHM_CELLS_SPARSE = [
    [(0.0, 1.0)], [(0.0, 0.5), (0.5, 0.5)], [(0.5, 0.5), (1.0, 1.0)],
    [(0.0, 0.75), (1.5, 0.5)], [(1.0, 0.5), (1.5, 0.5), (2.5, 1.0)],
    [(0.0, 0.5), (1.5, 0.5), (2.0, 0.5), (3.0, 0.5)],
]
RHYTHM_CELLS_DENSE = [
    [(0.0, 0.5), (0.5, 0.5), (1.0, 0.5), (1.5, 0.5), (2.5, 0.5), (3.0, 1.0)],
    [(0.0, 0.5), (0.75, 0.25), (1.0, 0.5), (1.5, 0.5), (2.0, 0.5),
     (2.5, 0.5), (3.0, 0.5), (3.5, 0.5)],
    [(0.5, 0.25), (0.75, 0.25), (1.0, 1.0), (2.5, 0.5), (3.0, 0.5), (3.5, 0.5)],
]


# =============================================================================
# PERCUSSÃO (kits por estilo)
# =============================================================================

def _kit_pattern(style: str, bar: int, rng: random.Random) -> List[Tuple[float, str, int]]:
    p: List[Tuple[float, str, int]] = []
    if style == "rock":
        p = [(0.0, "kick", 106), (1.0, "snare", 99), (2.0, "kick", 101), (3.0, "snare", 101)]
        p += [(i * 0.5, "hh_closed", 82 if i % 2 == 0 else 64) for i in range(8)]
        if bar % 4 == 3:
            p += [(3.5, "kick", 90), (3.75, "hh_open", 72)]
    elif style == "pop":
        p = [(0.0, "kick", 100), (1.0, "snare", 94), (2.0, "kick", 96), (3.0, "snare", 96)]
        p += [(i * 0.5, "hh_closed", 74 if i % 2 == 0 else 56) for i in range(8)]
        if bar % 2 == 1:
            p.append((3.75, "hh_open", 64))
    elif style == "samba":
        p = [(1.0, "kick", 102), (3.0, "kick", 96)]
        p += [(i * 0.25, "tamborim", [78, 52, 66, 52][i % 4]) for i in range(16)]
        p += [(0.0, "agogo_hi", 82), (0.75, "agogo_lo", 72), (1.5, "agogo_hi", 76),
              (2.0, "agogo_hi", 82), (2.75, "agogo_lo", 72), (3.5, "agogo_hi", 76)]
        p += [(i * 0.25, "shaker", 56 if i % 2 == 0 else 66) for i in range(16)]
        p += [(0.5, "snare", 44), (2.5, "snare", 42)]
    elif style == "bossa":
        p = [(0.0, "kick", 78), (0.75, "kick", 58), (2.0, "kick", 78), (2.75, "kick", 58)]
        p += [(1.0, "rim", 70), (3.0, "rim", 70)]
        p += [(i * 0.5, "hh_closed", 64 if i % 2 == 0 else 52) for i in range(8)]
    elif style == "funk":
        p = [(0.0, "kick", 105), (1.75, "kick", 92), (2.25, "kick", 100),
             (1.0, "clap", 96), (3.0, "clap", 98)]
        if bar % 2 == 1:
            p.append((3.5, "kick", 88))
        p += [(i * 0.25, "hh_closed", [58, 40, 50, 40][i % 4]) for i in range(16)]
    elif style == "reggae":
        p = [(2.0, "kick", 100), (2.0, "rim", 88)]
        if bar % 2 == 1:
            p.append((3.75, "kick", 68))
        p += [(i * 0.5, "hh_closed", 66 if i % 2 == 0 else 50) for i in range(8)]
    elif style == "trap":
        p = [(0.0, "kick", 106), (1.75, "kick", 94), (2.0, "snare", 100)]
        if bar % 4 == 2:
            p.append((3.5, "kick", 90))
        p += [(i * 0.25, "hh_closed", [82, 48, 64, 48][i % 4]) for i in range(16)]
        if bar % 4 == 3 and rng.random() < 0.5:
            p += [(3.0 + i * 0.125, "hh_closed", 60) for i in range(8)]
    elif style == "electronic":
        p = [(b, "kick", 100) for b in range(4)]
        p += [(1.0, "clap", 88), (3.0, "clap", 88)]
        p += [(b + 0.5, "hh_open", 70) for b in range(4)]
    elif style == "hiphop":
        p = [(0.0, "kick", 102), (2.5, "kick", 96), (1.0, "snare", 98), (3.0, "snare", 98)]
        p += [(i * 0.5, "hh_closed", 70 if i % 2 == 0 else 54) for i in range(8)]
    elif style == "jazz":
        p = [(i * 0.5, "ride", 74 if i % 2 == 0 else 58) for i in range(8)]
        p += [(1.0, "hh_pedal", 64), (3.0, "hh_pedal", 64)]
        p += [(0.0, "kick", 34), (2.0, "kick", 30)]
        if rng.random() < 0.4:
            p.append((rng.choice([1.5, 2.5, 3.5]), "snare", 48))
    elif style == "latin":
        p = [(0.5, "conga_low", 70), (1.5, "conga_open", 84), (2.5, "conga_low", 70),
             (3.5, "conga_open", 88), (3.75, "conga_slap", 74)]
        if bar % 2 == 0:
            p += [(0.0, "claves", 90), (1.5, "claves", 86), (3.0, "claves", 88)]
        else:
            p += [(1.0, "claves", 88), (2.5, "claves", 86)]
        p.append((0.0, "cowbell", 52))
    elif style == "cinematic":
        p = [(0.0, "timpani", 92), (2.0, "timpani", 84)]
        if bar % 4 == 0:
            p.append((0.0, "crash", 88))
    elif style == "forro":
        p = [(0.0, "kick", 100), (2.0, "kick", 94), (1.5, "snare", 56), (3.5, "snare", 58)]
        p += [(i * 0.5, "triangle", 62) for i in range(8)]
    elif style == "folk":
        p = [(1.0, "tambourine", 62), (3.0, "tambourine", 60)]
        if rng.random() < 0.3:
            p.append((0.0, "kick", 58))
    elif style == "choro":
        p = [(0.0, "kick", 60), (2.0, "kick", 54)]
        p += [(i * 0.5, "rim", 48 if i % 2 == 0 else 38) for i in range(8)]
    elif style == "capoeira":
        p = [(0.0, "conga_low", 88), (1.0, "conga_low", 66), (1.5, "conga_open", 74),
             (2.0, "conga_low", 88), (3.0, "conga_low", 66), (3.5, "conga_open", 74)]
    elif style == "bossfight":
        p = [(b, "kick", 104) for b in range(4)]
        p += [(1.0, "snare", 98), (3.0, "snare", 100)]
        p += [(0.5, "snare", 56), (1.5, "snare", 60), (2.5, "snare", 64),
              (3.25, "snare", 70), (3.5, "snare", 78), (3.75, "snare", 86)]
        p += [(0.0, "timpani", 96), (2.0, "timpani", 90)]
        if bar % 4 == 0:
            p.append((0.0, "crash", 94))
    elif style == "chiptune":
        p = [(0.0, "kick", 88), (2.0, "kick", 84), (1.0, "snare", 86), (3.0, "snare", 88)]
        p += [(i * 0.5, "hh_closed", [72, 38, 54, 38][i % 4]) for i in range(8)]
    elif style == "breakcore":
        p = [(0.0, "kick", 104), (1.0, "snare", 100), (2.0, "snare", 98),
             (2.5, "kick", 96), (3.0, "snare", 100)]
        if bar % 2 == 1:
            p += [(0.75, "snare", 84), (1.5, "kick", 88), (1.75, "snare", 80),
                  (3.25, "snare", 90), (3.5, "snare", 96), (3.75, "snare", 102)]
        p += [(i * 0.25, "hh_closed", [72, 40, 58, 40][i % 4]) for i in range(16)]
        if bar % 4 == 3:
            p += [(3.0 + i * 0.125, "snare", 50 + 6 * i) for i in range(8)]
    elif style == "dnb":
        p = [(0.0, "kick", 104), (2.5, "kick", 96), (2.0, "snare", 102), (3.75, "snare", 66)]
        p += [(i * 0.5, "hh_closed", 68 if i % 2 == 0 else 50) for i in range(8)]
        p.append((1.5, "hh_open", 58))
    elif style == "synthwave":
        p = [(0.0, "kick", 98), (1.0, "snare", 94), (2.0, "kick", 94), (3.0, "snare", 96)]
        p += [(b + 0.5, "hh_open", 62) for b in range(4)]
        if bar % 4 == 3:
            p.append((3.5, "tom_high", 70))
    elif style == "disco":
        p = [(b, "kick", 100) for b in range(4)]
        p += [(1.0, "snare", 90), (3.0, "snare", 92)]
        p += [(b + 0.5, "hh_open", 74) for b in range(4)]
    elif style == "blues":
        p = [(0.0, "kick", 96), (2.0, "kick", 90), (1.0, "snare", 88), (3.0, "snare", 90)]
        p += [(i * 0.5, "ride", 66 if i % 2 == 0 else 52) for i in range(8)]
    elif style == "metal":
        kick_pos = [i * 0.5 for i in range(8)]
        if bar % 2 == 1:
            kick_pos = [i * 0.25 for i in range(16)]
        p = [(k, "kick", 92) for k in kick_pos]
        p += [(1.0, "snare", 102), (3.0, "snare", 104)]
        p += [(i * 0.5, "ride", 70 if i % 2 == 0 else 56) for i in range(8)]
        if bar % 4 == 0:
            p.append((0.0, "crash", 100))
    elif style in ("dungeon", "ambient", "classical"):
        p = []
    else:
        return _kit_pattern("pop", bar, rng)
    return p


def _fill(style: str, rng: random.Random) -> List[Tuple[float, str, int]]:
    if style == "bossfight":
        return [(2.0, "snare", 60), (2.25, "snare", 68), (2.5, "snare", 76),
                (2.75, "snare", 84), (3.0, "snare", 92), (3.25, "snare", 98),
                (3.5, "timpani", 100), (3.75, "timpani", 104)]
    if style in ("breakcore", "dnb"):
        return [(2.0 + i * 0.125, "snare", 44 + 6 * i) for i in range(16)]
    if style == "metal":
        return [(2.0 + 0.25 * i, "tom_high" if i % 2 == 0 else "tom_mid", 84 + 3 * i)
                for i in range(8)]
    if style == "blues":
        return [(2.5, "snare", 70), (3.0, "tom_mid", 78), (3.5, "snare", 84)]
    if style in ("synthwave", "disco"):
        return [(3.0, "hh_open", 76), (3.5, "snare", 82)]
    if style == "chiptune":
        return [(2.5 + i * 0.125, "tom_high" if i % 2 == 0 else "snare", 60 + 4 * i)
                for i in range(12)]
    if style in ("rock", "pop", "electronic", "folk"):
        seq = ["snare", "snare", "tom_high", "tom_high", "tom_mid", "tom_mid", "tom_low", "tom_low"]
        return [(2.0 + 0.25 * i, seq[i], 78 + 5 * i) for i in range(8)]
    if style in ("samba", "bossa", "forro", "latin", "choro", "capoeira"):
        return [(2.5, "tamborim", 72), (2.75, "tamborim", 78), (3.0, "agogo_hi", 80),
                (3.25, "tamborim", 76), (3.5, "agogo_hi", 82), (3.75, "agogo_lo", 76)]
    if style in ("trap", "hiphop", "funk"):
        return [(2.0 + 0.125 * i, "hh_closed", 42 + 5 * i) for i in range(16)]
    if style == "jazz":
        return [(2.0, "ride", 82), (2.5, "snare", 62), (3.0, "snare", 74),
                (3.5, "tom_mid", 82), (3.75, "tom_low", 86)]
    if style == "cinematic":
        return [(2.0 + 0.25 * i, "timpani", 58 + 5 * i) for i in range(8)]
    if style == "reggae":
        return [(2.5, "rim", 72), (3.0, "rim", 78), (3.5, "rim", 84)]
    return []


PERC_16: Dict[str, Tuple[List[int], bool]] = {
    "pandeiro":   ([92, 0, 52, 68, 80, 0, 52, 68, 88, 0, 52, 68, 80, 0, 52, 72], False),
    "surdo":      ([0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0], False),
    "tamborim":   ([0, 64, 78, 64, 0, 64, 78, 64, 0, 64, 78, 64, 0, 64, 78, 70], False),
    "chocalho":   ([58, 70, 58, 70] * 4, False),
    "shaker":     ([50, 64, 50, 64] * 4, False),
    "reco_reco":  ([0, 72, 0, 72, 0, 72, 0, 72, 0, 72, 0, 72, 0, 72, 0, 72], False),
    "agogo":      ([84, 0, 66, 0, 76, 0, 66, 0, 84, 0, 66, 0, 76, 0, 66, 0], True),
    "cuica":      ([0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 76, 0], False),
    "berimbau":   ([82, 0, 60, 0, 74, 0, 60, 0, 82, 0, 60, 0, 74, 0, 60, 64], False),
    "conga":      ([80, 0, 0, 60, 70, 0, 84, 0, 78, 0, 0, 60, 70, 0, 86, 0], False),
    "tambourine": ([64, 0, 52, 0, 84, 0, 52, 0, 64, 0, 52, 0, 84, 0, 52, 64], False),
}
_DEFAULT_PERC_16 = ([60, 0, 0, 0, 60, 0, 0, 0, 60, 0, 0, 0, 60, 0, 0, 0], False)


# =============================================================================
# HELPERS
# =============================================================================

def _is_perc(inst_id: str) -> bool:
    return bool(INSTRUMENTS.get(inst_id, {}).get("is_percussion"))


def _perc_pitch(inst_id: str) -> int:
    entry = INSTRUMENTS.get(inst_id, {})
    for key in ("pitch", "gm_pitch", "note", "midi_note", "gm"):
        v = entry.get(key)
        if isinstance(v, int) and 35 <= v <= 81:
            return v
    return GM.get(inst_id, 76)


def _chord_offsets(scale: List[int], degree: int, add7: bool) -> List[int]:
    n = len(scale)
    offs = [scale[(degree + k) % n] + 12 * ((degree + k) // n) for k in (0, 2, 4, 6)]
    return offs if add7 else offs[:3]


def _voice_chord(prev: Optional[List[int]], offsets: List[int],
                 root_midi: int, center: int = 62) -> List[int]:
    voiced: List[int] = []
    for j, off in enumerate(offsets):
        target = prev[j % len(prev)] if prev else center
        pitch = root_midi + off
        while pitch < target - 6:
            pitch += 12
        while pitch > target + 6:
            pitch -= 12
        voiced.append(pitch)
    return sorted(set(min(127, max(0, p)) for p in voiced))


def _scale_pitch(root_midi: int, scale: List[int], idx: int) -> int:
    n = len(scale)
    return root_midi + scale[idx % n] + 12 * (idx // n)


def _snap_to_chord(idx: int, base_deg: int, n: int) -> int:
    best, best_d = idx, 99
    for d in (0, 2, 4):
        for oct_ in (-1, 0, 1):
            cand = base_deg + d + oct_ * n
            if abs(cand - idx) < best_d:
                best, best_d = cand, abs(cand - idx)
    return best


def _bass_pitch(kind: str, root_abs: int, next_root_abs: int,
                scale: List[int], degree: int, style: str) -> int:
    if kind == "fifth":
        p = root_abs + 7
    elif kind == "third":
        n = len(scale)
        interval = (scale[(degree + 2) % n] - scale[degree % n]) % 12
        p = root_abs + (3 if interval == 3 else 4)
    elif kind == "octave":
        p = root_abs + 12
    elif kind == "approach":
        p = next_root_abs - 1 if next_root_abs >= root_abs else next_root_abs + 1
    else:
        p = root_abs
    while p < 33:
        p += 12
    while p > 50:
        p -= 12
    if style == "trap" and p >= 33:
        p -= 12
    return p


def _monophonic(inst_id: str) -> bool:
    return any(m in inst_id for m in
               ("sax", "trumpet", "flute", "clarinet", "violin", "trombone",
                "synth_lead", "square_lead"))


# =============================================================================
# v9: MOTIVOS COM DESENVOLVIMENTO TEMÁTICO (fim do "tudo igual")
# =============================================================================

def _new_motif(rng: random.Random, dense: bool, up_bias: float = 0.5):
    cells = RHYTHM_CELLS_DENSE if (dense and rng.random() < 0.7) else RHYTHM_CELLS_SPARSE
    cell = list(rng.choice(cells))
    steps = [rng.choice([0, 2, 4])]
    for _ in range(len(cell) - 1):
        mag = rng.choice([1, 1, 2])
        direction = 1 if rng.random() < up_bias else -1
        steps.append(steps[-1] + direction * mag)
    steps = [max(-2, min(9, s)) for s in steps]
    return sorted(zip(cell, steps), key=lambda x: x[0][0])


def _develop_motif(motif, op: str, rng: Optional[random.Random] = None):
    """Transforma um motivo (técnica clássica de desenvolvimento temático)."""
    if motif is None:
        return None
    out = [((p, d), s) for (p, d), s in motif]
    if op == "invert":        # contorno espelhado
        out = [((p, d), max(-3, min(10, 2 - s))) for (p, d), s in out]
    elif op == "shift":       # transposição do tema
        step = rng.choice((2, -2, 3)) if rng is not None else 2
        out = [((p, d), max(-3, min(10, s + step))) for (p, d), s in out]
    elif op == "retro":       # melodia de trás pra frente
        steps = [s for _, s in out][::-1]
        out = [((p, d), s) for (p, d), s in zip([pd for pd, _ in out], steps)]
    elif op == "sparse":      # rarefação (menos notas)
        out = out[::2] or out[:1]
    return out


def _vary_motif(motif, rng: random.Random):
    varied = [((p, d), s) for (p, d), s in motif]
    if varied and rng.random() < 0.6:
        i = rng.randrange(len(varied))
        (p, d), s = varied[i]
        varied[i] = ((p, d), max(-3, min(9, s + rng.choice([-1, 1]))))
    if rng.random() < 0.4:
        shift = rng.choice([-2, -1, 1, 2])
        varied = [((p, d), max(-3, min(10, s + shift))) for (p, d), s in varied]
    return varied


def _new_phrase(rng: random.Random, dense: bool, up_bias: float = 0.5,
                prev: Optional[Dict[int, list]] = None) -> Dict[int, list]:
    """Frase de 4 compassos. Com `prev`, DERIVA do tema anterior (coerência
    com variedade) em vez de sortear tudo de novo."""
    if prev and prev.get(0) and rng.random() < 0.7:
        op = rng.choice(("invert", "shift", "retro", "sparse"))
        a = _develop_motif(prev[0], op)
    else:
        a = _new_motif(rng, dense, up_bias)
    rep = _vary_motif(a, rng) if rng.random() < 0.6 else [((p, d), s) for (p, d), s in a]
    if rng.random() < 0.35:
        rep = [((p, d), max(-3, min(9, 4 - s))) for (p, d), s in rep]
    answer = _new_motif(rng, dense, up_bias)
    if rng.random() < 0.5:
        answer = [((p, d), max(-3, min(9, s - 2))) for (p, d), s in answer]
    phrase: Dict[int, list] = {0: a, 1: rep, 2: answer}
    phrase[3] = None if rng.random() < 0.55 else _vary_motif(answer, rng)
    return phrase


def _vary_phrase(phrase: Dict[int, list], rng: random.Random) -> Dict[int, list]:
    return {k: (_vary_motif(v, rng) if v is not None and rng.random() < 0.6 else v)
            for k, v in phrase.items()}


# =============================================================================
# DETECÇÃO NO PROMPT (PT-BR)
# =============================================================================

_STYLE_MAP: List[Tuple[Tuple[str, ...], str]] = [
    (("boss fight", "bossfight", "chefe final", "batalha de chefe", "batalha",
      "luta", "guerra", "battle", "boss"), "bossfight"),
    (("chiptune", "chip tune", "8bit", "8 bits", "8 bit", "video game",
      "videogame", "jogo", "nes", "retro"), "chiptune"),
    (("dungeon", "masmorra", "terror", "horror", "sombrio", "assustador",
      "medo", "suspense", "dark"), "dungeon"),
    (("breakcore",), "breakcore"),
    (("drum and bass", "drum n bass", "dnb", "jungle", "liquid"), "dnb"),
    (("synthwave", "synth wave", "retrowave", "vaporwave", "anos 80", "oitenta"), "synthwave"),
    (("disco", "boogie", "discoteca"), "disco"),
    (("blues",), "blues"),
    (("metal", "heavy metal", "thrash", "death metal", "doom"), "metal"),
    (("choro", "chorinho"), "choro"),
    (("capoeira",), "capoeira"),
    (("samba", "sambinha", "pagode", "batucada"), "samba"),
    (("bossa nova", "bossa", "mpb"), "bossa"),
    (("rock and roll", "rock", "punk"), "rock"),
    (("pop",), "pop"),
    (("funk carioca", "baile funk", "tamborzao", "funk"), "funk"),
    (("reggae", "raggae"), "reggae"),
    (("trap",), "trap"),
    (("eletronica", "eletronico", "electronic", "edm", "techno", "house", "dance"), "electronic"),
    (("hip hop", "hiphop", "rap", "boom bap"), "hiphop"),
    (("jazz",), "jazz"),
    (("ambient", "ambiente", "chill", "lofi", "lo fi", "meditativo", "meditativa"), "ambient"),
    (("cinematografico", "cinematic", "epico", "epica", "trilha sonora", "soundtrack"), "cinematic"),
    (("classica", "classical", "erudita", "erudito"), "classical"),
    (("folk", "sertanejo", "acustico", "acustica", "country"), "folk"),
    (("latina", "latino", "salsa", "caribe", "caribenha", "cumbia"), "latin"),
    (("forro", "baiao", "xote", "xaxado"), "forro"),
]
_FLAT_STYLE: Dict[str, str] = {w: s for words, s in _STYLE_MAP for w in words}


def detect_style(text: str) -> Optional[str]:
    t = _norm(text)
    if not t:
        return None
    pairs = sorted(((w, s) for words, s in _STYLE_MAP for w in words),
                   key=lambda x: -len(x[0]))
    for w, s in pairs:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return s
    return None


def detect_key(text: str) -> Tuple[Optional[int], Optional[bool]]:
    t = _norm(text)
    m = re.search(
        r"\b(?:em|tom de|tonalidade de)\s+(do|re|mi|fa|sol|la|si|[a-g])"
        r"( sustenido| bemol)?( menor| maior)?", t)
    if not m:
        return None, None
    pc = NOTE_NAMES_PT.get(m.group(1), NOTE_NAMES_EN.get(m.group(1)))
    if pc is None:
        return None, None
    if "sustenido" in (m.group(2) or ""):
        pc += 1
    if "bemol" in (m.group(2) or ""):
        pc -= 1
    mode = m.group(3)
    return pc % 12, (True if mode == " menor" else (False if mode == " maior" else None))


def detect_bpm(text: str, rng: random.Random, style_cfg: Dict[str, Any]) -> int:
    t = _norm(text)
    m = re.search(r"(\d{2,3})\s*(?:bpm|bps|batidas)", t)
    if m:
        return max(40, min(220, int(m.group(1))))
    lo, hi = style_cfg["bpm"]
    if re.search(r"\blent[oa]s?\b|\bcalm[oa]s?\b", t):
        lo, hi = max(50, lo - 25), lo + 5
    elif re.search(r"\brapid[oa]s?\b|\banimad[oa]s?\b|\bpesad[oa]s?\b", t):
        lo, hi = hi - 5, min(200, hi + 25)
    return rng.randint(lo, hi)


def parse_key_arg(s: str) -> Tuple[Optional[int], Optional[bool]]:
    m = re.match(r"^([A-Ga-g])(#|b)?(m)?$", s.strip())
    if not m:
        return None, None
    pc = NOTE_NAMES_EN[m.group(1).lower()]
    if m.group(2) == "#":
        pc += 1
    elif m.group(2) == "b":
        pc -= 1
    return pc % 12, bool(m.group(3))


# =============================================================================
# CAMADA DE IA (dataset + autoencoder + feedback + referência)
# =============================================================================

_KB_GENRES: Dict[str, Tuple[str, ...]] = {
    "rock": ("rock",), "pop": ("pop",), "samba": ("samba", "latin"),
    "bossa": ("bossa", "latin"), "funk": ("funk", "hiphop"), "reggae": ("reggae", "latin"),
    "trap": ("trap", "hiphop"), "electronic": ("electronic", "edm"),
    "hiphop": ("hiphop", "hip hop", "rap"), "jazz": ("jazz",), "ambient": ("ambient",),
    "cinematic": ("cinematic", "soundtrack"), "classical": ("classical",),
    "folk": ("folk",), "latin": ("latin",), "forro": ("forro", "latin"),
    "bossfight": ("cinematic", "classical"), "chiptune": ("chiptune", "electronic"),
    "dungeon": ("ambient",), "breakcore": ("breakcore", "electronic"),
    "dnb": ("dnb", "electronic"), "synthwave": ("synthwave", "electronic"),
    "disco": ("disco", "pop"), "blues": ("blues",), "metal": ("metal", "rock"),
    "choro": ("choro", "latin"), "capoeira": ("capoeira", "latin"),
}

_VIBE_FALLBACK: Dict[str, Tuple[float, float, float]] = {
    "rock": (0.80, 0.45, 0.55), "pop": (0.65, 0.40, 0.70), "samba": (0.75, 0.60, 0.80),
    "bossa": (0.45, 0.65, 0.70), "funk": (0.85, 0.55, 0.60), "reggae": (0.55, 0.45, 0.65),
    "trap": (0.60, 0.50, 0.40), "electronic": (0.75, 0.35, 0.60), "hiphop": (0.65, 0.55, 0.50),
    "jazz": (0.55, 0.80, 0.60), "ambient": (0.20, 0.30, 0.55), "cinematic": (0.55, 0.50, 0.45),
    "classical": (0.40, 0.70, 0.55), "folk": (0.45, 0.45, 0.60), "latin": (0.75, 0.55, 0.80),
    "forro": (0.80, 0.50, 0.85), "bossfight": (0.90, 0.55, 0.35), "chiptune": (0.75, 0.40, 0.75),
    "dungeon": (0.15, 0.40, 0.20), "breakcore": (0.95, 0.50, 0.45), "dnb": (0.85, 0.45, 0.50),
    "synthwave": (0.60, 0.35, 0.55), "disco": (0.80, 0.40, 0.85), "blues": (0.50, 0.60, 0.30),
    "metal": (0.95, 0.50, 0.35), "choro": (0.60, 0.80, 0.65), "capoeira": (0.70, 0.55, 0.70),
}

_FEEDBACK_PATH = Path("memory/feedback.json")


def _kb_find(node: Any, keys: set, depth: int = 0) -> Optional[Dict[str, Any]]:
    if depth > 4 or not isinstance(node, dict):
        return None
    for k, v in node.items():
        if _norm(str(k)) in keys and isinstance(v, dict):
            return v
    for v in node.values():
        if isinstance(v, dict):
            found = _kb_find(v, keys, depth + 1)
            if found:
                return found
    return None


def _kb_bpm(style: str, log: logging.Logger) -> Optional[Tuple[float, float]]:
    try:
        kb = json.loads(Path("knowledge_base.json").read_text(encoding="utf-8"))
    except Exception as exc:
        log.debug("knowledge_base.json indisponível: %s", exc)
        return None
    keys = {_norm(k) for k in _KB_GENRES.get(style, (style,))}
    node = _kb_find(kb, keys)
    if not node:
        return None
    for k in ("bpm_mean", "avg_bpm", "mean_bpm", "bpm_avg", "tempo"):
        v = node.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v), 0.0
    v = node.get("bpm")
    if isinstance(v, dict):
        m = v.get("mean") or v.get("avg")
        s = v.get("std") or v.get("stdev")
        if isinstance(m, (int, float)) and m > 0:
            return float(m), float(s or 0.0)
    if isinstance(v, (int, float)) and v > 0:
        return float(v), 0.0
    return None


def _ai_personality(style: str, rng: random.Random, log: logging.Logger,
                    use_ae: bool = True) -> Dict[str, Any]:
    src = "priors"
    e: Optional[float] = None
    c: Optional[float] = None
    v: Optional[float] = None
    if use_ae:
        try:
            import train as _train
            model, _meta, backend = _train.load_autoencoder()
            genre = _KB_GENRES.get(style, (style,))[0]
            proto = _train.song_to_features(120, genre, 180.0, 0.5, 0.5, 0.5)
            X = _np.asarray([proto], dtype=_np.float32)
            r = _train.reconstruct(model, X, backend)[0]
            e, c, v = float(r[14]), float(r[15]), float(r[16])
            src = "autoencoder"
        except Exception as exc:
            log.debug("IA: autoencoder indisponível (%s) — usando priors.", exc)
    if e is None or c is None or v is None:
        e, c, v = _VIBE_FALLBACK.get(style, (0.6, 0.5, 0.5))

    def jit(x: float) -> float:
        return min(0.95, max(0.05, x + rng.gauss(0, 0.18)))

    p = dict(energy=jit(e), complexity=jit(c), valence=jit(v), source=src)
    log.info("IA: personalidade (%s) energia=%.2f complexidade=%.2f valência=%.2f",
             src, p["energy"], p["complexity"], p["valence"])
    return p


def _reference_bias(path: str, log: logging.Logger) -> Optional[Dict[str, Any]]:
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=22050, mono=True, duration=180.0)
        if y.size < sr:
            return None
        tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
        tempo = float(_np.atleast_1d(tempo)[0])
        if not (40 <= tempo <= 220):
            tempo = 120.0
        rms = float(_np.mean(librosa.feature.rms(y=y)))
        cent = float(_np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        energy = min(1.0, max(0.05, rms * 6.5))
        brightness = min(1.0, max(0.05, cent / 4000.0))
        return {"bpm": int(round(tempo)), "energy": round(energy, 3),
                "valence": round(0.3 + 0.5 * brightness, 3),
                "duration_s": round(len(y) / sr, 1)}
    except Exception as exc:
        log.warning("Análise da referência falhou (%s) — ignorada.", exc)
        return None


def _load_feedback() -> Dict[str, Any]:
    try:
        data = json.loads(_FEEDBACK_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"styles": {}}


def _save_feedback(fb: Dict[str, Any]) -> None:
    try:
        _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FEEDBACK_PATH.write_text(
            json.dumps(fb, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        LOG.debug("Falha ao salvar feedback: %s", exc)


def _feedback_style_state(style: str) -> Dict[str, Any]:
    fb = _load_feedback()
    return (fb.get("styles") or {}).get(style) or {}


def _repetitiveness_score(composer, beat_sec: float) -> float:
    bar_len = beat_sec * 4.0
    dup_w = div_w = wsum = 0.0
    for t in composer.tracks:
        notes = t.get("notes") or []
        if len(notes) < 8:
            continue
        w = 0.4 if _is_perc(t.get("instrument", "")) else 1.0
        bars: Dict[int, List[Tuple[int, int]]] = {}
        for n in notes:
            bar = int(n["start"] // bar_len)
            slot = int(round(((n["start"] % bar_len) / beat_sec) * 4))
            bars.setdefault(bar, []).append((slot, n["pitch"]))
        seen = set()
        dup = 0
        for bar in sorted(bars):
            sig = tuple(sorted(bars[bar]))
            if sig in seen:
                dup += 1
            else:
                seen.add(sig)
        dup_frac = dup / max(1, len(bars))
        uniq = len({n["pitch"] for n in notes})
        div = min(1.0, uniq / max(1, min(24, len(notes))))
        dup_w += w * dup_frac
        div_w += w * div
        wsum += w
    if wsum <= 0:
        return 0.5
    return float(min(1.0, max(0.0, 0.55 * (dup_w / wsum) + 0.45 * (1.0 - div_w / wsum))))


def _auto_evaluate_and_update(composer, style: str, beat_sec: float,
                              temperature: float, ai_on: bool,
                              log: logging.Logger) -> Optional[float]:
    try:
        score = _repetitiveness_score(composer, beat_sec)
    except Exception as exc:
        log.debug("Auto-avaliação falhou: %s", exc)
        return None
    log.info("Auto-avaliação: repetitividade=%.2f (0=variado, 1=repetitivo)", score)
    if not ai_on:
        return score
    try:
        fb = _load_feedback()
        styles = fb.setdefault("styles", {})
        st = styles.setdefault(style, {})
        prev = st.get("ema_rep")
        ema = score if prev is None else round(0.7 * float(prev) + 0.3 * score, 4)
        delta = float(st.get("temp_delta", 0.0))
        if ema > 0.55:
            delta = min(0.30, delta + 0.05)
        elif ema < 0.35:
            delta = max(-0.10, delta - 0.02)
        st.update({"ema_rep": ema, "temp_delta": round(delta, 3),
                   "n": int(st.get("n", 0)) + 1,
                   "last_score": round(score, 4),
                   "last_temperature": round(temperature, 3)})
        _save_feedback(fb)
        log.info("Feedback: EMA=%.2f | Δtemp=%+.2f", ema, delta)
    except Exception as exc:
        log.debug("Atualização de feedback falhou: %s", exc)
    return score


# =============================================================================
# v9: SOUNDFONT ENGINE — os 15 bancos do repositório como som principal
# =============================================================================

_SF_DIR = Path("soundfonts")
_SR_MIX = 44100
_SF_SCORES_PATH = Path("memory/sf2_scores.json")

# BANCO PRINCIPAL ("core") DE CADA PAPEL — troque o nome pelo EXATO do arquivo
# em soundfonts/ que você quer como padrão daquele papel.
SOUNDFONT_CORE: Dict[str, str] = {
    "lead":    "198_Juno106_LeadSynth.sf2",
    "bass":    "Crunk [moog]2007.with.sustain.sf2",
    "pad":     "HS Synth Collection I.sf2",
    "drums":   "Drum Set JD Rockset 5.sf2",
    "strings": "Crunk String.SF2",
}

_SF_SYNTH_HINTS = ("synth", "juno", "pro53", "flanger", "fatonic",
                   "syntlegend", "meinsynt", "electronic", "collection")


def _sf2_files() -> Dict[str, Path]:
    """{nome_minúsculo: caminho}. Case-insensitive (.sf2 e .SF2)."""
    out: Dict[str, Path] = {}
    if _SF_DIR.is_dir():
        for p in sorted(_SF_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() == ".sf2":
                out.setdefault(p.name.lower(), p)
    return out


def _gm_bank() -> Optional[Path]:
    for p in ("/usr/share/sounds/sf2/default-GM.sf2",
              "/usr/share/sounds/sf2/FluidR3_GM.sf2"):
        if Path(p).exists():
            return Path(p)
    return None


def _bank_role_of(inst_id: str) -> str:
    if inst_id == "drums":
        return "drums"
    if inst_id in ("bass", "synth_bass"):
        return "bass"
    if inst_id in ("synth_lead", "square_lead"):
        return "lead"
    if inst_id == "strings":
        return "strings"
    if inst_id == "synth_pad":
        return "pad"
    return "comp"


def _bank_name_score(inst_id: str, role: str, bl: str) -> int:
    """Pareamento banco↔trilha por NOME PARECIDO (e papel)."""
    if role == "drums":
        return 4 if ("drum" in bl or "darbuka" in bl or "giant" in bl) else -4
    if role == "bass":
        if "moog" in bl or "bass" in bl or "bandpass" in bl:
            return 4
        return -4 if ("drum" in bl or "darbuka" in bl) else 0
    if role == "lead":
        s = 4 if "lead" in bl else 0
        s += 1 if any(h in bl for h in _SF_SYNTH_HINTS) else 0
        return s - 4 if ("drum" in bl or "string" in bl) else s
    if role == "strings":
        return 5 if "string" in bl else -5
    # pad / comp: bancos de síntese servem para tudo sintético e também dão
    # timbre às trilhas harmônicas (pedido: usar os bancos do repo)
    if any(h in bl for h in _SF_SYNTH_HINTS):
        return 4 if inst_id == "synth_pad" else 2
    return -6 if ("drum" in bl or "darbuka" in bl) else 0


def _sf_learned() -> Dict[str, Any]:
    try:
        data = json.loads(_SF_SCORES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_sf_scores(data: Dict[str, Any]) -> None:
    try:
        _SF_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SF_SCORES_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        LOG.debug("Falha ao salvar sf2_scores: %s", exc)


def _role_metric(info: Optional[Dict[str, Any]], role: str) -> float:
    if not info or info.get("silent"):
        return -1.0
    if role == "drums":
        return float(info.get("drum", 0))
    if role == "bass":
        return 2.0 * float(info.get("low", 0)) + float(info.get("mel", 0))
    if role == "lead":
        return float(info.get("bright", 0)) + float(info.get("mel", 0))
    return float(info.get("mel", 0))


def _bank_for_track(inst_id: str, role: str, files: Dict[str, Path],
                    rng: random.Random, learned: Dict[str, Any]) -> Optional[Path]:
    """Escolhe o banco da trilha: nome + core + notas da audição + ROTAÇÃO
    (para que cada banco seja usado ao longo das gerações)."""
    banks = learned.get("banks") or {}
    scored: List[Tuple[float, Path]] = []
    for p in files.values():
        bl = p.name.lower()
        ns = _bank_name_score(inst_id, role, bl)
        ls = _role_metric(banks.get(bl), role)
        if ns <= 0 and ls <= 0:
            continue
        scored.append((2.0 * ns + ls, p))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    core = SOUNDFONT_CORE.get(role)
    if core:
        cp = files.get(core.lower())
        if cp is not None and any(p == cp for _, p in scored) and rng.random() < 0.55:
            return cp
    rot = learned.setdefault("rot", {})
    idx = int(rot.get(role, 0)) % min(3, len(scored))
    rot[role] = idx + 1
    return scored[idx][1]


# --- AUDIÇÃO DOS BANCOS ("treino" dos SoundFonts) ---------------------------

def _probe_midi(path: Path, drums: bool) -> None:
    import mido
    mid = mido.MidiFile(type=1, ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    mid.tracks.append(meta)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    if drums:
        tr.append(mido.Message("program_change", program=0, channel=9, time=0))
        for note in (36, 38, 42, 46, 49, 51, 41, 47, 56, 62):
            tr.append(mido.Message("note_on", note=note, velocity=115, channel=9, time=0))
            tr.append(mido.Message("note_off", note=note, velocity=0, channel=9, time=240))
    else:
        tr.append(mido.Message("program_change", program=0, channel=0, time=0))
        for pitch in range(48, 88, 2):
            tr.append(mido.Message("note_on", note=pitch, velocity=100, channel=0, time=0))
            tr.append(mido.Message("note_off", note=pitch, velocity=0, channel=0, time=360))
    mid.save(str(path))


def _bank_metrics(wav: Path) -> Optional[Dict[str, Any]]:
    try:
        data, sr = _wav_mono(wav)
    except Exception:
        return None
    if data is None or data.size < 512:
        return None
    rms = float(_np.sqrt(_np.mean(data.astype(_np.float64) ** 2)))
    seg = data[: 1 << 16]
    spec = _np.abs(_np.fft.rfft(seg.astype(_np.float64)))
    freqs = _np.fft.rfftfreq(len(seg), 1.0 / sr)
    tot = float(spec.sum()) or 1.0
    centroid = float((freqs * spec).sum() / tot)
    low = float(spec[freqs < 300].sum() / tot)
    return {"rms": round(rms, 5), "bright": round(min(1.0, centroid / 4000.0), 3),
            "low": round(low, 3)}


def run_audition(log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Testa TODOS os bancos: sonda cromática + sonda de bateria com cada um,
    mede RMS/brilho/graves, detecta mudos e salva o aprendizado."""
    log = log or LOG
    import shutil
    import tempfile
    fs = shutil.which("fluidsynth")
    files = _sf2_files()
    if not fs:
        log.error("FluidSynth não encontrado — instale para audicionar os bancos.")
        return {}
    if not files:
        log.error("Nenhum .sf2 em %s", _SF_DIR)
        return {}
    out: Dict[str, Any] = {"banks": {}}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mel_mid, drum_mid = td / "mel.mid", td / "drum.mid"
        _probe_midi(mel_mid, drums=False)
        _probe_midi(drum_mid, drums=True)
        log.info("Audição de %d bancos:", len(files))
        for name, bank in sorted(files.items()):
            info: Dict[str, Any] = {}
            w1 = td / (bank.stem + "_m.wav")
            w2 = td / (bank.stem + "_d.wav")
            m1 = _bank_metrics(w1) if _render_stem(fs, bank, mel_mid, w1) else None
            m2 = _bank_metrics(w2) if _render_stem(fs, bank, drum_mid, w2) else None
            mel = float((m1 or {}).get("rms", 0.0))
            drum = float((m2 or {}).get("rms", 0.0))
            info.update(m1 or {})
            info["mel"] = round(mel, 4)
            info["drum"] = round(drum, 4)
            info["silent"] = bool(mel < 1e-4 and drum < 1e-4)
            out["banks"][name] = info
            log.info("  %-40s mel=%.4f drum=%.4f brilho=%s%s", bank.name, mel, drum,
                     info.get("bright", "?"), "  ← MUDO" if info["silent"] else "")
    old = _sf_learned()
    out["rot"] = old.get("rot") or {}
    _save_sf_scores(out)
    log.info("Audição concluída → %s", _SF_SCORES_PATH)
    return out


# =============================================================================
# INTEGRAÇÃO com midi_composer.MidiComposer
# =============================================================================

def _safe_title(text: str) -> str:
    t = _strip_accents(str(text))
    t = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in t)
    return re.sub(r"\s+", " ", t).strip()


def _make_composer(params: "SongParams", log: logging.Logger) -> "MidiComposer":
    if MidiComposer is None:
        raise RuntimeError(
            "midi_composer.MidiComposer não pôde ser importado ("
            f"{_MIDI_COMPOSER_ERR or 'motivo desconhecido'}). "
            "Verifique real_instruments.py e mido (pip install -r requirements.txt)."
        )
    key_name = PC_TO_NAME[params.key_root]
    if "minor" in params.scale and "pentatonic" not in params.scale:
        scale_name = "minor"
    elif "pentatonic" in params.scale:
        scale_name = "pentatonic"
    else:
        scale_name = "major"
    title = _safe_title(params.prompt)[:48] or f"{params.style} em {key_name}"
    composer = MidiComposer(title=title, bpm=params.bpm, key=key_name,
                            scale=scale_name, style=params.style)
    log.info("Compositor: midi_composer.MidiComposer | %d BPM | %s %s",
             params.bpm, key_name, scale_name)
    return composer


def _export_midi(composer, path: Path, log: logging.Logger) -> bool:
    ok = composer.save_midi(path)
    if ok is False:
        log.error("save_midi() retornou False — mido está instalado? (pip install mido)")
        return False
    return True


def _midi_offset(mid) -> int:
    if mid.tracks:
        first = mid.tracks[0]
        if (not any(m.type == "note_on" and m.velocity > 0 for m in first)
                and any(m.type in ("set_tempo", "time_signature") for m in first)):
            return 1
    return 0


# =============================================================================
# RENDERIZAÇÃO — por trilha com os SEUS bancos; GM e procedural de fallback
# =============================================================================

_ROLE_GAIN = {"drums": 1.0, "perc": 0.9, "bass": 0.85, "lead": 0.95, "comp": 0.75}


def _stem_midi(src_mid, file_idx: int, out_path: Path, strip_programs: bool) -> bool:
    import mido
    if not (0 <= file_idx < len(src_mid.tracks)):
        return False
    tempo = 500000
    for t in src_mid.tracks:
        for m in t:
            if m.type == "set_tempo":
                tempo = m.tempo
                break
        else:
            continue
        break
    sub = mido.MidiFile(type=1, ticks_per_beat=src_mid.ticks_per_beat)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=tempo))
    sub.tracks.append(meta)
    tr = src_mid.tracks[file_idx]
    if strip_programs:  # banco especializado: usa o preset do próprio banco
        tr = mido.MidiTrack(m for m in tr if m.type != "program_change")
    sub.tracks.append(tr)
    sub.save(str(out_path))
    return True


def _render_stem(fs: str, bank: Path, midi_in: Path, wav_out: Path) -> bool:
    import subprocess
    try:
        subprocess.run([fs, "-ni", str(bank), str(midi_in), "-F", str(wav_out),
                        "-r", str(_SR_MIX), "-g", "0.8", "-R", "0", "-C", "0"],
                       check=True, capture_output=True, timeout=120)
    except Exception:
        return False
    return wav_out.exists() and wav_out.stat().st_size > 1000


def _wav_mono(path: Path):
    import soundfile as _sf
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def _render_per_track(midi_path: Path, wav_path: Path,
                      track_map: Dict[int, Tuple[str, str]], style: str,
                      log: logging.Logger) -> Optional[Path]:
    """Cada trilha com o banco CERTO dos seus .sf2; GM para o resto."""
    import shutil
    import tempfile
    if _np is None or not track_map:
        return None
    import mido
    fs = shutil.which("fluidsynth")
    gm = _gm_bank()
    files = _sf2_files()
    if not fs or gm is None:
        return None
    learned = _sf_learned()
    rng = random.Random()
    mid = mido.MidiFile(str(midi_path))

    stems: List[Tuple[Any, float]] = []
    used_banks: List[str] = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for file_idx, (inst, role) in sorted(track_map.items()):
            if not (0 <= file_idx < len(mid.tracks)):
                continue
            bank_role = _bank_role_of(inst)
            if _GM_INSTRUMENTS and bank_role == "comp":
                bank = None  # modo acústico: comp com GM
            else:
                bank = _bank_for_track(inst, bank_role, files, rng, learned) if files else None
            mono = None
            if bank:
                stem_mid, wav = td / f"t{file_idx}.mid", td / f"t{file_idx}.wav"
                if _stem_midi(mid, file_idx, stem_mid, strip_programs=True) \
                        and _render_stem(fs, bank, stem_mid, wav):
                    try:
                        m, _ = _wav_mono(wav)
                        if m.size and float(_np.max(_np.abs(m))) > 1e-3:
                            mono = m
                            used_banks.append(bank.name)
                            log.info("  trilha %-14s ← %s", inst, bank.name)
                    except Exception:
                        pass
            if mono is None:  # sem banco, ou banco mudo → GM
                stem_gm, wav = td / f"t{file_idx}_gm.mid", td / f"t{file_idx}_gm.wav"
                if _stem_midi(mid, file_idx, stem_gm, strip_programs=False) \
                        and _render_stem(fs, gm, stem_gm, wav):
                    try:
                        m, _ = _wav_mono(wav)
                        if m.size and float(_np.max(_np.abs(m))) > 1e-4:
                            mono = m
                            log.info("  trilha %-14s ← GM%s", inst,
                                     " (banco ficou mudo)" if bank else "")
                    except Exception:
                        pass
            if mono is not None:
                stems.append((mono, _ROLE_GAIN.get(role, 0.8)))
            else:
                log.warning("  trilha %s não renderizou", inst)

    if "rot" in learned:  # persiste a rotação (cada banco é usado com o tempo)
        _save_sf_scores(learned)
    if not stems:
        return None
    n = max(len(m) for m, _ in stems)
    mix = _np.zeros(n, dtype=_np.float64)
    for m, g in stems:
        mix[: len(m)] += m.astype(_np.float64) * g
    peak = float(_np.max(_np.abs(mix)))
    if peak > 0:
        mix *= 0.86 / peak
    import soundfile as _sf
    _sf.write(str(wav_path), mix.astype(_np.float32), _SR_MIX)
    log.info("WAV: %d trilhas | bancos usados: %s → %s",
             len(stems), ", ".join(dict.fromkeys(used_banks)) or "GM", wav_path)
    return wav_path


def _try_fluidsynth_cli(midi_path: Path, wav_path: Path, log: logging.Logger,
                        forced: Optional[str] = None) -> Optional[Path]:
    import shutil
    import subprocess
    fs = shutil.which("fluidsynth")
    if not fs:
        return None
    bank: Optional[Path] = None
    if forced:
        p = Path(forced)
        if not p.exists():
            p = _sf2_files().get(forced.lower(), _SF_DIR / forced)
        if p.exists():
            bank = p
        else:
            log.warning("--soundfont %r não encontrado — usando GM.", forced)
    if bank is None:
        bank = _gm_bank()  # full-band SOMENTE com GM completo
    if bank is None:
        return None
    try:
        subprocess.run([fs, "-ni", str(bank), str(midi_path), "-F", str(wav_path),
                        "-r", str(_SR_MIX), "-g", "0.7"],
                       check=True, capture_output=True, timeout=300)
        if wav_path.exists() and wav_path.stat().st_size > 1000:
            log.info("WAV: FluidSynth CLI + %s", bank.name)
            return wav_path
    except Exception as exc:
        log.warning("FluidSynth CLI falhou (%s).", exc)
    return None


# --- renderizador embutido (procedural — SÓ fallback) -----------------------

_SR = 22050


def _synth_drum(pitch: int, dur: float, vel: int, seed: int = 0):
    np = _np
    rng = np.random.default_rng((pitch + 1) * 7919 + seed)
    dur = float(min(max(dur, 0.02), 2.5))
    n = max(8, int(_SR * dur))
    t = np.arange(n) / _SR
    amp = max(vel, 1) / 127.0
    noise = rng.uniform(-1.0, 1.0, n)
    hi = np.diff(noise, prepend=noise[:1])

    def decay(tc: float):
        return np.exp(-t / max(tc, 1e-3))

    if pitch in (35, 36):
        f = 44 + 90 * np.exp(-t * 30)
        w = np.sin(2 * np.pi * np.cumsum(f) / _SR)
        out = (w + 0.3 * hi * decay(0.004)) * decay(0.11)
    elif pitch in (38, 40):
        out = (0.65 * noise * decay(0.075)
               + 0.45 * np.sin(2 * np.pi * 185 * t) * decay(0.05)
               + 0.25 * hi * decay(0.05))
    elif pitch == 37:
        out = (np.sin(2 * np.pi * 1700 * t) + 0.6 * hi) * decay(0.012) * 0.8
    elif pitch == 39:
        out = np.zeros(n)
        for off in (0.0, 0.011, 0.023):
            i0 = int(off * _SR)
            if i0 < n:
                out[i0:] += noise[i0:] * np.exp(-t[: n - i0] / 0.055)
        out *= 0.55
    elif pitch in (42, 44, 46):
        out = hi * decay({42: 0.030, 44: 0.050, 46: 0.300}[pitch]) * 1.4
    elif pitch in (49, 57):
        out = (0.7 * hi + 0.5 * noise) * decay(0.9) * 0.9
    elif pitch in (51, 53, 59):
        bell = np.sin(2 * np.pi * 1050 * t) + 0.5 * np.sin(2 * np.pi * 1570 * t)
        out = 0.5 * hi * decay(0.45) + 0.12 * bell * decay(0.35)
    elif pitch in (41, 43, 45, 47, 48, 50):
        base = {41: 85, 43: 105, 45: 125, 47: 155, 48: 185, 50: 215}[pitch]
        f = base * (1 + 0.35 * np.exp(-t * 18))
        w = np.sin(2 * np.pi * np.cumsum(f) / _SR)
        out = (w + 0.15 * hi * decay(0.01)) * decay(0.22)
    elif pitch == 54:
        jingle = np.sin(2 * np.pi * 5400 * t) + 0.7 * np.sin(2 * np.pi * 7300 * t)
        out = 0.5 * hi * decay(0.05) + 0.25 * jingle * decay(0.10)
    elif pitch == 56:
        out = (np.sin(2 * np.pi * 555 * t) + 0.7 * np.sin(2 * np.pi * 835 * t)) * decay(0.13)
    elif pitch in (61, 62, 63, 64):
        base = {61: 190, 62: 255, 63: 300, 64: 165}[pitch]
        out = (np.sin(2 * np.pi * base * t) + 0.25 * noise * decay(0.01)) * decay(0.11)
    elif pitch in (65, 66):
        out = np.sin(2 * np.pi * 310 * t) * decay(0.2) + 0.2 * hi * decay(0.02)
    elif pitch in (67, 68):
        base = 640 if pitch == 67 else 490
        out = np.sin(2 * np.pi * base * t) * decay(0.085) + 0.15 * hi * decay(0.006)
    elif pitch in (69, 70):
        out = hi * np.minimum(t / 0.008, 1.0) * decay(0.06)
    elif pitch == 74:
        out = noise * decay(0.035) * np.sign(np.sin(2 * np.pi * 9 * t))
    elif pitch == 75:
        out = np.sin(2 * np.pi * 1720 * t) * decay(0.016)
    elif pitch in (76, 77):
        base = 820 if pitch == 76 else 610
        out = (np.sin(2 * np.pi * base * t)
               + 0.3 * np.sin(2 * np.pi * base * 2.76 * t)) * decay(0.028)
    elif pitch == 78:
        f = 480 - 320 * np.clip(t / max(dur, 0.06), 0, 1)
        out = (np.sin(2 * np.pi * np.cumsum(f) / _SR) * np.minimum(t / 0.02, 1.0)
               * decay(max(dur * 0.7, 0.06)))
    elif pitch == 80:
        out = (np.sin(2 * np.pi * 4100 * t)
               + 0.6 * np.sin(2 * np.pi * 6170 * t)) * decay(0.6) * 0.5
    else:
        out = hi * decay(0.05)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out = out * (0.55 * amp / peak)
    return out


def _synth_melodic(program: int, pitch: int, dur: float, vel: int):
    np = _np
    freq = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
    dur = float(min(max(dur, 0.05), 8.0))
    rel = 0.09
    n = int(_SR * (dur + rel))
    t = np.arange(n) / _SR
    amp = max(vel, 1) / 127.0
    attack = np.minimum(t / 0.012, 1.0)
    if program < 8 or 24 <= program < 32 or 104 <= program < 112:
        env = np.exp(-t / max(dur * 0.45, 0.09)) * attack
        wave = (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t)
                + 0.12 * np.sin(6 * np.pi * freq * t))
        gain = 0.30
    elif 32 <= program < 40:
        env = np.exp(-t / max(dur * 0.8, 0.12)) * attack
        wave = (np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(4 * np.pi * freq * t)
                + 0.08 * np.sin(6 * np.pi * freq * t))
        gain = 0.34
    else:
        env = np.ones(n)
        a = int(0.025 * _SR)
        env[:a] = np.linspace(0, 1, a)
        r = int(rel * _SR)
        env[n - r:] *= np.linspace(1, 0, r)
        wave = (np.sin(2 * np.pi * freq * t) + 0.45 * np.sin(4 * np.pi * freq * t)
                + 0.28 * np.sin(6 * np.pi * freq * t) + 0.15 * np.sin(8 * np.pi * freq * t))
        gain = 0.22
    return wave * env * amp * gain


def _builtin_render_wav(midi_path: Path, wav_path: Path, log: logging.Logger) -> Optional[Path]:
    if _np is None:
        log.warning("numpy ausente — WAV embutido indisponível.")
        return None
    try:
        import mido
        import soundfile as _sf
    except ImportError as exc:
        log.warning("WAV embutido precisa de mido + soundfile (%s).", exc)
        return None

    mid = mido.MidiFile(str(midi_path))
    tempo = 500000
    done = False
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "set_tempo":
                tempo, done = msg.tempo, True
                break
        if done:
            break
    spt = tempo / 1e6 / mid.ticks_per_beat
    n = int((mid.length + 1.2) * _SR)
    audio = _np.zeros(n, dtype=_np.float64)
    count = 0

    def _emit(start, end, pitch, vel, ch, programs):
        nonlocal count
        dur = max(0.03, end - start)
        if ch == 9:
            w = _synth_drum(pitch, dur, vel, seed=int(start * 997))
        else:
            w = _synth_melodic(programs.get(ch, 0), pitch, dur, vel)
        i0 = int(start * _SR)
        i1 = min(n, i0 + len(w))
        if i0 < n:
            audio[i0:i1] += w[: i1 - i0]
            count += 1

    for track in mid.tracks:
        t = 0.0
        programs: Dict[int, int] = {}
        active: Dict[int, List[Tuple[float, int, int]]] = {}
        for msg in track:
            t += msg.time * spt
            if msg.type == "program_change":
                programs[msg.channel] = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                active.setdefault(msg.note, []).append((t, msg.velocity, msg.channel))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                lst = active.get(msg.note)
                if lst:
                    start, vel, ch = lst.pop(0)
                    _emit(start, t, msg.note, vel, ch, programs)
        for note_id, lst in active.items():
            for start, vel, ch in lst:
                _emit(start, start + 0.35, note_id, vel, ch, programs)

    if count == 0:
        log.warning("WAV embutido: nenhuma nota encontrada.")
        return None
    peak = float(_np.max(_np.abs(audio)))
    if peak > 0:
        audio *= 0.86 / peak
    _sf.write(str(wav_path), audio.astype(_np.float32), _SR)
    log.info("WAV embutido: %d notas sintetizadas → %s", count, wav_path)
    return wav_path


def _render_audio(midi_path: Path, wav_path: Path, prefer_soundfont: bool,
                  log: logging.Logger, track_map: Optional[Dict[int, Tuple[str, str]]] = None,
                  style: str = "", forced_soundfont: Optional[str] = None) -> Optional[Path]:
    if prefer_soundfont:
        # 1) POR TRILHA — os SEUS bancos (método principal)
        if track_map and not forced_soundfont:
            try:
                if _render_per_track(midi_path, wav_path, track_map, style, log):
                    return wav_path
            except Exception as exc:
                log.warning("Render por trilha falhou (%s) — full-band.", exc)
        # 2) FULL-BAND — GM completo (ou banco forçado pelo usuário)
        if _try_fluidsynth_cli(midi_path, wav_path, log, forced=forced_soundfont):
            return wav_path
        # 3) soundfont_renderer (compatibilidade)
        if _RENDERER is not None:
            import inspect
            try:
                sig = inspect.signature(_RENDERER)
                kw: Dict[str, Any] = {}
                for name in ("use_soundfont", "prefer_soundfont"):
                    if name in sig.parameters:
                        kw[name] = True
                _RENDERER(str(midi_path), str(wav_path), **kw)
                if wav_path.exists():
                    log.info("WAV: soundfont_renderer")
                    return wav_path
            except Exception as exc:
                log.warning("soundfont_renderer falhou (%s) — caindo.", exc)
    # 4) Embutido — procedural, APENAS fallback
    try:
        if _builtin_render_wav(midi_path, wav_path, log):
            return wav_path
    except Exception as exc:
        log.warning("Renderização embutida falhou: %s", exc)
    return None


# =============================================================================
# MIDI IMPORTADO (CLI)
# =============================================================================

def _inst_from_program(prog: int, is_drum: bool) -> str:
    if is_drum:
        return "drums"
    if prog >= 88:
        return "synth_pad"
    if 80 <= prog < 88:
        return "synth_lead"
    if 72 <= prog < 80:
        return "flute"
    if 64 <= prog < 72:
        return "saxophone"
    if 56 <= prog < 64:
        return "trumpet"
    if 40 <= prog < 56:
        return "strings"
    if 32 <= prog < 40:
        return "bass"
    if 24 <= prog < 32:
        return "electric_guitar"
    if 16 <= prog < 24:
        return "organ"
    return "piano"


def analyze_midi_tracks(midi_path: Path) -> Dict[int, Tuple[str, str]]:
    import mido
    mid = mido.MidiFile(str(midi_path))
    track_map: Dict[int, Tuple[str, str]] = {}
    for idx, tr in enumerate(mid.tracks):
        notes = [m for m in tr if m.type == "note_on" and m.velocity > 0]
        if not notes:
            continue
        chans = {m.channel for m in notes}
        is_drum = 9 in chans
        prog = 0
        for m in tr:
            if m.type == "program_change" and not is_drum:
                prog = m.program
                break
        inst = _inst_from_program(prog, is_drum)
        role = ("drums" if inst == "drums" else "bass" if inst in ("bass", "synth_bass")
                else "lead" if _monophonic(inst) or "lead" in inst else "comp")
        track_map[idx] = (inst, role)
    return track_map


def render_midi_file(midi_path: Path, out_dir: Path, style: str = "",
                     log: Optional[logging.Logger] = None) -> Optional[Path]:
    log = log or LOG
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / (Path(midi_path).stem + ".wav")
    track_map = analyze_midi_tracks(Path(midi_path))
    log.info("MIDI importado: %d trilhas com notas", len(track_map))
    return _render_audio(Path(midi_path), wav_path, prefer_soundfont=True, log=log,
                        track_map=track_map, style=style)


def audit_midi(path: Path, inst_by_track: Dict[int, str], log: logging.Logger) -> bool:
    try:
        import mido
    except ImportError:
        log.warning("mido não instalado — auditoria de MIDI pulada.")
        return True
    mid = mido.MidiFile(str(path))
    offset = _midi_offset(mid)
    ok = True
    drums_found = False
    log.info("Auditoria do MIDI (%d tracks no arquivo, %d de instrumento):",
             len(mid.tracks), len(mid.tracks) - offset)
    for i, track in enumerate(mid.tracks):
        if i < offset:
            log.info("  [track %d] (tempo/meta — sem notas, normal)", i)
            continue
        notes = [m for m in track if m.type == "note_on" and m.velocity > 0]
        chans = sorted({m.channel for m in notes})
        progs = [m.program for m in track if m.type == "program_change"]
        inst = inst_by_track.get(i - offset, "?")
        if not notes:
            log.warning("  [track %d] %-16s → 0 NOTAS (algo não foi gerado!)", i, inst)
            ok = False
            continue
        log.info("  [track %d] %-16s → %d notas | canais %s | programa %s",
                 i, inst, len(notes), chans, progs[:1] or "-")
        if 9 in chans:
            drums_found = True
        if _is_perc(inst) and chans and 9 not in chans:
            log.warning("  [track %d] %s é percussão mas está no canal %s — GM exige "
                        "canal 9.", i, inst, chans)
            ok = False
    if drums_found:
        log.info("BATERIA CONFIRMADA no MIDI: notas no canal 9 (percussão GM).")
    else:
        log.warning("Nenhuma nota no canal 9 — bateria ausente neste MIDI.")
        ok = False
    return ok


# =============================================================================
# PARÂMETROS E GERAÇÃO
# =============================================================================

@dataclass
class SongParams:
    prompt: str = ""
    style: str = "pop"
    bpm: int = 120
    key_root: int = 0
    scale: str = "major"
    instruments: List[str] = field(default_factory=list)
    duration: float = 30.0
    seed: int = 0
    temperature: float = 0.7
    output_dir: Path = Path("song_output")
    prefer_soundfont: bool = True
    bpm_explicit: bool = False
    use_ai: bool = True
    forced_soundfont: Optional[str] = None
    reference: Optional[str] = None


def _fit_form(bars: int, style: str) -> List[Tuple[str, int]]:
    form = list(SONG_FORMS.get(style, _DEFAULT_FORM))
    out: List[Tuple[str, int]] = []
    remaining = bars
    for name, b in form:
        if remaining <= 0:
            break
        take = min(b, remaining)
        out.append((name, take))
        remaining -= take
    toggle, cycle = True, 0
    while remaining >= 4:
        if cycle % 3 == 2 and remaining >= 8:
            out.append(("ponte", 4))
            out.append(("refrao", 4))
            remaining -= 8
        else:
            out.append(("verso" if toggle else "refrao", 4))
            toggle = not toggle
            remaining -= 4
        cycle += 1
    if remaining > 0:
        out.append(("refrao", remaining))
    return out


def _transpose_plan(sections: List[Tuple[str, int]],
                    rng: random.Random) -> List[int]:
    """v9: modulação por seção — a música muda de tom (e não 'toca igual').
    Ponte → tonalidade relativa; último refrão → +2 semitons (truck driver);
    versos posteriores podem subir para a relativa maior."""
    plan: List[int] = []
    total_chorus = sum(1 for n, _ in sections if n == "refrao")
    ci = 0
    for name, _ in sections:
        tr = 0
        if name == "ponte":
            tr = rng.choice((-3, 5, -5))
        elif name == "verso" and rng.random() < 0.25:
            tr = 5
        elif name == "refrao":
            ci += 1
            if ci == total_chorus and total_chorus >= 2 and rng.random() < 0.8:
                tr = 2
        plan.append(tr)
    return plan


def generate_song(params: SongParams, log: logging.Logger = LOG) -> Dict[str, Any]:
    rng = random.Random(params.seed)
    style = params.style
    style_cfg = STYLES.get(style, STYLES["pop"])
    ai_on = bool(params.use_ai)

    # temperatura adaptativa (feedback aprendido)
    temperature = params.temperature
    if ai_on:
        delta = float(_feedback_style_state(style).get("temp_delta", 0.0))
        if delta:
            temperature = float(min(1.0, max(0.05, params.temperature + delta)))
            log.info("IA (feedback): temperatura %.2f → %.2f (Δ%+.2f, aprendido)",
                     params.temperature, temperature, delta)

    # BPM do dataset real
    bpm_source = "style_default"
    if ai_on and not params.bpm_explicit:
        kb = _kb_bpm(style, log)
        if kb:
            mean, std = kb
            lo, hi = style_cfg["bpm"]
            bpm = int(round(mean + rng.gauss(0, max(2.0, min(12.0, std or 6.0)) * 0.6)))
            params.bpm = int(min(hi + 15, max(lo - 12, bpm)))
            bpm_source = "knowledge_base"
            log.info("IA (dataset): BPM → %d (média real do gênero: %d)",
                     params.bpm, int(mean))

    beat_sec = 60.0 / params.bpm
    beats_per_bar = 4
    scale = SCALES[params.scale]
    root_midi = 48 + params.key_root
    swing_on = style_cfg["swing"] > 0

    # personalidade + referência
    personality = _ai_personality(style, rng, log, use_ae=ai_on)
    ref_info: Optional[Dict[str, Any]] = None
    if params.reference:
        rb = _reference_bias(params.reference, log)
        if rb:
            ref_info = rb
            if ai_on:
                if not params.bpm_explicit and rb.get("bpm"):
                    params.bpm = int(rb["bpm"])
                    bpm_source = "reference"
                    beat_sec = 60.0 / params.bpm
                personality["energy"] = rb["energy"]
                personality["valence"] = rb["valence"]
                personality["source"] = "reference"
                log.info("IA (referência): BPM → %d | energia=%.2f (%s)",
                         params.bpm, rb["energy"], params.reference)

    vel_scale = 0.78 + 0.45 * personality["energy"]
    p_rhythm_change = 0.15 + 0.30 * personality["complexity"]
    p_bass_fill = 0.25 + 0.35 * personality["complexity"]
    up_bias = 0.35 + 0.30 * personality["valence"]

    # instrumentação
    inst_ids: List[str] = list(params.instruments)
    if not inst_ids:
        for cand in style_cfg["instruments"]:
            rid = _resolve_id(cand)
            if rid and rid not in inst_ids:
                inst_ids.append(rid)
        log.info("Instrumentação padrão do estilo '%s': %s", style, inst_ids)

    perc_ids = [i for i in inst_ids if _is_perc(i)]
    mel_ids = [i for i in inst_ids if i not in perc_ids]
    bass_id = next((i for i in mel_ids if "bass" in i), None)
    comp_ids = [i for i in mel_ids if i != bass_id]

    kit_id: Optional[str] = None
    if "drums" in perc_ids:
        kit_id = "drums"
        perc_ids = [i for i in perc_ids if i != "drums"]
    elif style not in NO_DRUMS_STYLES and not perc_ids:
        kit_id = _resolve_id(("drums", "bateria"))
        if kit_id:
            inst_ids.append(kit_id)
            log.info("Bateria (kit) adicionada automaticamente para o estilo '%s'.", style)

    if bass_id is None and style in AUTO_BASS_STYLES:
        bass_id = _resolve_id(("bass", "baixo"))
        if bass_id:
            inst_ids.append(bass_id)
            log.info("Baixo adicionado automaticamente para o estilo '%s'.", style)

    if not comp_ids:
        for cand in style_cfg["instruments"]:
            rid = _resolve_id(cand)
            if rid and not _is_perc(rid) and "bass" not in rid:
                comp_ids.append(rid)
                inst_ids.append(rid)
                break
    if not comp_ids:
        for cand in ("piano", "acoustic_guitar", "organ", "strings"):
            rid = _resolve_id((cand,))
            if rid:
                comp_ids.append(rid)
                inst_ids.append(rid)
                break

    mono_lead = next((i for i in comp_ids if _monophonic(i)), None)
    if mono_lead:
        lead_id = mono_lead
        comp_ids = [i for i in comp_ids if i != mono_lead]
    else:
        lead_id = comp_ids[0] if comp_ids else None
        if len(comp_ids) > 1:
            comp_ids = comp_ids[1:]
    log.info("Instrumentos: lead=%s | acordes=%s | baixo=%s | kit=%s | percussão=%s",
             lead_id, comp_ids or "-", bass_id, kit_id, perc_ids or "-")

    # canais únicos
    used_ch = {9}
    for inst in dict.fromkeys(inst_ids):
        entry = INSTRUMENTS.get(inst)
        if not entry or entry.get("is_percussion"):
            continue
        if entry.get("channel") in used_ch:
            for ch in range(16):
                if ch not in used_ch:
                    entry["channel"] = ch
                    break
        used_ch.add(entry["channel"])

    composer = _make_composer(params, log)
    track_of: Dict[str, int] = {}
    inst_by_track: Dict[int, str] = {}
    for inst in dict.fromkeys(inst_ids):
        try:
            idx = composer.add_track(inst)
        except Exception as exc:
            log.warning("add_track(%r) falhou (%s) — instrumento ignorado.", inst, exc)
            continue
        if not isinstance(idx, int):
            log.warning("add_track(%r) sem índice — instrumento ignorado.", inst)
            continue
        track_of[inst] = idx
        inst_by_track[idx] = inst

    def note(track: int, bar: int, pos: float, dur_beats: float,
             pitch: int, vel: int, swing: bool = False) -> None:
        p = bar * beats_per_bar + pos
        if swing and abs((p % 1.0) - 0.5) < 1e-6:
            p += style_cfg["swing"]
        jitter = rng.uniform(-0.012, 0.012) if temperature > 0 else 0.0
        start = max(0.0, p + jitter) * beat_sec
        dur = max(0.03, dur_beats * beat_sec * 0.96)
        strong = (pos == int(pos)) and (int(pos) % 2 == 0)
        v = int(min(127, max(20, vel + rng.randint(-6, 6) + (6 if strong else 0))))
        pit = int(round(pitch))
        if not (0 <= pit <= 127):
            log.warning("Pitch fora de 0..127 (%r) em bar=%d pos=%.2f — corrigido",
                        pitch, bar, pos)
            pit = min(127, max(21, pit))
        composer.add_note(track, start=start, duration=dur, pitch=pit, velocity=v)

    # plano: forma + modulação (v9)
    bars = max(4, int(round(params.duration / (beat_sec * beats_per_bar))))
    sections = _fit_form(bars, style)
    sec_tr = _transpose_plan(sections, rng)
    total_bars = sum(b for _, b in sections)
    log.info("Estrutura: %s", " → ".join(
        f"{n}:{b}{'' if not t else f'({t:+d})'}" for (n, b), t in zip(sections, sec_tr)))

    opts = PROGRESSIONS.get(style, PROGRESSIONS["pop"])
    prog_a = list(rng.choice(opts))
    rest_opts = [o for o in opts if o != prog_a]
    prog_a2 = list(rng.choice(rest_opts)) if rest_opts else prog_a
    rest2 = [o for o in rest_opts if o != prog_a2] or rest_opts or opts
    prog_b = list(rng.choice(rest2))
    add7 = bool(style_cfg.get("sevenths")) or rng.random() < (0.15 + 0.30 * personality["complexity"])
    log.info("Progressões: A=%s | A2=%s | B=%s | tom=%s %s | %d BPM | seed=%d",
             prog_a, prog_a2, prog_b, PC_TO_NAME[params.key_root], params.scale,
             params.bpm, params.seed)

    prev_voicing: Optional[List[int]] = None
    phrase: Optional[Dict[int, list]] = None
    prev_phrase: Optional[Dict[int, list]] = None
    verse_idx = 0
    chorus_count = 0
    abs_bar = 0
    n = len(scale)
    for sec_i, (sec_name, sec_bars) in enumerate(sections):
        profile = dict(SECTION_PROFILES.get(sec_name, SECTION_PROFILES["verso"]))
        if sec_name == "refrao":
            chorus_count += 1
            profile["vel"] = min(110, profile["vel"] + 3 * (chorus_count - 1))
        profile["vel"] = int(min(115, profile["vel"] * vel_scale))
        sec_root = root_midi + sec_tr[sec_i]        # v9: tom da seção
        comp_rhythms = COMP_RHYTHMS.get(style, [[0.0, 2.0]])
        comp_rhythm = rng.choice(comp_rhythms)
        if sec_name in ("refrao", "solo"):
            prog = prog_b
        elif sec_name == "verso":
            prog = prog_a if verse_idx % 2 == 0 else prog_a2
            verse_idx += 1
        elif sec_name == "ponte":
            prog = prog_a2 if prog_a2 != prog_a else prog_b
        else:
            prog = prog_a
        if profile["melody"] and lead_id and lead_id in track_of:
            phrase = _new_phrase(rng, profile["dense"], up_bias, prev=prev_phrase)
            prev_phrase = phrase
        next_is_final_chorus = (sec_i + 1 < len(sections)
                                and sections[sec_i + 1][0] == "refrao"
                                and sec_i + 2 >= len(sections) - 1)
        for b in range(sec_bars):
            # v9: arco dinâmico — cresce ao longo da música; fade no outro
            arc = 0.86 + 0.28 * (abs_bar / max(1, total_bars - 1))
            if sec_name == "outro":
                arc *= max(0.55, 1 - 0.18 * (b / max(1, sec_bars - 1)))
            bar_vel = int(min(115, profile["vel"] * arc))
            # v9: breakdown — compasso de vácuo antes do refrão final
            breakdown = (next_is_final_chorus and b == sec_bars - 1 and abs_bar > 8)
            # v9: dropout — 1 compasso esparso a cada ~16
            dropout = (abs_bar % 16 == 15 and rng.random() < 0.25)

            if b > 0 and len(comp_rhythms) > 1 and rng.random() < p_rhythm_change:
                comp_rhythm = rng.choice(
                    [r for r in comp_rhythms if r != comp_rhythm] or comp_rhythms)

            degree = prog[b % len(prog)]
            next_degree = prog[(b + 1) % len(prog)]
            is_last = (b == sec_bars - 1)
            if (len(prog) == 4 and len(scale) == 7 and style != "blues"
                    and abs_bar % 8 == 7 and not is_last):
                degree = 4
                next_degree = prog[0]

            # ---- acordes (comp) ----
            if profile["comp"] and comp_ids:
                offsets = _chord_offsets(scale, degree, add7)
                prev_voicing = _voice_chord(prev_voicing, offsets, sec_root + 12)
                for i, hpos in enumerate(comp_rhythm):
                    nxt = comp_rhythm[i + 1] if i + 1 < len(comp_rhythm) else beats_per_bar
                    dur = max(0.25, nxt - hpos - 0.05)
                    for inst in comp_ids:
                        if inst not in track_of:
                            continue
                        strum = any(f in inst for f in STRUM_FAMILIES)
                        for j, pit in enumerate(prev_voicing):
                            off = j * 0.018 if strum else 0.0
                            note(track_of[inst], abs_bar, hpos + off, dur,
                                 pit, bar_vel - 3 * j, swing=swing_on)

            # ---- baixo ----
            if profile["bass"] and bass_id and bass_id in track_of and not breakdown:
                use_fill = ((b % 4 == 3 or is_last) and rng.random() < p_bass_fill
                            and style not in ("ambient", "dungeon", "cinematic", "classical"))
                if use_fill:
                    pat = [(0.0, "root", 0.9), (1.0, "third", 0.9),
                           (2.0, "fifth", 0.9), (3.0, "approach", 0.9)]
                else:
                    pat = BASS_PATTERNS.get(style, BASS_PATTERNS["pop"])
                root_abs = sec_root + scale[degree % n]
                next_root_abs = sec_root + scale[next_degree % n]
                for pos, kind, dur in pat:
                    pitch = _bass_pitch(kind, root_abs, next_root_abs, scale, degree, style)
                    note(track_of[bass_id], abs_bar, pos, dur, pitch,
                         bar_vel - 8, swing=swing_on)

            # ---- bateria (kit) ----
            if kit_id and kit_id in track_of and profile["drums"] and not breakdown:
                pattern = _kit_pattern(style, abs_bar, rng)
                if dropout:  # v9: só ataques fortes
                    pattern = [h for h in pattern if float(h[0]).is_integer()]
                if is_last and sec_i < len(sections) - 1:
                    pattern = [h for h in pattern if h[0] < 2.0] + _fill(style, rng)
                if b == 0 and sec_name in ("refrao", "solo") and style in CRASH_STYLES:
                    pattern.append((0.0, "crash", 94))
                if rng.random() < 0.18 * (0.6 + 0.8 * personality["energy"]) and style in (
                        "rock", "pop", "funk", "hiphop", "blues", "metal", "bossfight", "dnb"):
                    pattern.append((rng.choice([0.75, 1.5, 2.5, 3.25]), "snare", 30))
                if rng.random() < 0.12 * (0.6 + 0.8 * personality["energy"]) and style in (
                        "rock", "pop", "funk", "disco", "synthwave", "electronic", "bossfight"):
                    pattern.append((rng.choice([0.5, 1.5, 2.5, 3.5]), "hh_open", 60))
                gain = bar_vel / 76.0
                for pos, name, vel in pattern:
                    dur_beats = DRUM_DUR.get(name, DEFAULT_DRUM_DUR) / beat_sec
                    note(track_of[kit_id], abs_bar, pos, dur_beats,
                         GM[name], int(vel * gain), swing=swing_on)

            # ---- percussão individual ----
            for pid in perc_ids:
                if pid not in track_of or not profile["drums"] or breakdown:
                    continue
                vels, alt = PERC_16.get(pid, _DEFAULT_PERC_16)
                hi = _perc_pitch(pid)
                lo = 68 if pid == "agogo" else hi
                dur = 0.3 if pid == "surdo" else DEFAULT_DRUM_DUR
                hit = 0
                for i, v in enumerate(vels):
                    if v <= 0:
                        continue
                    pitch = hi if (not alt or hit % 2 == 0) else lo
                    hit += 1
                    note(track_of[pid], abs_bar, i * 0.25, dur / beat_sec,
                         pitch, int(v * (bar_vel / 76.0)))

            # ---- melodia (frases com DESENVOLVIMENTO temático) ----
            if profile["melody"] and lead_id and lead_id in track_of:
                if b % 4 == 0 and b > 0:
                    r = rng.random()
                    if r < 0.55 or phrase is None:
                        phrase = _new_phrase(rng, profile["dense"], up_bias, prev=prev_phrase)
                        prev_phrase = phrase
                    elif r < 0.85:
                        phrase = _vary_phrase(phrase, rng)
                cell = phrase.get(b % 4) if phrase else None
                if cell:
                    for (pos, dur), step in cell:
                        idx = degree + step
                        if pos in (0.0, 2.0):
                            idx = _snap_to_chord(idx, degree, n)
                        pitch = max(58, min(86, _scale_pitch(sec_root + 12, scale, idx)))
                        note(track_of[lead_id], abs_bar, pos, dur,
                             pitch, bar_vel + 4, swing=swing_on)
            abs_bar += 1

    score = _auto_evaluate_and_update(composer, style, beat_sec, temperature,
                                      ai_on, log)

    # exportação
    out_dir = params.output_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{style}_s{params.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_path = out_dir / f"musica_{style}_s{params.seed}.mid"
    if not _export_midi(composer, midi_path, log):
        raise RuntimeError("Falha ao exportar o MIDI.")
    audit_midi(midi_path, inst_by_track, log)

    try:
        composer.save_json(out_dir / "composicao.json")
    except Exception as exc:
        log.debug("save_json() indisponível: %s", exc)

    # track_map com índices DO ARQUIVO (offset da track de tempo)
    import mido as _mido
    _mid = _mido.MidiFile(str(midi_path))
    _off = _midi_offset(_mid)
    track_map: Dict[int, Tuple[str, str]] = {}
    for inst, ti in track_of.items():
        role = ("drums" if inst == kit_id else "perc" if inst in perc_ids
                else "bass" if inst == bass_id else "lead" if inst == lead_id
                else "comp")
        track_map[ti + _off] = (inst, role)

    wav_path = _render_audio(midi_path, out_dir / f"musica_{style}_s{params.seed}.wav",
                             params.prefer_soundfont, log, track_map=track_map,
                             style=style, forced_soundfont=params.forced_soundfont)

    meta = {
        "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": params.seed, "prompt": params.prompt, "style": style,
        "bpm": params.bpm, "key": PC_TO_NAME[params.key_root], "scale": params.scale,
        "instruments": [i for i in inst_ids if i in track_of],
        "duration_s": round(abs_bar * beats_per_bar * beat_sec, 1),
        "bars": abs_bar,
        "sections": [{"nome": nm, "bars": bb, "transpose": tr}
                     for (nm, bb), tr in zip(sections, sec_tr)],
        "progression_a": prog_a, "progression_a2": prog_a2, "progression_b": prog_b,
        "ai": {"enabled": ai_on, "bpm_source": bpm_source,
               "personality": personality,
               "temperature_used": round(temperature, 3),
               "auto_score_repetitividade": round(score, 4) if score is not None else None,
               "reference": ref_info},
        "forced_soundfont": params.forced_soundfont,
        "files": {"midi": str(midi_path), "wav": str(wav_path) if wav_path else None},
        "generator": "music_generator.py v9",
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("=" * 60)
    log.info("MIDI:  %s", midi_path)
    if wav_path:
        log.info("WAV:   %s", wav_path)
    log.info("META:  %s", meta_path)
    log.info("Seed: %d — use --seed %d para reproduzir esta música",
             params.seed, params.seed)
    return {"midi": midi_path, "wav": wav_path, "meta": meta_path, "seed": params.seed}


# =============================================================================
# CLI
# =============================================================================

def list_instruments() -> None:
    print(f"\n{'ID':<22} {'NOME':<26} PERC")
    print("-" * 60)
    for key, entry in sorted(INSTRUMENTS.items()):
        perc = "Sim" if entry.get("is_percussion") else ""
        name = str(entry.get("display_name") or entry.get("name") or key)
        print(f"{key:<22} {name[:26]:<26} {perc}")
    print()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gerador de música IA Music Pro.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--prompt", default="", help="prompt em português")
    p.add_argument("--style", default=None,
                   help="rock, pop, samba, bossa, funk, reggae, trap, electronic, hiphop, "
                        "jazz, ambient, cinematic, classical, folk, latin, forro, "
                        "bossfight, chiptune, dungeon, breakcore, dnb, synthwave, "
                        "disco, blues, metal, choro, capoeira")
    p.add_argument("--instruments", default=None,
                   help='lista separada por vírgula (ex.: "bateria,piano")')
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--bpm", type=int, default=None)
    p.add_argument("--key", default=None, help="ex.: C, F#, Am")
    p.add_argument("--scale", default=None, choices=sorted(SCALES))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--output-dir", type=Path, default=Path("song_output"))
    p.add_argument("--no-soundfont", action="store_true",
                   help="pula SoundFont/FluidSynth e usa o renderizador embutido")
    p.add_argument("--no-ai", action="store_true",
                   help="desativa knowledge_base/autoencoder/feedback/referência")
    p.add_argument("--soundfont", default=None,
                   help="força um .sf2 (nome em soundfonts/ ou caminho) para a "
                        "música INTEIRA")
    p.add_argument("--reference", default=None,
                   help="áudio de referência (MP3/WAV) — BPM/energia guiam a geração")
    p.add_argument("--render-midi", default=None,
                   help="renderiza um MIDI existente com os SoundFonts do repo")
    p.add_argument("--audition", action="store_true",
                   help="v9: testa TODOS os .sf2 (sonda melódica + bateria), "
                        "mede qualidade e salva o aprendizado em memory/")
    p.add_argument("--gm-instruments", action="store_true",
                   help="v9: trilhas harmônicas (piano/órgão) com timbre GM em vez "
                        "dos bancos de síntese do repo")
    p.add_argument("--list-instruments", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    global _GM_INSTRUMENTS
    args = parse_args(argv)
    log = _setup_logging(args.verbose)
    _GM_INSTRUMENTS = bool(args.gm_instruments)
    if _RI_ERR:
        log.warning("real_instruments.py não importado (%s) — catálogo de emergência.",
                    _RI_ERR)
    if args.list_instruments:
        list_instruments()
        return 0
    if args.audition:
        run_audition(log)
        return 0
    if args.render_midi:
        src = Path(args.render_midi)
        if not src.exists():
            log.error("Arquivo não encontrado: %s", src)
            return 1
        wav = render_midi_file(src, Path("song_output"), style="", log=log)
        if wav:
            log.info("Pronto: %s", wav)
            return 0
        log.error("Não foi possível renderizar %s", src)
        return 1

    if not args.prompt and not args.style:
        log.error("Informe --prompt e/ou --style.")
        return 1

    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 999_999)
    rng = random.Random(seed)

    raw_style = _norm(args.style) if args.style else detect_style(args.prompt)
    style = _FLAT_STYLE.get(raw_style, raw_style) if raw_style else "pop"
    if style not in STYLES:
        close = difflib.get_close_matches(style, list(STYLES), n=1)
        if close:
            log.info("Estilo '%s' não reconhecido — usando '%s'.", args.style, close[0])
            style = close[0]
        else:
            style = "pop"

    key_pc, minor_hint = None, None
    if args.key:
        key_pc, minor_hint = parse_key_arg(args.key)
        if key_pc is None:
            log.warning("--key inválido (%r) — ignorado.", args.key)
    if key_pc is None:
        key_pc, minor_hint = detect_key(args.prompt)
    if key_pc is None:
        key_pc = rng.choice([0, 2, 3, 5, 7, 8, 9, 10])

    scale = args.scale
    if not scale:
        if minor_hint is True:
            scale = "minor"
        elif minor_hint is False:
            scale = "major"
        else:
            scale = STYLES[style]["scale"]

    instruments: List[str] = []
    if args.instruments:
        instruments = parse_instrument_arg(args.instruments)
    elif args.prompt:
        instruments = parse_instruments_from_text(args.prompt)
    if instruments:
        log.info("Instrumentos detectados: %s", instruments)
    else:
        log.info("Nenhum instrumento no prompt — padrão do estilo '%s'.", style)

    params = SongParams(
        prompt=args.prompt, style=style,
        bpm=args.bpm or detect_bpm(args.prompt, rng, STYLES[style]),
        key_root=key_pc, scale=scale, instruments=instruments,
        duration=max(8.0, args.duration), seed=seed,
        temperature=min(1.0, max(0.0, args.temperature)),
        output_dir=args.output_dir, prefer_soundfont=not args.no_soundfont,
        bpm_explicit=bool(args.bpm), use_ai=not args.no_ai,
        forced_soundfont=args.soundfont, reference=args.reference,
    )
    try:
        generate_song(params, log)
    except Exception:
        log.exception("Falha na geração.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
