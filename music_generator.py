#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
music_generator.py — Gerador principal do IA Music Pro (v5).

Integrado à API REAL do projeto:
    * midi_composer.MidiComposer:
        - MidiComposer(title, bpm, key="C", scale="major", style)
        - add_track(instrument_id) -> int          (ValueError se desconhecido)
        - add_note(idx, *, start, duration, pitch, velocity)  [SEGUNDOS]
        - save_midi(path) -> bool  (False se mido faltar — CHECAR o retorno!)
        - save_json(path)
    * real_instruments.INSTRUMENTS (campos: display_name, channel,
      midi_program, is_percussion, aliases)

v5 — 12 TEMAS NOVOS (28 estilos no total):
    * games:     bossfight, chiptune, dungeon
    * eletrônicos: breakcore, dnb, synthwave, disco
    * raízes:    blues (12 compassos + shuffle), metal (double kick),
                 choro, capoeira
    * + instrumentos GM: square_lead (80), synth_bass (38)
    * 'batalha', 'luta', 'guerra', 'chefe final' etc. agora detectam bossfight

v4 — correções mantidas:
    1. CATÁLOGO ESSENCIAL GARANTIDO (fim do "só piano / bateria não usa").
    2. CLAMP DE PITCH em note() — ValueError nunca mais derruba o pipeline.
    3. CANAIS ÚNICOS por instrumento melódico.
    * Bateria: engine própria (pitches GM) + auditoria pós-geração.
    * Seed aleatório por padrão (--seed N reproduz); parsing PT-BR tolerante.

Uso:
    python music_generator.py --prompt "música de bossfight épica" --duration 30
    python music_generator.py --prompt "breakcore caótico" --duration 15
    python music_generator.py --list-instruments
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

# --- catálogo de instrumentos ------------------------------------------------
try:
    from real_instruments import INSTRUMENTS
    _RI_ERR: Optional[str] = None
except ImportError as _exc:  # catálogo de emergência (só para --list-instruments)
    _RI_ERR = str(_exc)
    INSTRUMENTS = {
        "piano":    {"display_name": "Piano", "channel": 0, "midi_program": 0,
                     "is_percussion": False},
        "violin":   {"display_name": "Violino", "channel": 7, "midi_program": 40,
                     "is_percussion": False},
        "cello":    {"display_name": "Violoncelo", "channel": 8, "midi_program": 42,
                     "is_percussion": False},
    }

# =============================================================================
# INSTRUMENTOS ESSENCIAIS (garantia mínima) — v4, CORREÇÃO 1
# =============================================================================
# setdefault() NÃO sobrescreve entradas existentes: o real_instruments.py
# continua sendo a fonte da verdade; isto é uma rede de segurança.
# DEVE ficar antes de _build_catalog_index() (executado no import).
_ESSENTIAL_INSTRUMENTS: Dict[str, Dict[str, Any]] = {
    # melódicos
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
    "saxophone":        {"display_name": "Saxofone", "channel": 15, "midi_program": 65,
                        "is_percussion": False, "aliases": ["sax", "saxofone"]},
    # percussão — canal 9 (GM)
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

# v5: instrumentos dos novos temas
INSTRUMENTS.setdefault("square_lead",
    {"display_name": "Square Lead", "channel": 0, "midi_program": 80,
     "is_percussion": False, "aliases": ["chiptune lead", "8bit lead"]})
INSTRUMENTS.setdefault("synth_bass",
    {"display_name": "Synth Bass", "channel": 3, "midi_program": 38,
     "is_percussion": False, "aliases": ["reese", "808 bass"]})

# --- compositor --------------------------------------------------------------
try:
    from midi_composer import MidiComposer
except ImportError as _exc2:
    MidiComposer = None
    _MIDI_COMPOSER_ERR = str(_exc2)

# --- renderizador de áudio (API ainda não confirmada → probe por nome) -------
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
# MAPA DE PERCUSSÃO GM (canal 9)
# =============================================================================

GM: Dict[str, int] = {
    "kick": 36, "snare": 38, "rim": 37, "clap": 39, "hh_closed": 42,
    "hh_pedal": 44, "hh_open": 46, "crash": 49, "ride": 51, "ride_bell": 53,
    "tambourine": 54, "cowbell": 56, "tom_low": 41, "tom_mid": 47, "tom_high": 50,
    "conga_slap": 62, "conga_open": 63, "conga_low": 64, "timbale": 65,
    "agogo_hi": 67, "agogo_lo": 68, "cabasa": 69, "shaker": 70, "maracas": 70,
    "guiro": 74, "claves": 75, "wood_hi": 76, "wood_lo": 77, "cuica": 78,
    "triangle": 80, "timpani": 41, "whistle": 72,
    # aproximações GM para percussão brasileira
    "surdo": 36, "tamborim": 76, "pandeiro": 54, "reco": 74, "reco_reco": 74,
    "chocalho": 70, "berimbau": 47, "agogo": 67, "conga": 63,
}
DRUM_DUR: Dict[str, float] = {  # segundos (pratos precisam de cauda!)
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
    """minúsculas, sem acentos, só letras/números/espaços."""
    return re.sub(r"[^a-z0-9 ]", " ", _strip_accents(str(text).lower())).strip()


def _build_catalog_index() -> Dict[str, str]:
    """Índice nome-normalizado -> id, usando id + display_name + aliases."""
    index: Dict[str, str] = {}
    for key, entry in INSTRUMENTS.items():
        names = [key, str(entry.get("display_name") or entry.get("name") or "")]
        names += [str(a) for a in (entry.get("aliases") or [])]
        for nm in names:
            if nm:
                index.setdefault(_norm(nm).replace("_", " "), key)
    return index


_CAT_INDEX = _build_catalog_index()

# (palavras PT/EN normalizadas, ids candidatos em ordem de preferência)
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
    """Tenta id direto, depois nome de exibição/alias no catálogo."""
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
        if rid:
            if rid not in ids:
                ids.append(rid)
        else:
            LOG.warning("Instrumento não reconhecido (ignorado): %r — "
                        "use --list-instruments", part)
    return ids


def parse_instruments_from_text(text: str) -> List[str]:
    tokens = _norm(text).split()
    bigrams = [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
    found: List[str] = []
    for phrase in bigrams + tokens:  # bigramas primeiro ("reco reco")
        rid = find_instrument(phrase)
        if rid and rid not in found:
            found.append(rid)
    return found


# =============================================================================
# TEORIA MUSICAL
# =============================================================================

SCALES: Dict[str, List[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
}
NOTE_NAMES_PT = {"do": 0, "re": 2, "mi": 4, "fa": 5, "sol": 7, "la": 9, "si": 11}
NOTE_NAMES_EN = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
PC_TO_NAME = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

STYLES: Dict[str, Dict[str, Any]] = {
    # --- base -----------------------------------------------------------------
    "rock":       dict(bpm=(100, 150), scale="minor", swing=0.0,
                       instruments=[("electric_guitar", "guitarra eletrica", "guitarra"),
                                    ("bass", "baixo"), ("drums",)]),
    "pop":        dict(bpm=(95, 130), scale="major", swing=0.0,
                       instruments=[("piano", "teclado", "electric_piano"),
                                    ("bass", "baixo"), ("drums",)]),
    "samba":      dict(bpm=(85, 112), scale="major", swing=0.0,
                       instruments=[("cavaquinho", "cavaco", "banjo", "acoustic_guitar", "violao"),
                                    ("acoustic_guitar", "violao"),
                                    ("bass", "baixo"), ("surdo",), ("pandeiro",),
                                    ("agogo", "tamborim")]),
    "bossa":      dict(bpm=(68, 100), scale="major", swing=0.0, sevenths=True,
                       instruments=[("acoustic_guitar", "violao", "nylon_guitar"),
                                    ("piano",), ("bass", "baixo"), ("drums",)]),
    "funk":       dict(bpm=(120, 140), scale="minor", swing=0.0,
                       instruments=[("synth_lead", "sintetizador", "electric_piano", "piano"),
                                    ("bass", "baixo"), ("drums",)]),
    "reggae":     dict(bpm=(72, 92), scale="minor", swing=0.0,
                       instruments=[("electric_guitar", "guitarra eletrica", "acoustic_guitar", "violao"),
                                    ("organ", "orgao"), ("bass", "baixo"), ("drums",)]),
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
                       instruments=[("saxophone", "saxofone", "sax", "trumpet", "trompete", "piano"),
                                    ("piano",), ("bass", "contrabaixo", "baixo"), ("drums",)]),
    "ambient":    dict(bpm=(55, 85), scale="major", swing=0.0,
                       instruments=[("synth_pad", "pad", "strings", "cordas", "piano"),
                                    ("piano",)]),
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
    # --- v5: games ---------------------------------------------------------------
    "bossfight":  dict(bpm=(142, 165), scale="harmonic_minor", swing=0.0,
                       instruments=[("strings", "cordas"), ("trumpet", "trompete"),
                                    ("piano",), ("drums",)]),
    "chiptune":   dict(bpm=(130, 165), scale="major", swing=0.0,
                       instruments=[("square_lead", "chiptune lead"),
                                    ("synth_bass", "synth bass"), ("drums",)]),
    "dungeon":    dict(bpm=(50, 70), scale="harmonic_minor", swing=0.0,
                       instruments=[("strings", "cordas"), ("piano",)]),
    # --- v5: eletrônicos -----------------------------------------------------------
    "breakcore":  dict(bpm=(170, 200), scale="minor", swing=0.0,
                       instruments=[("synth_lead", "sintetizador"), ("synth_pad", "pad"),
                                    ("bass", "baixo"), ("drums",)]),
    "dnb":        dict(bpm=(168, 178), scale="minor", swing=0.0,
                       instruments=[("synth_lead", "sintetizador", "electric_piano", "piano"),
                                    ("synth_pad", "pad", "strings", "cordas"),
                                    ("bass", "baixo"), ("drums",)]),
    "synthwave":  dict(bpm=(100, 116), scale="minor", swing=0.0,
                       instruments=[("synth_lead", "sintetizador"), ("synth_pad", "pad"),
                                    ("bass", "baixo"), ("drums",)]),
    "disco":      dict(bpm=(112, 126), scale="major", swing=0.0,
                       instruments=[("electric_piano", "piano"),
                                    ("synth_pad", "pad", "strings", "cordas"),
                                    ("bass", "baixo"), ("drums",)]),
    # --- v5: raízes ---------------------------------------------------------------
    "blues":      dict(bpm=(72, 100), scale="blues", swing=0.12, sevenths=True,
                       instruments=[("electric_guitar", "guitarra", "acoustic_guitar", "violao"),
                                    ("organ", "orgao"), ("bass", "baixo"), ("drums",)]),
    "metal":      dict(bpm=(140, 180), scale="harmonic_minor", swing=0.0,
                       instruments=[("electric_guitar", "guitarra", "guitarra eletrica"),
                                    ("bass", "baixo"), ("drums",)]),
    "choro":      dict(bpm=(110, 150), scale="major", swing=0.0, sevenths=True,
                       instruments=[("cavaquinho", "cavaco"), ("flute", "flauta"),
                                    ("pandeiro",)]),
    "capoeira":   dict(bpm=(122, 138), scale="mixolydian", swing=0.0,
                       instruments=[("acoustic_guitar", "violao"), ("berimbau",),
                                    ("conga", "congas"), ("agogo",), ("pandeiro",)]),
}
NO_DRUMS_STYLES = {"ambient", "classical", "dungeon"}
AUTO_BASS_STYLES = {"pop", "rock", "funk", "reggae", "trap", "electronic",
                    "hiphop", "jazz", "bossa", "latin", "forro", "samba",
                    "cinematic",
                    # v5
                    "bossfight", "chiptune", "breakcore", "dnb", "synthwave",
                    "disco", "blues", "metal", "choro", "dungeon"}
CRASH_STYLES = {"rock", "pop", "electronic", "funk", "trap", "cinematic", "latin",
                # v5
                "bossfight", "metal", "breakcore", "disco"}

# Progressões por GRAU da escala (0 = tônica)
PROGRESSIONS: Dict[str, List[List[int]]] = {
    "pop":        [[0, 4, 5, 3], [0, 5, 3, 4], [5, 3, 0, 4], [0, 4, 5, 4]],
    "rock":       [[0, 3, 4, 3], [0, 5, 3, 4], [0, 0, 3, 4], [5, 3, 0, 4]],
    "samba":      [[1, 4, 0, 0], [2, 5, 1, 4], [1, 4, 0, 3], [0, 4, 5, 3]],
    "bossa":      [[1, 4, 0, 0], [0, 5, 1, 4], [1, 4, 2, 5], [2, 5, 1, 4]],
    "funk":       [[0, 0, 3, 3], [0, 3, 0, 3], [0, 3, 4, 3]],
    "reggae":     [[0, 3, 4, 3], [0, 5, 3, 4], [0, 0, 3, 3]],
    "trap":       [[0, 5, 3, 4], [5, 3, 0, 4], [0, 5, 0, 4]],
    "electronic": [[0, 5, 3, 4], [5, 3, 0, 4], [0, 3, 5, 4]],
    "hiphop":     [[0, 3, 4, 3], [0, 5, 1, 4], [1, 4, 0, 0]],
    "jazz":       [[1, 4, 0, 0], [1, 4, 0, 3], [0, 3, 6, 2]],   # ii-V-I
    "ambient":    [[0, 3, 0, 4], [0, 4, 3, 0], [0, 3, 5, 4]],
    "cinematic":  [[0, 5, 3, 4], [0, 3, 5, 4], [0, 5, 1, 4]],
    "classical":  [[0, 4, 5, 3], [0, 3, 4, 0], [0, 4, 0, 4]],
    "folk":       [[0, 3, 4, 3], [0, 4, 5, 3], [0, 3, 0, 4]],
    "latin":      [[0, 3, 4, 3], [0, 4, 3, 4], [1, 4, 0, 0]],
    "forro":      [[0, 3, 0, 4], [0, 4, 3, 4], [0, 3, 4, 0]],
    # v5
    "bossfight":  [[0, 5, 3, 4], [0, 1, 4, 0], [0, 3, 0, 4]],   # grau bII = ameaça épica
    "chiptune":   [[0, 4, 5, 3], [0, 3, 4, 4], [0, 5, 3, 4], [5, 3, 0, 4]],
    "dungeon":    [[0, 3, 5, 4], [0, 5, 3, 0], [0, 1, 0, 4]],
    "breakcore":  [[0, 5, 3, 4], [0, 3, 4, 3], [5, 3, 0, 4]],
    "dnb":        [[0, 5, 3, 4], [0, 3, 0, 4], [5, 3, 0, 4]],
    "synthwave":  [[0, 5, 3, 4], [5, 3, 0, 4], [0, 3, 5, 4]],
    "disco":      [[0, 5, 3, 4], [0, 3, 4, 3], [0, 4, 5, 3]],
    "blues":      [[0, 0, 0, 0, 3, 3, 0, 0, 4, 3, 0, 4]],  # 12 compassos: I-I-I-I / IV-IV-I-I / V-IV-I-V
    "metal":      [[0, 3, 4, 3], [0, 5, 3, 4], [0, 0, 5, 4]],
    "choro":      [[1, 4, 0, 0], [2, 5, 1, 4], [0, 4, 1, 4]],
    "capoeira":   [[0, 3, 0, 4], [0, 4, 3, 4], [0, 0, 3, 4]],
}

# (posição em tempos, tipo de nota, duração em tempos)
BASS_PATTERNS: Dict[str, List[Tuple[float, str, float]]] = {
    "rock":       [(i * 0.5, "root" if i != 5 else "fifth", 0.45) for i in range(8)],
    "pop":        [(0.0, "root", 0.9), (1.5, "root", 0.4), (2.0, "fifth", 0.9), (3.5, "root", 0.4)],
    "samba":      [(0.0, "root", 0.7), (1.5, "root", 0.35), (2.0, "root", 0.7), (3.5, "fifth", 0.35)],
    "bossa":      [(0.0, "root", 1.4), (1.5, "root", 0.4), (2.0, "fifth", 1.4), (3.5, "fifth", 0.4)],
    "jazz":       [(0.0, "root", 0.9), (1.0, "third", 0.9), (2.0, "fifth", 0.9), (3.0, "approach", 0.9)],
    "funk":       [(0.0, "root", 0.4), (0.75, "root", 0.2), (1.5, "octave", 0.4),
                   (2.0, "root", 0.4), (2.75, "fifth", 0.2), (3.5, "root", 0.4)],
    "reggae":     [(2.0, "root", 0.7), (3.5, "fifth", 0.35)],
    "trap":       [(0.0, "root", 1.4), (2.5, "root", 1.3)],
    "electronic": [(i * 0.5, "root", 0.45) for i in range(8)],
    "hiphop":     [(0.0, "root", 0.7), (2.0, "fifth", 0.65), (3.25, "root", 0.35)],
    "ambient":    [(0.0, "root", 3.8)],
    "cinematic":  [(0.0, "root", 3.8)],
    "classical":  [(0.0, "root", 1.9), (2.0, "fifth", 1.9)],
    "folk":       [(0.0, "root", 1.9), (2.0, "fifth", 1.9)],
    "latin":      [(0.0, "root", 0.9), (1.5, "root", 0.4), (2.5, "root", 0.9), (3.5, "fifth", 0.4)],
    "forro":      [(0.0, "root", 0.7), (1.75, "root", 0.3), (2.0, "root", 0.7), (3.5, "fifth", 0.35)],
    # v5
    "bossfight":  [(i * 0.5, "root" if i not in (3, 7) else "fifth", 0.45) for i in range(8)],
    "chiptune":   [(i * 0.5, "root" if i % 2 == 0 else "octave", 0.4) for i in range(8)],
    "dungeon":    [(0.0, "root", 3.8)],                                  # drone grave
    "breakcore":  [(0.0, "root", 0.2), (0.75, "root", 0.2), (1.5, "octave", 0.2),
                   (2.25, "root", 0.2), (3.0, "fifth", 0.2), (3.5, "octave", 0.2)],
    "dnb":        [(i * 0.5, "root" if i % 2 == 0 else "octave", 0.45) for i in range(8)],  # rolling
    "synthwave":  [(i * 0.5, "root" if i % 2 == 0 else "octave", 0.42) for i in range(8)],
    "disco":      [(i * 0.25, "root" if i % 2 == 0 else "octave", 0.22) for i in range(16)],  # oitavas 16avos
    "blues":      [(0.0, "root", 0.9), (1.0, "fifth", 0.9), (2.0, "root", 0.9), (3.0, "approach", 0.9)],
    "metal":      [(i * 0.5, "root", 0.45) for i in range(8)],
    "choro":      [(0.0, "root", 0.7), (1.5, "root", 0.35), (2.0, "root", 0.7), (3.5, "fifth", 0.35)],
    "capoeira":   [(0.0, "root", 0.7), (1.5, "root", 0.35), (2.0, "fifth", 0.7), (3.5, "root", 0.35)],
}

COMP_RHYTHMS: Dict[str, List[List[float]]] = {
    "pop":        [[0.0, 2.0], [0.0, 1.5, 2.0, 3.0], [0.0, 2.0, 3.5]],
    "rock":       [[0.0, 2.0], [0.0, 2.0, 3.5], [0.0, 1.5, 2.0, 3.0]],
    "samba":      [[0.0, 1.75, 2.5, 3.25], [0.75, 1.5, 2.25, 3.0, 3.75], [0.0, 0.75, 2.0, 2.75]],
    "bossa":      [[0.0, 0.75, 2.0, 2.75], [0.0, 1.5, 2.5], [0.75, 1.5, 2.0, 2.75, 3.5]],
    "jazz":       [[0.0, 1.5], [1.5], [0.0], [2.5, 3.5]],   # charleston & cia
    "funk":       [[0.0, 0.75, 1.5, 2.25, 3.0], [0.0, 1.5, 2.5, 3.25]],
    "reggae":     [[0.5, 1.5, 2.5, 3.5]],                    # skank no contratempo
    "trap":       [[0.0], [0.0, 2.5]],
    "electronic": [[0.0], [0.0, 1.5, 3.5]],
    "hiphop":     [[0.0, 2.5], [0.0, 1.75, 3.25]],
    "ambient":    [[0.0]],
    "cinematic":  [[0.0]],
    "classical":  [[0.0, 2.0]],
    "folk":       [[0.0, 2.0], [0.0, 1.0, 2.0, 3.0]],
    "latin":      [[0.0, 0.75, 1.5, 2.0, 2.75, 3.5], [0.0, 0.75, 1.5, 2.25, 3.0, 3.75]],
    "forro":      [[0.0, 0.75, 2.0, 2.75], [0.0, 1.5, 2.0, 3.5]],
    # v5
    "bossfight":  [[0.0, 1.5, 2.0, 3.5], [0.0, 0.75, 2.0, 2.75]],           # stabs marciais
    "chiptune":   [[i * 0.25 for i in range(16)]],                          # arpejo em 16avos
    "dungeon":    [[0.0]],
    "breakcore":  [[0.0, 0.75, 1.5, 2.25, 3.0, 3.75]],
    "dnb":        [[0.0, 2.5], [0.0, 1.75, 3.25]],
    "synthwave":  [[0.0, 1.5, 2.5], [0.0, 0.75, 2.0, 2.75]],
    "disco":      [[0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.0, 3.5]],             # stabs no contratempo
    "blues":      [[0.0, 1.0, 2.0, 3.0]],                                   # shuffle via swing
    "metal":      [[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]],               # power chords 8avos
    "choro":      [[0.0, 0.75, 1.5, 2.0, 2.75, 3.5], [0.75, 1.5, 2.25, 3.0, 3.75]],  # paradigma
    "capoeira":   [[0.0, 0.75, 1.5, 2.25, 3.0], [0.0, 1.5, 2.0, 3.5]],
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
    "pop":        [("intro", 2), ("verso", 4), ("refrao", 4), ("verso", 4),
                   ("refrao", 4), ("ponte", 2), ("refrao", 4), ("outro", 2)],
    "rock":       [("intro", 2), ("verso", 4), ("refrao", 4), ("verso", 4),
                   ("refrao", 4), ("solo", 4), ("refrao", 4)],
    "samba":      [("intro", 2), ("verso", 4), ("refrao", 4), ("verso", 4),
                   ("refrao", 4), ("outro", 2)],
    "bossa":      [("intro", 2), ("verso", 4), ("refrao", 4), ("verso", 4), ("refrao", 4)],
    "jazz":       [("intro", 1), ("verso", 4), ("solo", 4), ("verso", 4), ("outro", 1)],
    "cinematic":  [("intro", 2), ("verso", 4), ("refrao", 4), ("verso", 4),
                   ("refrao", 4), ("ponte", 2), ("refrao", 4), ("outro", 2)],
    "ambient":    [("verso", 4), ("verso", 4), ("verso", 4)],
    # v5
    "bossfight":  [("intro", 2), ("verso", 4), ("refrao", 4), ("ponte", 2),
                   ("solo", 4), ("refrao", 4)],
    "chiptune":   [("intro", 2), ("refrao", 4), ("verso", 4), ("refrao", 4),
                   ("ponte", 2), ("refrao", 4)],
    "breakcore":  [("intro", 2), ("verso", 4), ("refrao", 4), ("ponte", 2),
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
# PADRÕES DE PERCUSSÃO
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
        p = [(1.0, "kick", 102), (3.0, "kick", 96)]                       # surdo: 2 e 4
        p += [(i * 0.25, "tamborim", [78, 52, 66, 52][i % 4]) for i in range(16)]  # teleco-teco
        p += [(0.0, "agogo_hi", 82), (0.75, "agogo_lo", 72), (1.5, "agogo_hi", 76),
              (2.0, "agogo_hi", 82), (2.75, "agogo_lo", 72), (3.5, "agogo_hi", 76)]
        p += [(i * 0.25, "shaker", 56 if i % 2 == 0 else 66) for i in range(16)]
        p += [(0.5, "snare", 44), (2.5, "snare", 42)]                    # caixa fantasma
    elif style == "bossa":
        p = [(0.0, "kick", 78), (0.75, "kick", 58), (2.0, "kick", 78), (2.75, "kick", 58)]
        p += [(1.0, "rim", 70), (3.0, "rim", 70)]
        p += [(i * 0.5, "hh_closed", 64 if i % 2 == 0 else 52) for i in range(8)]
    elif style == "funk":  # funk carioca (tamborzão aproximado)
        p = [(0.0, "kick", 105), (1.75, "kick", 92), (2.25, "kick", 100),
             (1.0, "clap", 96), (3.0, "clap", 98)]
        if bar % 2 == 1:
            p.append((3.5, "kick", 88))
        p += [(i * 0.25, "hh_closed", [58, 40, 50, 40][i % 4]) for i in range(16)]
    elif style == "reggae":  # one drop
        p = [(2.0, "kick", 100), (2.0, "rim", 88)]
        if bar % 2 == 1:
            p.append((3.75, "kick", 68))
        p += [(i * 0.5, "hh_closed", 66 if i % 2 == 0 else 50) for i in range(8)]
    elif style == "trap":  # half-time
        p = [(0.0, "kick", 106), (1.75, "kick", 94), (2.0, "snare", 100)]
        if bar % 4 == 2:
            p.append((3.5, "kick", 90))
        p += [(i * 0.25, "hh_closed", [82, 48, 64, 48][i % 4]) for i in range(16)]
        if bar % 4 == 3 and rng.random() < 0.5:
            p += [(3.0 + i * 0.125, "hh_closed", 60) for i in range(8)]   # roll
    elif style == "electronic":  # four-on-the-floor
        p = [(b, "kick", 100) for b in range(4)]
        p += [(1.0, "clap", 88), (3.0, "clap", 88)]
        p += [(b + 0.5, "hh_open", 70) for b in range(4)]
    elif style == "hiphop":  # boom bap
        p = [(0.0, "kick", 102), (2.5, "kick", 96), (1.0, "snare", 98), (3.0, "snare", 98)]
        p += [(i * 0.5, "hh_closed", 70 if i % 2 == 0 else 54) for i in range(8)]
    elif style == "jazz":  # ride com swing aplicado na escrita
        p = [(i * 0.5, "ride", 74 if i % 2 == 0 else 58) for i in range(8)]
        p += [(1.0, "hh_pedal", 64), (3.0, "hh_pedal", 64)]
        p += [(0.0, "kick", 34), (2.0, "kick", 30)]                      # feathering
        if rng.random() < 0.4:
            p.append((rng.choice([1.5, 2.5, 3.5]), "snare", 48))         # comping
    elif style == "latin":  # tumbao + clave son 3-2
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
    elif style == "forro":  # zabumba + triângulo (baião)
        p = [(0.0, "kick", 100), (2.0, "kick", 94), (1.5, "snare", 56), (3.5, "snare", 58)]
        p += [(i * 0.5, "triangle", 62) for i in range(8)]
    elif style == "folk":
        p = [(1.0, "tambourine", 62), (3.0, "tambourine", 60)]
        if rng.random() < 0.3:
            p.append((0.0, "kick", 58))
    # --- v5: novos temas -------------------------------------------------------
    elif style == "bossfight":  # marcha épica + rolo de caixa crescente
        p = [(b, "kick", 104) for b in range(4)]
        p += [(1.0, "snare", 98), (3.0, "snare", 100)]
        p += [(0.5, "snare", 56), (1.5, "snare", 60), (2.5, "snare", 64),
              (3.25, "snare", 70), (3.5, "snare", 78), (3.75, "snare", 86)]
        p += [(0.0, "timpani", 96), (2.0, "timpani", 90)]
        if bar % 4 == 0:
            p.append((0.0, "crash", 94))
    elif style == "chiptune":  # bateria 8-bit minimalista
        p = [(0.0, "kick", 88), (2.0, "kick", 84), (1.0, "snare", 86), (3.0, "snare", 88)]
        p += [(i * 0.5, "hh_closed", [72, 38, 54, 38][i % 4]) for i in range(8)]
    elif style == "breakcore":  # amen break picotado
        p = [(0.0, "kick", 104), (1.0, "snare", 100), (2.0, "snare", 98),
             (2.5, "kick", 96), (3.0, "snare", 100)]
        if bar % 2 == 1:  # chop no compasso ímpar
            p += [(0.75, "snare", 84), (1.5, "kick", 88), (1.75, "snare", 80),
                  (3.25, "snare", 90), (3.5, "snare", 96), (3.75, "snare", 102)]
        p += [(i * 0.25, "hh_closed", [72, 40, 58, 40][i % 4]) for i in range(16)]
        if bar % 4 == 3:
            p += [(3.0 + i * 0.125, "snare", 50 + 6 * i) for i in range(8)]
    elif style == "dnb":  # two-step clássico
        p = [(0.0, "kick", 104), (2.5, "kick", 96), (2.0, "snare", 102),
             (3.75, "snare", 66)]
        p += [(i * 0.5, "hh_closed", 68 if i % 2 == 0 else 50) for i in range(8)]
        p.append((1.5, "hh_open", 58))
    elif style == "synthwave":  # gated snare + open hats no contratempo
        p = [(0.0, "kick", 98), (1.0, "snare", 94), (2.0, "kick", 94), (3.0, "snare", 96)]
        p += [(b + 0.5, "hh_open", 62) for b in range(4)]
        if bar % 4 == 3:
            p.append((3.5, "tom_high", 70))
    elif style == "disco":  # four-on-the-floor + open hat no contratempo
        p = [(b, "kick", 100) for b in range(4)]
        p += [(1.0, "snare", 90), (3.0, "snare", 92)]
        p += [(b + 0.5, "hh_open", 74) for b in range(4)]
    elif style == "blues":  # shuffle (o swing do estilo balança os 8avos)
        p = [(0.0, "kick", 96), (2.0, "kick", 90), (1.0, "snare", 88), (3.0, "snare", 90)]
        p += [(i * 0.5, "ride", 66 if i % 2 == 0 else 52) for i in range(8)]
    elif style == "metal":  # double kick alternando com 8avos
        kick_pos = [i * 0.5 for i in range(8)]
        if bar % 2 == 1:
            kick_pos = [i * 0.25 for i in range(16)]  # double kick em 16avos
        p = [(k, "kick", 92) for k in kick_pos]
        p += [(1.0, "snare", 102), (3.0, "snare", 104)]
        p += [(i * 0.5, "ride", 70 if i % 2 == 0 else 56) for i in range(8)]
        if bar % 4 == 0:
            p.append((0.0, "crash", 100))
    elif style == "dungeon":
        p = []  # silêncio e tensão
    elif style in ("ambient", "classical"):
        p = []
    else:
        return _kit_pattern("pop", bar, rng)
    return p


def _fill(style: str, rng: random.Random) -> List[Tuple[float, str, int]]:
    # v5
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
    # base
    if style in ("rock", "pop", "electronic", "folk"):
        seq = ["snare", "snare", "tom_high", "tom_high",
               "tom_mid", "tom_mid", "tom_low", "tom_low"]
        return [(2.0 + 0.25 * i, seq[i], 78 + 5 * i) for i in range(8)]
    if style in ("samba", "bossa", "forro", "latin"):
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


# Padrões por instrumento de percussão individual: velocity por 16avo (0 = pausa)
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
# HELPERS DE TEORIA / INSTRUMENTOS
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
    offs = []
    for k in (0, 2, 4, 6):
        idx = degree + k
        offs.append(scale[idx % n] + 12 * (idx // n))
    return offs if add7 else offs[:3]


def _voice_chord(prev: Optional[List[int]], offsets: List[int],
                 root_midi: int, center: int = 62) -> List[int]:
    """Voice leading: cada nota na oitava mais próxima da anterior."""
    voiced: List[int] = []
    for j, off in enumerate(offsets):
        target = prev[j % len(prev)] if prev else center
        pitch = root_midi + off
        while pitch < target - 6:
            pitch += 12
        while pitch > target + 6:
            pitch -= 12
        voiced.append(pitch)
    return sorted(set(min(127, max(0, p)) for p in voiced))   # defesa extra


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
    elif kind == "approach":  # cromático que resolve na root seguinte
        p = next_root_abs - 1 if next_root_abs >= root_abs else next_root_abs + 1
    else:
        p = root_abs
    while p < 33:
        p += 12
    while p > 50:
        p -= 12
    if style == "trap" and p >= 33:  # 808 mais grave
        p -= 12
    return p


def _monophonic(inst_id: str) -> bool:
    # v5: "square_lead" incluído — chiptune: arpejo e melodia NÃO colidem
    return any(m in inst_id for m in
               ("sax", "trumpet", "flute", "clarinet", "violin", "trombone",
                "synth_lead", "square_lead"))


# =============================================================================
# MOTIVOS (melodia)
# =============================================================================

def _new_motif(rng: random.Random, temperature: float, dense: bool):
    cells = RHYTHM_CELLS_DENSE if (dense and rng.random() < 0.7) else RHYTHM_CELLS_SPARSE
    cell = list(rng.choice(cells))
    steps = [rng.choice([0, 2, 4])]
    for _ in range(len(cell) - 1):
        steps.append(steps[-1] + rng.choice([-2, -1, -1, 1, 1, 2]))
    steps = [max(-2, min(9, s)) for s in steps]
    return sorted(zip(cell, steps), key=lambda x: x[0][0])


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


# =============================================================================
# DETECÇÃO NO PROMPT (PT-BR)
# =============================================================================

_STYLE_MAP: List[Tuple[Tuple[str, ...], str]] = [
    # v5 — novos temas (detect_style() ordena por tamanho: nomes compostos e
    # aliases longos ganham de palavras curtas; "bossa" vence "boss", etc.)
    (("boss fight", "bossfight", "chefe final", "batalha de chefe", "batalha",
      "luta", "guerra", "battle", "boss"), "bossfight"),
    (("chiptune", "chip tune", "8bit", "8 bits", "8 bit", "video game",
      "videogame", "jogo", "nes", "retro"), "chiptune"),
    (("dungeon", "masmorra", "terror", "horror", "sombrio", "assustador",
      "medo", "suspense", "dark"), "dungeon"),
    (("breakcore",), "breakcore"),
    (("drum and bass", "drum n bass", "dnb", "jungle", "liquid"), "dnb"),
    (("synthwave", "synth wave", "retrowave", "vaporwave", "anos 80",
      "oitenta"), "synthwave"),
    (("disco", "boogie", "discoteca"), "disco"),
    (("blues",), "blues"),
    (("metal", "heavy metal", "thrash", "death metal", "doom"), "metal"),
    (("choro", "chorinho"), "choro"),
    (("capoeira",), "capoeira"),
    # base
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
                   key=lambda x: -len(x[0]))  # nomes compostos primeiro
    for w, s in pairs:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return s
    return None


def detect_key(text: str) -> Tuple[Optional[int], Optional[bool]]:
    """(pitch class, é menor?) a partir de 'em dó menor', 'tom de ré'..."""
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
# INTEGRAÇÃO com midi_composer.MidiComposer (API real)
# =============================================================================

def _safe_title(text: str) -> str:
    """ASCII puro (metadados MIDI usam latin-1; acentos ok, emoji não)."""
    t = _strip_accents(str(text))
    t = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in t)
    return re.sub(r"\s+", " ", t).strip()


def _make_composer(params: "SongParams", log: logging.Logger) -> "MidiComposer":
    if MidiComposer is None:
        raise RuntimeError(
            "midi_composer.MidiComposer não pôde ser importado ("
            f"{_MIDI_COMPOSER_ERR or 'motivo desconhecido'}). Verifique se "
            "real_instruments.py e mido estão disponíveis "
            "(pip install -r requirements.txt)."
        )
    key_name = PC_TO_NAME[params.key_root]
    # O composer aceita major/minor/pentatonic; o resto é só metadado para ele
    # (o gerador calcula os pitches ele mesmo).
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
    if ok is False:  # save_midi() retorna False (sem exceção!) se mido faltar
        log.error("save_midi() retornou False — mido está instalado? (pip install mido)")
        return False
    return True


def _render_audio(midi_path: Path, wav_path: Path, prefer_soundfont: bool,
                  log: logging.Logger) -> Optional[Path]:
    if _RENDERER is None:
        log.warning("soundfont_renderer não disponível (nenhuma função conhecida "
                    "encontrada) — WAV não gerado. O MIDI foi gerado normalmente. "
                    "Envie 'grep -n \"^def \" soundfont_renderer.py' para ajustar "
                    "a integração.")
        return None
    import inspect
    attempts: List[Tuple[Tuple, Dict[str, Any]]] = []
    try:
        sig = inspect.signature(_RENDERER)
        kw: Dict[str, Any] = {}
        for name in ("use_soundfont", "prefer_soundfont"):
            if name in sig.parameters:
                kw[name] = prefer_soundfont
        if "soundfont" in sig.parameters and not prefer_soundfont:
            kw["soundfont"] = None
        attempts.append(((), kw))
    except (TypeError, ValueError):
        pass
    attempts.append(((), {}))
    for args_, kw in attempts:
        try:
            _RENDERER(str(midi_path), str(wav_path), **kw)
            return wav_path
        except TypeError:
            continue
        except Exception as exc:
            log.warning("Renderização falhou: %s", exc)
            return None
    log.warning("Não consegui chamar o renderer (esperado: f(midi, wav, ...)). "
                "Ajuste _render_audio().")
    return None


def audit_midi(path: Path, inst_by_track: Dict[int, str], log: logging.Logger) -> bool:
    """Auditoria pós-geração: torna 'bateria não toca' VISÍVEL no log."""
    try:
        import mido
    except ImportError:
        log.warning("mido não instalado — auditoria de MIDI pulada.")
        return True
    mid = mido.MidiFile(str(path))
    ok = True
    log.info("Auditoria do MIDI (%d tracks):", len(mid.tracks))
    for i, track in enumerate(mid.tracks):
        notes = [m for m in track if m.type == "note_on" and m.velocity > 0]
        chans = sorted({m.channel for m in notes})
        progs = [m.program for m in track if m.type == "program_change"]
        inst = inst_by_track.get(i, "?")
        if not notes:
            log.warning("  [track %d] %-16s → 0 NOTAS (algo não foi gerado!)", i, inst)
            ok = False
            continue
        log.info("  [track %d] %-16s → %d notas | canais %s | programa %s",
                 i, inst, len(notes), chans, progs[:1] or "-")
        if _is_perc(inst) and chans and 9 not in chans:
            log.warning("  [track %d] %s é percussão mas está no canal %s — GM exige "
                        "canal 9. Corrija o 'channel' no real_instruments.py.",
                        i, inst, chans)
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
    toggle = True
    while remaining >= 4:
        out.append(("verso" if toggle else "refrao", 4))
        toggle = not toggle
        remaining -= 4
    if remaining > 0:
        out.append(("refrao", remaining))
    return out


def generate_song(params: SongParams, log: logging.Logger = LOG) -> Dict[str, Any]:
    rng = random.Random(params.seed)
    style = params.style
    style_cfg = STYLES.get(style, STYLES["pop"])
    beat_sec = 60.0 / params.bpm
    beats_per_bar = 4
    scale = SCALES[params.scale]
    root_midi = 48 + params.key_root
    swing_on = style_cfg["swing"] > 0

    # --- instrumentação ------------------------------------------------------
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
        kit_id = _resolve_id(("drums", "drum kit", "bateria"))
        if kit_id:
            inst_ids.append(kit_id)
            log.info("Bateria (kit) adicionada automaticamente para o estilo '%s'.", style)

    if bass_id is None and style in AUTO_BASS_STYLES:
        bass_id = _resolve_id(("bass", "baixo", "electric_bass", "fingered_bass",
                               "acoustic_bass"))
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
    if not comp_ids:  # último recurso
        for cand in ("piano", "acoustic_guitar", "organ", "strings"):
            rid = _resolve_id((cand,))
            if rid:
                comp_ids.append(rid)
                inst_ids.append(rid)
                break

    # lead monofônico toca melodia; senão o 1º harmônico vira lead DEDICADO
    # (se houver outro para compor os acordes)
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

    # --- v4, CORREÇÃO 3: canais únicos por instrumento melódico -----------------
    # O MidiComposer lê o canal do catálogo NO MOMENTO do save_midi(); então
    # realocamos aqui para evitar que dois melódicos partilhem canal (o
    # program_change de um sobrescreveria o timbre do outro).
    used_ch = {9}  # 9 = percussão GM
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

    # --- composer e tracks ---------------------------------------------------
    composer = _make_composer(params, log)
    track_of: Dict[str, int] = {}
    inst_by_track: Dict[int, str] = {}
    for inst in dict.fromkeys(inst_ids):
        try:
            idx = composer.add_track(inst)
        except Exception as exc:  # ValueError se desconhecido no catálogo
            log.warning("add_track(%r) falhou (%s) — instrumento ignorado.", inst, exc)
            continue
        if not isinstance(idx, int):
            log.warning("add_track(%r) não retornou índice — instrumento ignorado.", inst)
            continue
        track_of[inst] = idx
        inst_by_track[idx] = inst

    def note(track: int, bar: int, pos: float, dur_beats: float,
             pitch: int, vel: int, swing: bool = False) -> None:
        """Escreve UMA nota. v4: clamp de pitch — nunca mais ValueError."""
        p = bar * beats_per_bar + pos
        if swing and abs((p % 1.0) - 0.5) < 1e-6:
            p += style_cfg["swing"]
        jitter = rng.uniform(-0.012, 0.012) if params.temperature > 0 else 0.0
        start = max(0.0, p + jitter) * beat_sec
        dur = max(0.03, dur_beats * beat_sec * 0.96)
        strong = (pos == int(pos)) and (int(pos) % 2 == 0)
        v = int(min(127, max(20, vel + rng.randint(-6, 6) + (6 if strong else 0))))
        pit = int(round(pitch))
        if not (0 <= pit <= 127):
            log.warning("Pitch fora de 0..127 (%r) em bar=%d pos=%.2f — corrigido "
                        "(investigue a origem se isto repetir)", pitch, bar, pos)
            pit = min(127, max(21, pit))
        composer.add_note(track, start=start, duration=dur, pitch=pit, velocity=v)

    # --- plano da música -------------------------------------------------------
    bars = max(4, int(round(params.duration / (beat_sec * beats_per_bar))))
    sections = _fit_form(bars, style)
    log.info("Estrutura: %s", " → ".join(f"{n}:{b}" for n, b in sections))

    opts = PROGRESSIONS.get(style, PROGRESSIONS["pop"])
    prog_a = list(rng.choice(opts))
    prog_b = list(rng.choice([o for o in opts if o != prog_a] or opts))
    add7 = bool(style_cfg.get("sevenths")) or rng.random() < 0.25
    log.info("Progressões: A=%s | B=%s (refrão) | tom=%s %s | %d BPM | seed=%d",
             prog_a, prog_b, PC_TO_NAME[params.key_root], params.scale,
             params.bpm, params.seed)

    # --- composição ------------------------------------------------------------
    prev_voicing: Optional[List[int]] = None
    motif = None
    abs_bar = 0
    n = len(scale)
    for sec_i, (sec_name, sec_bars) in enumerate(sections):
        profile = SECTION_PROFILES.get(sec_name, SECTION_PROFILES["verso"])
        comp_rhythm = rng.choice(COMP_RHYTHMS.get(style, [[0.0, 2.0]]))
        prog = prog_b if sec_name in ("refrao", "solo") else prog_a
        if profile["melody"] and lead_id and lead_id in track_of:
            motif = _new_motif(rng, params.temperature, profile["dense"])
        for b in range(sec_bars):
            degree = prog[b % len(prog)]
            next_degree = prog[(b + 1) % len(prog)]
            is_last = (b == sec_bars - 1)

            # ---- acordes (comp) ----
            if profile["comp"] and comp_ids:
                offsets = _chord_offsets(scale, degree, add7)
                prev_voicing = _voice_chord(prev_voicing, offsets, root_midi + 12)
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
                                 pit, profile["vel"] - 3 * j, swing=swing_on)

            # ---- baixo ----
            if profile["bass"] and bass_id and bass_id in track_of:
                pat = BASS_PATTERNS.get(style, BASS_PATTERNS["pop"])
                root_abs = root_midi + scale[degree % n]
                next_root_abs = root_midi + scale[next_degree % n]
                for pos, kind, dur in pat:
                    pitch = _bass_pitch(kind, root_abs, next_root_abs, scale, degree, style)
                    note(track_of[bass_id], abs_bar, pos, dur, pitch,
                         profile["vel"] - 8, swing=swing_on)

            # ---- bateria (kit) ----
            if kit_id and kit_id in track_of and profile["drums"]:
                pattern = _kit_pattern(style, abs_bar, rng)
                if is_last and sec_i < len(sections) - 1:
                    pattern = [h for h in pattern if h[0] < 2.0] + _fill(style, rng)
                if b == 0 and sec_name in ("refrao", "solo") and style in CRASH_STYLES:
                    pattern.append((0.0, "crash", 94))
                gain = profile["vel"] / 76.0
                for pos, name, vel in pattern:
                    dur_beats = DRUM_DUR.get(name, DEFAULT_DRUM_DUR) / beat_sec
                    note(track_of[kit_id], abs_bar, pos, dur_beats,
                         GM[name], int(vel * gain), swing=swing_on)

            # ---- percussão individual (pandeiro, surdo, tamborim...) ----
            for pid in perc_ids:
                if pid not in track_of or not profile["drums"]:
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
                         pitch, int(v * (profile["vel"] / 76.0)))

            # ---- melodia (motivo) ----
            if profile["melody"] and lead_id and lead_id in track_of:
                if motif is None:
                    motif = _new_motif(rng, params.temperature, profile["dense"])
                if b % 2 == 1 and rng.random() < params.temperature * 0.5:
                    motif = _vary_motif(motif, rng)
                rest = (abs_bar % 4 == 3) and rng.random() < 0.30 * (1.4 - params.temperature)
                if not rest:
                    for (pos, dur), step in motif:
                        idx = degree + step
                        if pos in (0.0, 2.0):  # tempo forte → nota do acorde
                            idx = _snap_to_chord(idx, degree, n)
                        pitch = max(58, min(86, _scale_pitch(root_midi + 12, scale, idx)))
                        note(track_of[lead_id], abs_bar, pos, dur,
                             pitch, profile["vel"] + 4, swing=swing_on)
            abs_bar += 1

    # --- exportação --------------------------------------------------------------
    out_dir = params.output_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{style}_s{params.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_path = out_dir / f"musica_{style}_s{params.seed}.mid"
    if not _export_midi(composer, midi_path, log):
        raise RuntimeError("Falha ao exportar o MIDI.")
    audit_midi(midi_path, inst_by_track, log)

    try:  # formato nativo do composer (resumo de tracks) — best effort
        composer.save_json(out_dir / "composicao.json")
    except Exception as exc:
        log.debug("save_json() indisponível: %s", exc)

    wav_path = _render_audio(midi_path, out_dir / f"musica_{style}_s{params.seed}.wav",
                             params.prefer_soundfont, log)

    meta = {
        "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": params.seed,
        "prompt": params.prompt,
        "style": style,
        "bpm": params.bpm,
        "key": PC_TO_NAME[params.key_root],
        "scale": params.scale,
        "instruments": [i for i in inst_ids if i in track_of],
        "duration_s": round(abs_bar * beats_per_bar * beat_sec, 1),
        "bars": abs_bar,
        "sections": [{"nome": nm, "bars": bb} for nm, bb in sections],
        "progression_a": prog_a,
        "progression_b": prog_b,
        "files": {"midi": str(midi_path), "wav": str(wav_path) if wav_path else None},
        "generator": "music_generator.py v5 (MidiComposer)",
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
                   help="base: rock, pop, samba, bossa, funk, reggae, trap, electronic, "
                        "hiphop, jazz, ambient, cinematic, classical, folk, latin, forro | "
                        "v5: bossfight, chiptune, dungeon, breakcore, dnb, synthwave, "
                        "disco, blues, metal, choro, capoeira")
    p.add_argument("--instruments", default=None,
                   help='lista separada por vírgula (ex.: "bateria,piano")')
    p.add_argument("--duration", type=float, default=30.0, help="duração em segundos")
    p.add_argument("--bpm", type=int, default=None)
    p.add_argument("--key", default=None, help="ex.: C, F#, Am")
    p.add_argument("--scale", default=None, choices=sorted(SCALES))
    p.add_argument("--seed", type=int, default=None,
                   help="seed do RNG (padrão: aleatório — cada execução gera "
                        "música diferente)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="0..1 — quanto maior, mais variação")
    p.add_argument("--output-dir", type=Path, default=Path("song_output"))
    p.add_argument("--no-soundfont", action="store_true",
                   help="prefere síntese procedural (fallback) ao SoundFont")
    p.add_argument("--list-instruments", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    log = _setup_logging(args.verbose)
    if _RI_ERR:
        log.warning("real_instruments.py não importado (%s) — usando catálogo "
                    "de emergência + essenciais.", _RI_ERR)
    if args.list_instruments:
        list_instruments()
        return 0
    if not args.prompt and not args.style:
        log.error("Informe --prompt e/ou --style.")
        return 1

    # ANTI-REPETIÇÃO: seed aleatório por padrão; o valor é logado para reprodução
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
        log.info("Nenhum instrumento no prompt — usando padrão do estilo '%s'.", style)

    params = SongParams(
        prompt=args.prompt, style=style,
        bpm=args.bpm or detect_bpm(args.prompt, rng, STYLES[style]),
        key_root=key_pc, scale=scale, instruments=instruments,
        duration=max(8.0, args.duration), seed=seed,
        temperature=min(1.0, max(0.0, args.temperature)),
        output_dir=args.output_dir, prefer_soundfont=not args.no_soundfont,
    )
    try:
        generate_song(params, log)
    except Exception:
        log.exception("Falha na geração.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
