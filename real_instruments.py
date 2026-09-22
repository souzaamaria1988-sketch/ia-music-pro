"""Catálogo de instrumentos reais baseados no General MIDI."""
from __future__ import annotations
from typing import Dict, Any
import unicodedata
import re

INSTRUMENTS: Dict[str, Dict[str, Any]] = {
    "piano": {"display_name": "Piano acústico", "family": "keys", "midi_program": 0, "channel": 0, "aliases": ["piano acústico", "piano"], "fallback": "piano", "realistic": True, "is_percussion": False},
    "acoustic_guitar": {"display_name": "Violão", "family": "strings", "midi_program": 25, "channel": 0, "aliases": ["violao", "violão"], "fallback": "acoustic_guitar", "realistic": True, "is_percussion": False},
    "electric_guitar": {"display_name": "Guitarra", "family": "strings", "midi_program": 27, "channel": 0, "aliases": ["guitarra"], "fallback": "electric_guitar", "realistic": True, "is_percussion": False},
    "electric_bass": {"display_name": "Baixo elétrico", "family": "bass", "midi_program": 33, "channel": 1, "aliases": ["baixo"], "fallback": "electric_bass", "realistic": True, "is_percussion": False},
    "violin": {"display_name": "Violino", "family": "strings", "midi_program": 40, "channel": 0, "aliases": ["violino"], "fallback": "violin", "realistic": True, "is_percussion": False},
    "cello": {"display_name": "Violoncelo", "family": "strings", "midi_program": 42, "channel": 0, "aliases": ["violoncelo"], "fallback": "cello", "realistic": True, "is_percussion": False},
    "accordion": {"display_name": "Acordeão", "family": "keys", "midi_program": 21, "channel": 0, "aliases": ["sanfona", "acordeão"], "fallback": "accordion", "realistic": True, "is_percussion": False},
    "kick": {"display_name": "Bumbo", "family": "drums", "midi_note": 36, "channel": 9, "aliases": ["bumbo"], "fallback": "kick", "realistic": True, "is_percussion": True},
    "snare": {"display_name": "Caixa", "family": "drums", "midi_note": 38, "channel": 9, "aliases": ["caixa"], "fallback": "snare", "realistic": True, "is_percussion": True},
    "pandeiro": {"display_name": "Pandeiro", "family": "brazilian", "midi_note": None, "channel": 9, "aliases": ["pandeiro"], "fallback": "pandeiro", "realistic": False, "is_percussion": True},
    "surdo": {"display_name": "Surdo", "family": "brazilian", "midi_note": None, "channel": 9, "aliases": ["surdo"], "fallback": "surdo", "realistic": False, "is_percussion": True},
    "cavaquinho": {"display_name": "Cavaquinho", "family": "brazilian", "midi_note": None, "channel": 0, "aliases": ["cavaquinho"], "fallback": "cavaquinho", "realistic": False, "is_percussion": False},
}

BRAZILIAN_FALLBACK_ONLY = {
    "pandeiro", "surdo", "cavaquinho",
}


def normalize_instrument_name(name: str) -> str:
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().replace("-", " ").replace("_", " ")
    n = re.sub(r"\s+", " ", n).strip()
    n = re.sub(r"s$", "", n)
    return n


def resolve_instrument_alias(name: str):
    target = normalize_instrument_name(name)
    if target in INSTRUMENTS:
        return target
    for key, data in INSTRUMENTS.items():
        if key == target:
            return key
        for a in data.get("aliases", []):
            if normalize_instrument_name(a) == target:
                return key
    return None


def list_families():
    out = {}
    for key, data in INSTRUMENTS.items():
        out.setdefault(data["family"], []).append(key)
    return out
