"""Catálogo de instrumentos reais baseados no General MIDI."""
from __future__ import annotations
from typing import Dict, Any
import unicodedata
import re

INSTRUMENTS: Dict[str, Dict[str, Any]] = {
    "piano":            {"display_name": "Piano acústico",  "family": "keys",     "midi_program": 0,  "channel": 0, "aliases": ["piano acústico", "piano", "piano de cauda"], "fallback": "piano",      "realistic": True,  "is_percussion": False},
    "electric_piano":   {"display_name": "Piano elétrico",  "family": "keys",     "midi_program": 4,  "channel": 0, "aliases": ["piano elétrico", "rhodes"], "fallback": "electric_piano", "realistic": True,  "is_percussion": False},
    "organ":            {"display_name": "Órgão",           "family": "keys",     "midi_program": 19, "channel": 0, "aliases": ["orgao", "órgão"],           "fallback": "organ",    "realistic": True,  "is_percussion": False},
    "acoustic_guitar":  {"display_name": "Violão",          "family": "strings",  "midi_program": 25, "channel": 0, "aliases": ["violao", "violão"], "fallback": "acoustic_guitar", "realistic": True,  "is_percussion": False},
    "electric_guitar":  {"display_name": "Guitarra",        "family": "strings",  "midi_program": 27, "channel": 0, "aliases": ["guitarra"],                 "fallback": "electric_guitar", "realistic": True,  "is_percussion": False},
    "electric_bass":    {"display_name": "Baixo elétrico",  "family": "bass",     "midi_program": 33, "channel": 1, "aliases": ["baixo", "baixo eletrico"],  "fallback": "electric_bass", "realistic": True,  "is_percussion": False},
    "violin":           {"display_name": "Violino",         "family": "strings",  "midi_program": 40, "channel": 0, "aliases": ["violino"],                  "fallback": "violin",   "realistic": True,  "is_percussion": False},
    "viola":            {"display_name": "Viola",           "family": "strings",  "midi_program": 41, "channel": 0, "aliases": ["viola"],                    "fallback": "viola",    "realistic": True,  "is_percussion": False},
    "cello":            {"display_name": "Violoncelo",      "family": "strings",  "midi_program": 42, "channel": 0, "aliases": ["violoncelo", "cello"],      "fallback": "cello",    "realistic": True,  "is_percussion": False},
    "double_bass":      {"display_name": "Contrabaixo",     "family": "strings",  "midi_program": 43, "channel": 1, "aliases": ["contrabaixo"], "fallback": "double_bass", "realistic": True,  "is_percussion": False},
    "flute":            {"display_name": "Flauta",          "family": "woodwind", "midi_program": 73, "channel": 0, "aliases": ["flauta"],                   "fallback": "flute",    "realistic": True,  "is_percussion": False},
    "clarinet":         {"display_name": "Clarinete",       "family": "woodwind", "midi_program": 71, "channel": 0, "aliases": ["clarinete"],                "fallback": "clarinet", "realistic": True,  "is_percussion": False},
    "oboe":             {"display_name": "Oboé",            "family": "woodwind", "midi_program": 68, "channel": 0, "aliases": ["oboe", "oboé"],             "fallback": "oboe",     "realistic": True,  "is_percussion": False},
    "bassoon":          {"display_name": "Fagote",          "family": "woodwind", "midi_program": 70, "channel": 0, "aliases": ["fagote"],                   "fallback": "bassoon",  "realistic": True,  "is_percussion": False},
    "trumpet":          {"display_name": "Trompete",        "family": "brass",    "midi_program": 56, "channel": 0, "aliases": ["trompete"],                 "fallback": "trumpet",  "realistic": True,  "is_percussion": False},
    "trombone":         {"display_name": "Trombone",        "family": "brass",    "midi_program": 57, "channel": 0, "aliases": ["trombone"],                 "fallback": "trombone", "realistic": True,  "is_percussion": False},
    "tuba":             {"display_name": "Tuba",            "family": "brass",    "midi_program": 58, "channel": 0, "aliases": ["tuba"],                     "fallback": "tuba",     "realistic": True,  "is_percussion": False},
    "french_horn":      {"display_name": "Trompa",          "family": "brass",    "midi_program": 60, "channel": 0, "aliases": ["trompa"],                   "fallback": "french_horn", "realistic": True, "is_percussion": False},
    "soprano_sax":      {"display_name": "Sax soprano",     "family": "woodwind", "midi_program": 64, "channel": 0, "aliases": ["saxofone soprano"],         "fallback": "soprano_sax", "realistic": True, "is_percussion": False},
    "alto_sax":         {"display_name": "Sax alto",        "family": "woodwind", "midi_program": 65, "channel": 0, "aliases": ["sax alto"], "fallback": "alto_sax", "realistic": True,  "is_percussion": False},
    "tenor_sax":        {"display_name": "Sax tenor",       "family": "woodwind", "midi_program": 66, "channel": 0, "aliases": ["sax tenor", "saxofone"],     "fallback": "tenor_sax", "realistic": True,  "is_percussion": False},
    "baritone_sax":     {"display_name": "Sax barítono",    "family": "woodwind", "midi_program": 67, "channel": 0, "aliases": ["sax baritono"],              "fallback": "baritone_sax", "realistic": True, "is_percussion": False},
    "accordion":        {"display_name": "Acordeão",        "family": "keys",     "midi_program": 21, "channel": 0, "aliases": ["acordeao", "acordeão", "sanfona"], "fallback": "accordion", "realistic": True, "is_percussion": False},
    "harmonica":        {"display_name": "Gaita",           "family": "woodwind", "midi_program": 22, "channel": 0, "aliases": ["gaita"],                    "fallback": "harmonica", "realistic": True,  "is_percussion": False},
    "synth_lead":       {"display_name": "Sintetizador lead","family": "synth",   "midi_program": 80, "channel": 0, "aliases": ["synth lead", "lead"],        "fallback": "synth_lead", "realistic": True,  "is_percussion": False},
    "synth_pad":        {"display_name": "Sintetizador pad","family": "synth",    "midi_program": 89, "channel": 0, "aliases": ["pad", "synth pad"],          "fallback": "synth_pad", "realistic": True,  "is_percussion": False},
    "synth_bass":       {"display_name": "Baixo sintético", "family": "synth",    "midi_program": 38, "channel": 1, "aliases": ["synth bass"],                "fallback": "synth_bass", "realistic": True,  "is_percussion": False},
    "kick":             {"display_name": "Bumbo",           "family": "drums",    "midi_note": 36, "channel": 9, "aliases": ["bumbo", "kick"], "fallback": "kick", "realistic": True, "is_percussion": True},
    "snare":            {"display_name": "Caixa",           "family": "drums",    "midi_note": 38, "channel": 9, "aliases": ["caixa", "snare"], "fallback": "snare", "realistic": True, "is_percussion": True},
    "hihat_closed":     {"display_name": "Chimbal fechado", "family": "drums",   "midi_note": 42, "channel": 9, "aliases": ["chimbal fechado"], "fallback": "hihat_closed", "realistic": True, "is_percussion": True},
    "hihat_open":       {"display_name": "Chimbal aberto",  "family": "drums",   "midi_note": 46, "channel": 9, "aliases": ["chimbal aberto"],  "fallback": "hihat_open", "realistic": True, "is_percussion": True},
    "crash":            {"display_name": "Prato de ataque", "family": "drums",    "midi_note": 49, "channel": 9, "aliases": ["crash"],           "fallback": "crash", "realistic": True, "is_percussion": True},
    "ride":             {"display_name": "Prato de condução","family": "drums",   "midi_note": 51, "channel": 9, "aliases": ["ride"],            "fallback": "ride",  "realistic": True, "is_percussion": True},
    "timpani":          {"display_name": "Tímpano",         "family": "drums",    "midi_program": 47, "channel": 9, "aliases": ["timpano", "timpani"], "fallback": "timpani", "realistic": True, "is_percussion": True},
    "pandeiro":         {"display_name": "Pandeiro",        "family": "brazilian","midi_note": None, "channel": 9, "aliases": ["pandeiro"],       "fallback": "pandeiro",  "realistic": False, "is_percussion": True},
    "surdo":            {"display_name": "Surdo",           "family": "brazilian","midi_note": None, "channel": 9, "aliases": ["surdo"],          "fallback": "surdo",     "realistic": False, "is_percussion": True},
    "tamborim":         {"display_name": "Tamborim",        "family": "brazilian","midi_note": None, "channel": 9, "aliases": ["tamborim"],       "fallback": "tamborim",  "realistic": False, "is_percussion": True},
    "agogo":            {"display_name": "Agogô",           "family": "brazilian","midi_note": None, "channel": 9, "aliases": ["agogo", "agogô"], "fallback": "agogo",     "realistic": False, "is_percussion": True},
    "cuica":            {"display_name": "Cuíca",           "family": "brazilian","midi_note": None, "channel": 9, "aliases": ["cuica", "cuíca"], "fallback": "cuica",     "realistic": False, "is_percussion": True},
    "berimbau":         {"display_name": "Berimbau",        "family": "brazilian","midi_note": None, "channel": 0, "aliases": ["berimbau"],       "fallback": "berimbau",  "realistic": False, "is_percussion": False},
    "cavaquinho":       {"display_name": "Cavaquinho",      "family": "brazilian","midi_note": None, "channel": 0, "aliases": ["cavaquinho"],     "fallback": "cavaquinho","realistic": False, "is_percussion": False},
    "reco_reco":        {"display_name": "Reco-reco",       "family": "brazilian","midi_note": None, "channel": 9, "aliases": ["reco reco"],      "fallback": "reco_reco", "realistic": False, "is_percussion": True},
    "chocalho":         {"display_name": "Chocalho",        "family": "brazilian","midi_note": None, "channel": 9, "aliases": ["chocalho"],       "fallback": "chocalho",  "realistic": False, "is_percussion": True},
    "maracas":          {"display_name": "Maracas",         "family": "brazilian","midi_note": None, "channel": 9, "aliases": ["maracas"],        "fallback": "maracas",   "realistic": False, "is_percussion": True},
    "tabla":            {"display_name": "Tabla",           "family": "world",    "midi_note": None, "channel": 9, "aliases": ["tabla"],          "fallback": "tabla",     "realistic": False, "is_percussion": True},
    "djembe":           {"display_name": "Djembe",          "family": "world",    "midi_note": None, "channel": 9, "aliases": ["djembe"],         "fallback": "djembe",    "realistic": False, "is_percussion": True},
    "kalimba":          {"display_name": "Kalimba",         "family": "world",    "midi_program": 108,"channel": 0,"aliases": ["kalimba"],        "fallback": "kalimba",   "realistic": True,  "is_percussion": False},
    "sitar":            {"display_name": "Sitar",           "family": "world",    "midi_program": 105,"channel": 0,"aliases": ["sitar"],          "fallback": "sitar",     "realistic": True,  "is_percussion": False},
}

BRAZILIAN_FALLBACK_ONLY = {
    "pandeiro", "surdo", "tamborim", "agogo", "cuica", "berimbau",
    "cavaquinho", "reco_reco", "chocalho", "maracas", "tabla", "djembe",
}


def normalize_instrument_name(name: str) -> str:
    """Remove acentos, caixa, hifens, plural e espaços extras."""
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().replace("-", " ").replace("_", " ")
    n = re.sub(r"\s+", " ", n).strip()
    n = re.sub(r"s$", "", n)
    return n


def resolve_instrument_alias(name: str):
    """Retorna o ID canônico do instrumento ou None."""
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
