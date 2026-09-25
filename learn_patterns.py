#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learn_patterns.py — Nível 2 da IA: aprende padrões de MIDIs REAIS.

Extrai de cada .mid:
  * acordes por compasso → graus da escala → MATRIZ DE MARKOV de transição
    (por gênero). O gerador passa a amostrar progressões com a distribuição
    estatística da música real, não tabelas manuais.
  * levadas de bateria (janelas de 2 compassos, canal 10, quantizadas em
    16avos) → biblioteca por gênero.
  * ritmos de baixo (posições de onset por compasso) → biblioteca.

Fontes (em ordem):
  1. datasets/midi/<gênero>/*.mid        ← seus MIDIs de referência
  2. song_output/*/metadata.json + .mid   ← suas PRÓPRIAS gerações
     (feedback loop real: a IA aprende do que ela própria compôs)

Uso:
    python learn_patterns.py                          # varre tudo
    python learn_patterns.py --dir "caminho" --genre breakcore
    python learn_patterns.py --min-songs 1            # default: 2

Saída: memory/patterns.json  → lido pelo music_generator.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
LOG = logging.getLogger("learn_patterns")

PATTERNS_PATH = Path("memory/patterns.json")
DATASET_DIR = Path("datasets/midi")
SONG_OUTPUT = Path("song_output")

# Perfil Krumhansl-Schmuckler para detecção de tonalidade
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_MAJ_SCALE = {0, 2, 4, 5, 7, 9, 11}
_MIN_SCALE = {0, 2, 3, 5, 7, 8, 10}
_TRIADS = [(0, 4, 7, ""), (0, 3, 7, "m"), (0, 4, 7, ""), (0, 3, 7, "m")]


def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", str(s))
                if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ---------------------------------------------------------------- MIDI parsing

def _load_notes(midi_path: Path) -> Tuple[List[Tuple[float, float, int, int]], float, int]:
    """[(start_s, end_s, pitch, channel)] ordenado + BPM + TPQ."""
    import mido
    mid = mido.MidiFile(str(midi_path))
    tempo = 500000
    for tr in mid.tracks:
        for m in tr:
            if m.type == "set_tempo":
                tempo = m.tempo
                break
        else:
            continue
        break
    spt = tempo / 1e6 / mid.ticks_per_beat
    notes: List[Tuple[float, float, int, int]] = []
    for track in mid.tracks:
        t = 0.0
        active: Dict[int, List[Tuple[float, int]]] = {}
        for m in track:
            t += m.time * spt
            if m.type == "note_on" and m.velocity > 0:
                active.setdefault(m.note, []).append((t, m.channel))
            elif m.type == "note_off" or (m.type == "note_on" and m.velocity == 0):
                lst = active.get(m.note)
                if lst:
                    s, ch = lst.pop(0)
                    notes.append((s, t, m.note, ch))
    notes.sort()
    return notes, 60.0 / (tempo / 1e6), mid.ticks_per_beat


def _detect_key(pcs: Counter) -> Tuple[int, bool]:
    """(pitch class da tônica, é menor?) via correlação K-S."""
    if not pcs:
        return 0, False
    n = sum(pcs.values()) or 1
    hist = [pcs.get(i, 0) / n for i in range(12)]

    def corr(prof, rot):
        h = hist[rot:] + hist[:rot]
        mh, mp = sum(h) / 12, sum(prof) / 12
        num = sum((a - mh) * (b - mp) for a, b in zip(h, prof))
        den = (sum((a - mh) ** 2 for a in h) * sum((b - mp) ** 2 for b in prof)) ** .5
        return num / den if den else 0.0

    best = (-2.0, 0, False)
    for root in range(12):
        c_maj, c_min = corr(_KS_MAJOR, root), corr(_KS_MINOR, root)
        if c_maj > best[0]:
            best = (c_maj, root, False)
        if c_min > best[0]:
            best = (c_min, root, True)
    return best[1], best[2]


def _chord_degrees(notes: List[Tuple[float, float, int, int]], bpm: float
                   ) -> Tuple[List[int], int, bool]:
    """Grau (0-6) do acorde de cada compasso + tonalidade detectada."""
    beat = 60.0 / max(bpm, 1)
    bar = beat * 4
    mel = [n for n in notes if n[3] != 9]
    if not mel:
        return [], 0, False
    root_pc, is_min = _detect_key(Counter(n[2] % 12 for n in mel))
    scale = _MIN_SCALE if is_min else _MAJ_SCALE

    bars: Dict[int, Counter] = defaultdict(Counter)
    for s, e, p, _ in mel:
        bars[int(s // bar)][p % 12] += max(1, int((e - s) / beat) + 1)

    degrees: List[int] = []
    for bi in sorted(bars):
        # melhor tríade: root que maximiza presença dos seus intervalos
        best_deg, best_sc = 0, -99.0
        for d in range(7):
            root = (root_pc + sorted(scale)[d]) % 12
            third = (root + (3 if is_min else 4)) % 12
            fifth = (root + 7) % 12
            h = bars[bi]
            sc = h.get(root, 0) * 3 + h.get(third, 0) * 2 + h.get(fifth, 0) * 2
            sc -= sum(v for pc, v in h.items() if pc not in (root, third, fifth)) * 0.3
            if sc > best_sc:
                best_sc, best_deg = sc, d
        degrees.append(best_deg)
    return degrees, root_pc, is_min


def _markov(degrees: List[int]) -> List[List[int]]:
    m = [[0] * 7 for _ in range(7)]
    for a, b in zip(degrees, degrees[1:]):
        if 0 <= a < 7 and 0 <= b < 7:
            m[a][b] += 1
    # suavização de Laplace fraca (permite transições não vistas)
    for i in range(7):
        for j in range(7):
            m[i][j] += 1
    return m


# ------------------------------------------------------------- bateria / baixo

def _drum_windows(notes: List[Tuple[float, float, int, int]], bpm: float) -> List[str]:
    """Janelas de 2 compassos como assinaturas (16avos × notas)."""
    beat = 60.0 / max(bpm, 1)
    step = beat / 4
    win = beat * 8
    drums = [n for n in notes if n[3] == 9]
    if not drums:
        return []
    out: List[str] = []
    nw = int(max(0, drums[-1][0]) // win) + 1
    for w in range(nw):
        sig = []
        for s, e, p, _ in drums:
            if w * win <= s < (w + 1) * win:
                slot = int(round((s - w * win) / step))
                if 0 <= slot < 32:
                    sig.append(f"{slot}:{p}:{'s' if (e - s) > step * 1.5 else 'x'}")
        if sig:
            out.append(",".join(sig))
    return out


def _bass_rhythms(notes: List[Tuple[float, float, int, int]], bpm: float) -> List[str]:
    """Onsets do canal/baixo mais grave, por compasso (16 slots)."""
    beat = 60.0 / max(bpm, 1)
    step = beat / 4
    cands = defaultdict(list)
    for n in notes:
        if n[3] != 9:
            cands[n[3]].append(n)
    if not cands:
        return []
    ch = min(cands, key=lambda c: sum(n[2] for n in cands[c]) / len(cands[c]))
    out: List[str] = []
    bar = beat * 4
    nbar = int(cands[ch][-1][0] // bar) + 1
    for b in range(nbar):
        sig = ",".join(
            str(int(round((s - b * bar) / step)))
            for s, e, p, c in cands[ch]
            if b * bar <= s < (b + 1) * bar and 0 <= (s - b * bar) / step < 16)
        if sig:
            out.append(sig)
    return out


# -------------------------------------------------------------------- pipeline

def _genre_from_metadata(mid_path: Path) -> Optional[str]:
    meta = mid_path.parent / "metadata.json"
    if meta.exists():
        try:
            st = json.loads(meta.read_text(encoding="utf-8")).get("style")
            if st:
                return str(st)
        except Exception:
            pass
    return None


def learn(sources: List[Tuple[str, Path]], min_songs: int = 2) -> Dict[str, Any]:
    agg: Dict[str, Dict[str, Any]] = {}
    for genre, path in sources:
        try:
            notes, bpm, _ = _load_notes(path)
            if len(notes) < 24:
                continue
            g = agg.setdefault(genre, {"markov": [[0] * 7 for _ in range(7)],
                                       "drums": Counter(), "bass": Counter(),
                                       "files": 0, "keys": Counter()})
            degs, _r, _m = _chord_degrees(notes, bpm)
            if len(degs) >= 3:
                for a, b in zip(degs, degs[1:]):
                    g["markov"][a][b] += 1
            g["drums"].update(_drum_windows(notes, bpm))
            g["bass"].update(_bass_rhythms(notes, bpm))
            g["files"] += 1
            LOG.info("  %-44s %-12s %d notas", path.name, genre, len(notes))
        except Exception as exc:
            LOG.debug("  %s: %s", path.name, exc)
    out: Dict[str, Any] = {"genres": {}}
    for genre, g in agg.items():
        if g["files"] < min_songs:
            LOG.info("Gênero '%s': %d arquivo(s) — abaixo de --min-songs, pulado.",
                     genre, g["files"])
            continue
        mat = g["markov"]
        for i in range(7):          # Laplace + normaliza
            for j in range(7):
                mat[i][j] += 1
            s = sum(mat[i]) or 1
            mat[i] = [round(c / s, 4) for c in mat[i]]
        out["genres"][genre] = {
            "markov": mat,
            "n_songs": g["files"],
            "drum_patterns": [p for p, _ in g["drums"].most_common(24)],
            "bass_rhythms": [p for p, _ in g["bass"].most_common(16)],
        }
        LOG.info("Gênero '%s': %d músicas | %d levadas de bateria | %d ritmos de baixo",
                 genre, g["files"], len(out["genres"][genre]["drum_patterns"]),
                 len(out["genres"][genre]["bass_rhythms"]))
    return out


def collect_sources(explicit: List[Tuple[str, Path]], min_songs: int) -> List[Tuple[str, Path]]:
    srcs: List[Tuple[str, Path]] = list(explicit)
    if DATASET_DIR.is_dir():        # datasets/midi/<gênero>/*.mid
        for gdir in DATASET_DIR.iterdir():
            if gdir.is_dir():
                for f in gdir.glob("*.mid"):
                    srcs.append((_norm(gdir.name) or "generic", f))
                for f in gdir.glob("*.midi"):
                    srcs.append((_norm(gdir.name) or "generic", f))
    if SONG_OUTPUT.is_dir():        # suas próprias gerações (feedback loop)
        for mid in SONG_OUTPUT.rglob("*.mid"):
            g = _genre_from_metadata(mid)
            if g:
                srcs.append((_norm(g) or "generic", mid))
    return srcs


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=None, help="pasta extra de .mid")
    p.add_argument("--genre", default=None, help="gênero da pasta --dir")
    p.add_argument("--min-songs", type=int, default=2)
    a = p.parse_args(argv)

    explicit: List[Tuple[str, Path]] = []
    if a.dir:
        d = Path(a.dir)
        for f in list(d.glob("*.mid")) + list(d.glob("*.midi")):
            explicit.append((_norm(a.genre or d.name) or "generic", f))

    srcs = collect_sources(explicit, a.min_songs)
    if not srcs:
        LOG.error("Nenhum .mid encontrado. Coloque arquivos em %s/<gênero>/ "
                  "ou aponte --dir. (song_output/ também é varrido.)", DATASET_DIR)
        return 1
    LOG.info("Aprendendo de %d arquivos MIDI…", len(srcs))
    data = learn(srcs, a.min_songs)
    if not data["genres"]:
        LOG.error("Nada aprendido (arquivos pequenos demais ou --min-songs alto).")
        return 1
    PATTERNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    old: Dict[str, Any] = {}
    try:
        old = json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    merged = old.get("genres") or {}
    merged.update(data["genres"])
    PATTERNS_PATH.write_text(
        json.dumps({"genres": merged}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    LOG.info("Padrões aprendidos → %s (%d gêneros)", PATTERNS_PATH, len(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main())