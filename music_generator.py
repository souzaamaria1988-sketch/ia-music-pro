#!/usr/bin/env python3
"""Gerador de música com IA."""
from __future__ import annotations
import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("music_generator")

from real_instruments import INSTRUMENTS, resolve_instrument_alias, list_families
from midi_composer import MidiComposer
from soundfont_renderer import SoundFontRenderer


def parse_instruments_arg(instruments_str):
    if not instruments_str:
        return []
    results = []
    for part in instruments_str.split(","):
        part = part.strip()
        if not part:
            continue
        resolved = resolve_instrument_alias(part)
        if resolved:
            results.append(resolved)
        else:
            log.warning("Instrumento não reconhecido: %s", part)
    return results


def generate_music(args):
    style = args.style or "cinematic"
    bpm = args.bpm or random.randint(90, 130)
    key = args.key or random.choice(["C", "G", "D", "F"])
    scale = args.scale or "major"

    if args.instruments:
        instrument_ids = parse_instruments_arg(args.instruments)
    else:
        instrument_ids = ["piano"]

    composer = MidiComposer(
        title="Música " + style + " gerada",
        bpm=bpm,
        key=key,
        scale=scale,
        style=style,
    )

    for inst_id in instrument_ids:
        track_idx = composer.add_track(inst_id)
        if INSTRUMENTS[inst_id]["is_percussion"]:
            continue
        composer.add_chord_progression(track_idx, bars=max(1, int(args.duration / 4)))

    output_dir = Path("song_output")
    output_dir.mkdir(exist_ok=True)
    Path("memory").mkdir(exist_ok=True)

    midi_path = output_dir / (style + "_generated.mid")
    midi_saved = composer.save_midi(midi_path)

    json_path = output_dir / (style + "_composition.json")
    composer.save_json(json_path)

    wav_path = output_dir / (style + "_generated.wav")

    if args.soundfont and not args.no_soundfont:
        renderer = SoundFontRenderer(soundfont_path=args.soundfont)
        if renderer.is_available() and midi_saved:
            renderer.render_midi(midi_path, wav_path)
        else:
            log.warning("SoundFont/FluidSynth indisponível. Usando fallback sintético.")
    else:
        log.info("SoundFont desativado. Usando síntese procedural.")

    metadata = {
        "prompt": args.prompt,
        "style": style,
        "bpm": bpm,
        "key": key,
        "scale": scale,
        "duration": args.duration,
        "instruments": instrument_ids,
        "midi_file": str(midi_path) if midi_saved else None,
        "json_file": str(json_path),
        "wav_file": str(wav_path) if wav_path.exists() else None,
    }

    metadata_path = output_dir / (style + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    return metadata


def main():
    parser = argparse.ArgumentParser(description="IA Music Pro")
    parser.add_argument("--prompt", type=str, default="música gerada por IA")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--style", type=str, default=None)
    parser.add_argument("--bpm", type=int, default=None)
    parser.add_argument("--instruments", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--soundfont", type=str, default=None)
    parser.add_argument("--no-soundfont", action="store_true")
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--scale", type=str, default=None)
    parser.add_argument("--list-instruments", action="store_true")
    parser.add_argument("--list-families", action="store_true")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.list_instruments:
        for name, data in sorted(INSTRUMENTS.items()):
            print(name + " - " + data["display_name"])
        return

    if args.list_families:
        families = list_families()
        for family, instruments in sorted(families.items()):
            print(family.upper() + ": " + ", ".join(instruments))
        return

    metadata = generate_music(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
