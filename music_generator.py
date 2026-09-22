#!/usr/bin/env python3
"""Gerador de música com IA."""
from __future__ import annotations
import argparse, json, logging, random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("music_generator")

from real_instruments import (
    INSTRUMENTS, normalize_instrument_name, resolve_instrument_alias,
    list_families, BRAZILIAN_FALLBACK_ONLY
)
from midi_composer import MidiComposer
from soundfont_renderer import SoundFontRenderer

try:
    from instruments import generate_instrument_sound
    HAS_SYNTH = True
except ImportError:
    HAS_SYNTH = False


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
            candidates = list(INSTRUMENTS.keys())[:5]
            log.warning("Instrumento '%s' não reconhecido. Disponíveis: %s", part, ", ".join(candidates))
    return results


def choose_style_params(style, knowledge_base):
    kb = knowledge_base.get(style, {})
    bpm_range = kb.get("bpm_range", [100, 130])
    scales = kb.get("scales", ["major"])
    instruments = kb.get("instruments", ["piano"])
    energy = kb.get("energy", 0.6)
    return {
        "bpm": random.randint(bpm_range[0], bpm_range[1]),
        "scale": random.choice(scales),
        "instruments": instruments,
        "energy": energy,
    }


def generate_music(args):
    kb_path = Path("knowledge_base.json")
    knowledge_base = {}
    if kb_path.is_file():
        try:
            knowledge_base = json.loads(kb_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("knowledge_base.json inválido")

    style = args.style or "cinematic"
    params = choose_style_params(style, knowledge_base)
    bpm = args.bpm or params["bpm"]
    key = args.key or random.choice(["C", "G", "D", "F"])
    scale = args.scale or params["scale"]

    if args.instruments:
        instrument_ids = parse_instruments_arg(args.instruments)
    else:
        instrument_ids = params["instruments"]

    if not instrument_ids:
        instrument_ids = ["piano"]
        log.info("Nenhum instrumento especificado, usando piano")

    composer = MidiComposer(title=f"Música {style} gerada", bpm=bpm, key=key, scale=scale, style=style)

    track_map = {}
    for inst_id in instrument_ids:
        if inst_id in BRAZILIAN_FALLBACK_ONLY:
            log.info("Instrumento '%s' usará síntese procedural", inst_id)
        track_idx = composer.add_track(inst_id)
        track_map[inst_id] = track_idx

    duration_beats = int((args.duration / 60.0) * bpm)
    bars = max(1, duration_beats // 4)

    for inst_id, track_idx in track_map.items():
        inst_data = INSTRUMENTS[inst_id]
        if inst_data["is_percussion"]:
            composer.add_drum_pattern(track_idx, bars=bars, style=style)
        elif inst_data["family"] in ("bass", "synth") and "bass" in inst_id:
            pitches = composer.scale_pitches(root_octave=2)
            beat = 60.0 / bpm
            for bar in range(bars):
                for b in range(4):
                    t = (bar * 4 + b) * beat
                    p = random.choice(pitches)
                    composer.add_note(track_idx, start=t, duration=beat, pitch=p % 128, velocity=85)
        else:
            composer.add_chord_progression(track_idx, bars=bars)

    output_dir = Path("song_output")
    output_dir.mkdir(exist_ok=True)
    Path("memory").mkdir(exist_ok=True)

    midi_path = output_dir / f"{style}_generated.mid"
    midi_saved = composer.save_midi(midi_path)

    json_path = output_dir / f"{style}_composition.json"
    composer.save_json(json_path)

    wav_path = output_dir / f"{style}_generated.wav"
    soundfont_path = args.soundfont if args.soundfont else None

    if not args.no_soundfont and soundfont_path:
        renderer = SoundFontRenderer(soundfont_path=soundfont_path)
        log.info("Status do SoundFont: %s", renderer.status())
        if renderer.is_available() and midi_saved:
            success = renderer.render_midi(midi_path, wav_path)
            if success:
                log.info("Áudio renderizado: %s", wav_path)
            else:
                log.warning("Falha, usando fallback sintético")
        else:
            log.warning("SoundFont indisponível, usando fallback sintético")
    else:
        log.info("SoundFont desativado, usando síntese procedural")

    metadata = {
        "prompt": args.prompt,
        "style": style,
        "bpm": bpm,
        "key": key,
        "scale": scale,
        "duration": args.duration,
        "instruments": instrument_ids,
        "seed": args.seed,
        "midi_file": str(midi_path) if midi_saved else None,
        "json_file": str(json_path),
        "wav_file": str(wav_path) if wav_path.exists() else None,
        "soundfont_used": not args.no_soundfont and soundfont_path is not None,
    }

    metadata_path = output_dir / f"{style}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    log.info("Geração concluída!")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="IA Music Pro")
    parser.add_argument("--prompt", type=str, default="música gerada por IA")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--style", type=str, default=None)
    parser.add_argument("--bpm", type=int, default=None)
    parser.add_argument("--instruments", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--midi-output", action="store_true")
    parser.add_argument("--soundfont", type=str, default=None)
    parser.add_argument("--no-soundfont", action="store_true")
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--scale", type=str, default=None, choices=["major", "minor", "pentatonic", "blues", "dorian"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--quality-attempts", type=int, default=1)
    parser.add_argument("--list-instruments", action="store_true")
    parser.add_argument("--list-families", action="store_true")

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    if args.list_instruments:
        print("\n=== INSTRUMENTOS DISPONÍVEIS ===\n")
        for name, data in sorted(INSTRUMENTS.items()):
            realistic = "V" if data["realistic"] else "X"
            percussion = "D" if data["is_percussion"] else "M"
            print(f"{percussion} {name:20s} - {data['display_name']:25s} [{data['family']}] Realista: {realistic}")
        return

    if args.list_families:
        print("\n=== FAMÍLIAS DE INSTRUMENTOS ===\n")
        families = list_families()
        for family, instruments in sorted(families.items()):
            print(f"\n{family.upper()}:")
            for inst in instruments:
                print(f"  - {inst}")
        return

    metadata = generate_music(args)
    print("\n" + "="*50)
    print("METADADOS DA GERAÇÃO")
    print("="*50)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
