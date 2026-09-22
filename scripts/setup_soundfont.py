#!/usr/bin/env python3
"""Utilitário para adicionar um SoundFont local ao projeto."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOUNDFONT_DIR = ROOT / "soundfonts"
CONFIG_PATH = ROOT / "soundfont_config.json"
DEFAULT_TARGET = SOUNDFONT_DIR / "default.sf2"


def load_config():
    if CONFIG_PATH.is_file():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("ERRO: soundfont_config.json inválido.", file=sys.stderr)
            sys.exit(1)

    return {
        "enabled": True,
        "default_soundfont": "soundfonts/default.sf2",
        "renderer": "fluidsynth",
        "sample_rate": 44100,
        "channels": 1,
        "gain": 0.8,
        "fallback_to_synthesis": True,
        "install_fluidsynth_automatically": False,
        "license": {
            "name": None,
            "url": None,
            "commercial_use_allowed": False,
            "notes": "Nenhuma licença declarada.",
        },
    }


def save_config(config):
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Copia SoundFont local")
    parser.add_argument("--source", required=True)
    parser.add_argument("--license-name", default=None)
    parser.add_argument("--license-url", default=None)
    parser.add_argument("--commercial", action="store_true")
    parser.add_argument("--no-license", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()

    if not source.is_file():
        print("ERRO: arquivo não encontrado: " + str(source), file=sys.stderr)
        return 1

    if source.suffix.lower() not in {".sf2", ".sf3"}:
        print("ERRO: o arquivo deve ter extensão .sf2 ou .sf3", file=sys.stderr)
        return 1

    SOUNDFONT_DIR.mkdir(parents=True, exist_ok=True)

    if DEFAULT_TARGET.exists() and not args.force:
        print("ERRO: " + str(DEFAULT_TARGET) + " já existe. Use --force.", file=sys.stderr)
        return 1

    shutil.copy2(source, DEFAULT_TARGET)

    config = load_config()
    config["default_soundfont"] = "soundfonts/default.sf2"
    config["enabled"] = True
    config["fallback_to_synthesis"] = True

    if args.no_license:
        config["license"] = {
            "name": None,
            "url": None,
            "commercial_use_allowed": False,
            "notes": "Nenhuma licença declarada.",
        }
    else:
        config["license"] = {
            "name": args.license_name,
            "url": args.license_url,
            "commercial_use_allowed": bool(args.commercial),
            "notes": "Verifique a licença antes de distribuir.",
        }

    save_config(config)

    print("✓ SoundFont copiado para " + str(DEFAULT_TARGET))
    print("✓ Configuração atualizada em " + str(CONFIG_PATH))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
