#!/usr/bin/env python3
"""Baixa e processa o dataset Kukedlc/suno-ai-music-dataset do Hugging Face.

Este script NÃO baixa os MP3s diretamente (URLs do CDN da Suno bloqueiam hotlinking).
Em vez disso, usa os metadados do dataset, que são suficientes para treinar a IA.

Uso:
    python download_dataset.py                    # processa tudo (usa cache do HF)
    python download_dataset.py --max-songs 50     # processa só 50 músicas
    python download_dataset.py --streaming        # modo streaming (não baixa tudo)
    python download_dataset.py --update-knowledge # atualiza knowledge_base.json

O dataset é baixado em datasets/suno-ai-music-dataset/
O cache do Hugging Face fica em ~/.cache/huggingface/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_dataset")

DATASET_NAME = "Kukedlc/suno-ai-music-dataset"
DATASETS_DIR = Path("datasets")
CACHE_DIR = DATASETS_DIR / "suno-ai-music-dataset"
METADATA_FILE = CACHE_DIR / "metadata.json"
KNOWLEDGE_BASE_FILE = Path("knowledge_base.json")
MINIMUM_SONGS_FOR_TRAINING = 20


def check_dependencies():
    """Verifica se as dependências necessárias estão instaladas."""
    missing = []
    try:
        import datasets  # noqa: F401
    except ImportError:
        missing.append("datasets")

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")

    if missing:
        log.error("Dependências faltando: %s", ", ".join(missing))
        log.error("Execute: pip install %s", " ".join(missing))
        sys.exit(1)


def load_dataset_cached(streaming: bool = False):
    """Carrega o dataset usando o cache do Hugging Face.
    Se já estiver baixado, não baixa de novo.
    """
    from datasets import load_dataset, DownloadConfig

    download_config = DownloadConfig(
        cache_dir=str(DATASETS_DIR / "hf_cache"),
        resume_download=True,
        max_retries=3,
        num_proc=1,
    )

    log.info("Carregando dataset '%s' (cache: %s)...", DATASET_NAME, download_config.cache_dir)
    log.info("Se for a primeira vez, o download pode levar vários minutos (~4 GB).")

    start = time.time()

    ds = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=streaming,
        download_config=download_config,
        trust_remote_code=False,
    )

    elapsed = time.time() - start
    log.info("Dataset carregado em %.1f segundos", elapsed)
    return ds


def normalize_genre(genre: str) -> str:
    """Normaliza nome de gênero para nossa taxonomia interna."""
    if not genre:
        return "generic"

    g = genre.lower().strip()

    mapping = {
        "electrónica": "electronic",
        "electronic": "electronic",
        "electrônica": "electronic",
        "hip-hop": "hiphop",
        "hip hop": "hiphop",
        "hiphop": "hiphop",
        "latin": "latin",
        "latino": "latin",
        "jazz": "jazz",
        "world": "world",
        "rock": "rock",
        "ambient": "ambient",
        "pop": "pop",
        "reggae": "reggae",
        "clásica": "classical",
        "classical": "classical",
        "clássica": "classical",
    }

    for key, value in mapping.items():
        if key in g:
            return value

    return "generic"


def parse_bpm_range(bpm_str: str) -> List[int]:
    """Converte string tipo '125-135' em [125, 135]."""
    if not bpm_str:
        return [100, 130]

    import re
    numbers = re.findall(r"\d+", bpm_str)
    if len(numbers) >= 2:
        return [int(numbers[0]), int(numbers[1])]
    if len(numbers) == 1:
        v = int(numbers[0])
        return [v - 5, v + 5]
    return [100, 130]


def update_knowledge_base(genre_stats: Dict[str, Dict[str, Any]]):
    """Atualiza knowledge_base.json com dados do dataset."""
    if KNOWLEDGE_BASE_FILE.is_file():
        try:
            kb = json.loads(KNOWLEDGE_BASE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            kb = {}
    else:
        kb = {}

    for genre, stats in genre_stats.items():
        bpm_min = stats["bpm_values"][0] if stats["bpm_values"] else 100
        bpm_max = stats["bpm_values"][-1] if stats["bpm_values"] else 130
        scales = stats["scales"][:3] if stats["scales"] else ["major", "minor"]
        instruments = stats["instruments"][:6] if stats["instruments"] else []
        energy = min(1.0, stats["avg_rms"] * 5) if stats["avg_rms"] > 0 else 0.5

        kb[genre] = {
            "bpm_range": [bpm_min, bpm_max],
            "scales": scales,
            "instruments": instruments,
            "rhythm": "varied",
            "energy": round(energy, 2),
            "source": "suno-ai-music-dataset",
            "sample_count": stats["count"],
        }

    KNOWLEDGE_BASE_FILE.write_text(
        json.dumps(kb, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("✓ knowledge_base.json atualizado com %d gêneros", len(kb))


def process_dataset(max_songs: Optional[int] = None, streaming: bool = False,
                    update_kb: bool = True):
    """Processa o dataset e salva metadados.
    
    NOTA: Não baixa os arquivos MP3 (URLs do CDN da Suno bloqueiam hotlinking).
    Usa apenas os metadados, que são suficientes para treinar a IA.
    """
    check_dependencies()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset_cached(streaming=streaming)

    metadata = {
        "dataset": DATASET_NAME,
        "songs": [],
        "genre_stats": {},
        "total_processed": 0,
    }

    genre_stats: Dict[str, Dict[str, Any]] = {}
    processed = 0

    iterator = ds
    if max_songs:
        iterator = ds.take(max_songs) if streaming else list(ds)[:max_songs]

    log.info("Processando músicas (apenas metadados)...")

    for item in iterator:
        if max_songs and processed >= max_songs:
            break

        song_id = item.get("id") or str(processed)
        title = item.get("title", "Sem título")
        genre_raw = item.get("tax_genero") or item.get("tags", "")
        subgenre = item.get("tax_subgenero", "")
        bpm_str = item.get("tax_bpm_rango", "")
        key_info = item.get("tax_key_tipico", "")
        mood = item.get("tax_mood", "")
        prompt = item.get("gpt_description_prompt") or item.get("tags", "")
        duration = item.get("duration", 0)
        is_instrumental = item.get("is_instrumental", True)

        genre = normalize_genre(genre_raw)

        if genre not in genre_stats:
            genre_stats[genre] = {
                "count": 0,
                "bpm_values": [],
                "scales": [],
                "instruments": [],
                "avg_rms": 0.2,
                "durations": [],
            }

        stats = genre_stats[genre]
        stats["count"] += 1

        bpm_range = parse_bpm_range(bpm_str)
        if bpm_range[0] not in stats["bpm_values"]:
            stats["bpm_values"].extend(bpm_range)
            stats["bpm_values"] = sorted(set(stats["bpm_values"]))

        if key_info:
            for scale in ["major", "minor", "dorian", "pentatonic", "blues"]:
                if scale in key_info.lower() and scale not in stats["scales"]:
                    stats["scales"].append(scale)

        if prompt:
            common_instruments = [
                "piano", "violin", "cello", "guitar", "bass", "drums",
                "synth", "pad", "kick", "snare", "hihat",
                "flute", "trumpet", "sax", "accordion"
            ]
            for inst in common_instruments:
                if inst in prompt.lower() and inst not in stats["instruments"]:
                    stats["instruments"].append(inst)

        if duration:
            stats["durations"].append(float(duration))

        song_data = {
            "id": song_id,
            "title": title,
            "genre": genre,
            "genre_raw": genre_raw,
            "subgenre": subgenre,
            "bpm_range": bpm_range,
            "key": key_info,
            "mood": mood,
            "prompt": prompt[:500] if prompt else "",
            "duration": float(duration) if duration else 0,
            "is_instrumental": bool(is_instrumental),
        }

        metadata["songs"].append(song_data)
        processed += 1

        if processed % 10 == 0:
            log.info("  Processadas %d músicas...", processed)

    metadata["total_processed"] = processed
    metadata["genre_stats"] = {
        genre: {
            "count": s["count"],
            "bpm_range": [min(s["bpm_values"]) if s["bpm_values"] else 100,
                          max(s["bpm_values"]) if s["bpm_values"] else 130],
            "scales": s["scales"][:5],
            "instruments": s["instruments"][:8],
            "avg_rms": round(s["avg_rms"], 4),
            "avg_duration": round(
                sum(s["durations"]) / len(s["durations"]) if s["durations"] else 0, 2
            ),
        }
        for genre, s in genre_stats.items()
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("✓ Metadados salvos em %s", METADATA_FILE)
    log.info("  Total processadas: %d", processed)
    log.info("  Gêneros encontrados: %d", len(genre_stats))

    for genre, s in metadata["genre_stats"].items():
        log.info("    %s: %d músicas (BPM %s)", genre, s["count"], s["bpm_range"])

    if update_kb:
        update_knowledge_base(genre_stats)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Baixa o dataset Suno AI Music")

    parser.add_argument("--max-songs", type=int, default=None,
                       help="Número máximo de músicas para processar")
    parser.add_argument("--streaming", action="store_true",
                       help="Modo streaming (não baixa tudo de uma vez)")
    parser.add_argument("--update-knowledge", action="store_true",
                       help="Apenas atualiza knowledge_base.json com metadados existentes")
    parser.add_argument("--stats-only", action="store_true",
                       help="Apenas mostrar estatísticas do dataset")

    args = parser.parse_args()

    if args.update_knowledge:
        if not METADATA_FILE.is_file():
            log.error("Metadados não encontrados. Execute primeiro sem --update-knowledge")
            sys.exit(1)
        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        genre_stats_raw = metadata.get("genre_stats", {})
        genre_stats = {
            g: {
                "count": s["count"],
                "bpm_values": list(range(s["bpm_range"][0], s["bpm_range"][1] + 1)),
                "scales": s.get("scales", []),
                "instruments": s.get("instruments", []),
                "avg_rms": s.get("avg_rms", 0.2),
            }
            for g, s in genre_stats_raw.items()
        }
        update_knowledge_base(genre_stats)
        return

    if args.stats_only:
        if METADATA_FILE.is_file():
            metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
            print(f"Total de músicas: {metadata['total_processed']}")
            print(f"Gêneros: {list(metadata['genre_stats'].keys())}")
            for genre, stats in metadata['genre_stats'].items():
                print(f"  {genre}: {stats['count']} músicas")
        else:
            log.error("Metadados não encontrados.")
        return

    process_dataset(
        max_songs=args.max_songs,
        streaming=args.streaming,
    )


if __name__ == "__main__":
    main()
