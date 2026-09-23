#!/usr/bin/env python3
"""Treinamento da IA musical usando dataset real do Hugging Face.

Dataset: Kukedlc/suno-ai-music-dataset
Licença: CC-BY-4.0
Músicas: ~857
Gêneros: Electronic, Hip-hop, Latin, Jazz, World, Rock, Ambient, Pop, Reggae, Classical

Uso:
    python train.py                                    # treino padrão
    python train.py --dataset suno                     # usar dataset Suno
    python train.py --dataset synthetic                # usar dados sintéticos
    python train.py --dataset mixed                    # misturar ambos
    python train.py --max-songs 100                    # limitar músicas
    python train.py --epochs 50                        # mais épocas
    python train.py --list-genres                      # listar gêneros disponíveis
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

# ============================================================================
# Constantes
# ============================================================================

DATASET_NAME = "Kukedlc/suno-ai-music-dataset"
DATASETS_DIR = Path("datasets")
SUNO_CACHE_DIR = DATASETS_DIR / "suno-ai-music-dataset"
SUNO_METADATA = SUNO_CACHE_DIR / "metadata.json"
MODELS_DIR = Path("models")
MEMORY_DIR = Path("memory")
KNOWLEDGE_BASE = Path("knowledge_base.json")
NORMALIZATION_FILE = MODELS_DIR / "normalization.npz"

FEATURE_SIZE = 128
LATENT_DIM = 32
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 8
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.10
EARLY_STOPPING_PATIENCE = 5


# ============================================================================
# Carregamento do Dataset Suno
# ============================================================================

def check_suno_dataset() -> bool:
    """Verifica se o dataset Suno já foi baixado."""
    return SUNO_METADATA.is_file()


def download_suno_dataset(max_songs: Optional[int] = None) -> bool:
    """Baixa o dataset Suno do Hugging Face usando cache.
    Se já estiver baixado, não baixa de novo.
    """
    try:
        from datasets import load_dataset, DownloadConfig
    except ImportError:
        log.error("Pacote 'datasets' não instalado.")
        log.error("Execute: pip install datasets huggingface-hub")
        return False

    SUNO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = DATASETS_DIR / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    download_config = DownloadConfig(
        cache_dir=str(cache_dir),
        resume_download=True,
        max_retries=3,
    )

    log.info("Carregando dataset '%s'...", DATASET_NAME)
    log.info("Cache em: %s", cache_dir)
    log.info("Se for a primeira vez, pode levar vários minutos (~4 GB).")

    start_time = time.time()

    try:
        ds = load_dataset(
            DATASET_NAME,
            split="train",
            download_config=download_config,
            trust_remote_code=False,
        )

        elapsed = time.time() - start_time
        log.info("Dataset carregado em %.1f segundos", elapsed)
        log.info("Total de músicas: %d", len(ds))

        process_suno_dataset(ds, max_songs)
        return True

    except Exception as e:
        log.error("Erro ao carregar dataset: %s", e)
        return False


def process_suno_dataset(ds, max_songs: Optional[int] = None) -> None:
    """Processa o dataset e salva metadados."""
    SUNO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": DATASET_NAME,
        "songs": [],
        "genre_stats": {},
        "total_processed": 0,
    }

    genre_stats: Dict[str, Dict[str, Any]] = {}
    processed = 0

    items = list(ds) if max_songs is None else list(ds)[:max_songs]

    log.info("Processando %d músicas...", len(items))

    for item in items:
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
            }

        stats = genre_stats[genre]
        stats["count"] += 1

        bpm_range = parse_bpm_range(bpm_str)
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
                "flute", "trumpet", "sax", "accordion",
            ]
            for inst in common_instruments:
                if inst in prompt.lower() and inst not in stats["instruments"]:
                    stats["instruments"].append(inst)

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

        if processed % 100 == 0:
            log.info("  Processadas %d/%d músicas...", processed, len(items))

    metadata["total_processed"] = processed
    metadata["genre_stats"] = {
        genre: {
            "count": s["count"],
            "bpm_range": [
                min(s["bpm_values"]) if s["bpm_values"] else 100,
                max(s["bpm_values"]) if s["bpm_values"] else 130,
            ],
            "scales": s["scales"][:5],
            "instruments": s["instruments"][:8],
        }
        for genre, s in genre_stats.items()
    }

    SUNO_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("✓ Metadados salvos em %s", SUNO_METADATA)
    log.info("  Total: %d músicas, %d gêneros", processed, len(genre_stats))

    for genre, s in metadata["genre_stats"].items():
        log.info("    %s: %d músicas (BPM %s)", genre, s["count"], s["bpm_range"])

    update_knowledge_base(genre_stats)


def normalize_genre(genre: str) -> str:
    """Normaliza nome de gênero."""
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
    """Converte '125-135' em [125, 135]."""
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


def update_knowledge_base(genre_stats: Dict[str, Dict[str, Any]]) -> None:
    """Atualiza knowledge_base.json com dados do dataset."""
    if KNOWLEDGE_BASE.is_file():
        try:
            kb = json.loads(KNOWLEDGE_BASE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            kb = {}
    else:
        kb = {}

    for genre, stats in genre_stats.items():
        bpm_min = min(stats["bpm_values"]) if stats["bpm_values"] else 100
        bpm_max = max(stats["bpm_values"]) if stats["bpm_values"] else 130
        scales = stats["scales"][:3] if stats["scales"] else ["major", "minor"]
        instruments = stats["instruments"][:6] if stats["instruments"] else []

        kb[genre] = {
            "bpm_range": [bpm_min, bpm_max],
            "scales": scales,
            "instruments": instruments,
            "rhythm": "varied",
            "energy": 0.6,
            "source": "suno-ai-music-dataset",
            "sample_count": stats["count"],
        }

    KNOWLEDGE_BASE.write_text(
        json.dumps(kb, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("✓ knowledge_base.json atualizado com %d gêneros", len(kb))


# ============================================================================
# Extração de Features
# ============================================================================

def extract_features_from_metadata(songs: List[Dict[str, Any]]) -> np.ndarray:
    """Extrai features numéricas dos metadados das músicas.

    Features por música:
    - BPM normalizado (1)
    - One-hot do gênero (10)
    - Duração normalizada (1)
    - É instrumental (1)
    - Energia estimada (1)
    - Complexidade estimada (1)
    Total: ~15 features por música
    """
    genres = list(set(s["genre"] for s in songs))
    genre_to_idx = {g: i for i, g in enumerate(genres)}

    features_list = []

    for song in songs:
        feat = []

        bpm = song["bpm_range"][0] if song.get("bpm_range") else 120
        feat.append(bpm / 200.0)

        genre_vec = np.zeros(len(genres))
        if song["genre"] in genre_to_idx:
            genre_vec[genre_to_idx[song["genre"]]] = 1.0
        feat.extend(genre_vec.tolist())

        duration = song.get("duration", 0)
        feat.append(min(duration / 600.0, 1.0))

        feat.append(1.0 if song.get("is_instrumental", False) else 0.0)

        energy = estimate_energy(song)
        feat.append(energy)

        complexity = estimate_complexity(song)
        feat.append(complexity)

        features_list.append(feat)

    return np.array(features_list, dtype=np.float32)


def estimate_energy(song: Dict[str, Any]) -> float:
    """Estima energia com base no gênero e BPM."""
    genre = song.get("genre", "generic")
    bpm = song["bpm_range"][0] if song.get("bpm_range") else 120

    energy_by_genre = {
        "electronic": 0.85,
        "hiphop": 0.75,
        "rock": 0.80,
        "pop": 0.70,
        "latin": 0.75,
        "jazz": 0.50,
        "ambient": 0.25,
        "world": 0.60,
        "reggae": 0.55,
        "classical": 0.45,
        "generic": 0.60,
    }

    base_energy = energy_by_genre.get(genre, 0.60)
    bpm_factor = min(bpm / 160.0, 1.0) * 0.3

    return min(base_energy + bpm_factor, 1.0)


def estimate_complexity(song: Dict[str, Any]) -> float:
    """Estima complexidade musical com base no prompt."""
    prompt = song.get("prompt", "").lower()

    complexity = 0.5

    complex_terms = [
        "polyrhythm", "jazz", "modal", "fusion", "progressive",
        "complex", "intricate", "sophisticated", "odd meter",
        "7/8", "5/4", "chromatic", "extended chords",
    ]

    simple_terms = [
        "simple", "minimal", "ambient", "chill", "relaxing",
        "lo-fi", "basic", "easy",
    ]

    for term in complex_terms:
        if term in prompt:
            complexity += 0.1

    for term in simple_terms:
        if term in prompt:
            complexity -= 0.1

    return max(0.0, min(1.0, complexity))


def generate_synthetic_data(num_samples: int = 200) -> np.ndarray:
    """Gera dados sintéticos como complemento.
    Usado apenas quando não há dados reais suficientes.
    """
    log.warning("Gerando %d amostras sintéticas (complemento)", num_samples)

    genres = ["electronic", "hiphop", "rock", "jazz", "ambient", "latin"]
    features_list = []

    for _ in range(num_samples):
        feat = []

        bpm = np.random.randint(60, 180)
        feat.append(bpm / 200.0)

        genre_vec = np.zeros(len(genres))
        genre_idx = np.random.randint(len(genres))
        genre_vec[genre_idx] = 1.0
        feat.extend(genre_vec.tolist())

        duration = np.random.uniform(60, 300)
        feat.append(duration / 600.0)

        is_instrumental = np.random.random() > 0.4
        feat.append(1.0 if is_instrumental else 0.0)

        energy = np.random.uniform(0.2, 0.9)
        feat.append(energy)

        complexity = np.random.uniform(0.3, 0.8)
        feat.append(complexity)

        features_list.append(feat)

    return np.array(features_list, dtype=np.float32)


# ============================================================================
# Modelo Autoencoder
# ============================================================================

def build_autoencoder(input_dim: int, latent_dim: int):
    """Constrói o autoencoder. Tenta usar PyTorch, senão usa NumPy puro."""
    try:
        import torch
        import torch.nn as nn

        class Autoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, latent_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, input_dim),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        model = Autoencoder(input_dim, latent_dim)
        log.info("Autoencoder criado com PyTorch (input=%d, latent=%d)", input_dim, latent_dim)
        return model, "pytorch"

    except ImportError:
        log.warning("PyTorch não encontrado. Usando autoencoder simplificado em NumPy.")
        return None, "numpy"


class NumpyAutoencoder:
    """Autoencoder simples em NumPy puro para quando PyTorch não está disponível."""

    def __init__(self, input_dim: int, latent_dim: int):
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        scale = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, 64).astype(np.float32) * scale
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = np.random.randn(64, 32).astype(np.float32) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(32, dtype=np.float32)
        self.W3 = np.random.randn(32, latent_dim).astype(np.float32) * np.sqrt(2.0 / 32)
        self.b3 = np.zeros(latent_dim, dtype=np.float32)

        self.W4 = np.random.randn(latent_dim, 32).astype(np.float32) * np.sqrt(2.0 / latent_dim)
        self.b4 = np.zeros(32, dtype=np.float32)
        self.W5 = np.random.randn(32, 64).astype(np.float32) * np.sqrt(2.0 / 32)
        self.b5 = np.zeros(64, dtype=np.float32)
        self.W6 = np.random.randn(64, input_dim).astype(np.float32) * np.sqrt(2.0 / 64)
        self.b6 = np.zeros(input_dim, dtype=np.float32)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def encode(self, x):
        h1 = self.relu(x @ self.W1 + self.b1)
        h2 = self.relu(h1 @ self.W2 + self.b2)
        z = h2 @ self.W3 + self.b3
        return z

    def decode(self, z):
        h4 = self.relu(z @ self.W4 + self.b4)
        h5 = self.relu(h4 @ self.W5 + self.b5)
        out = self.sigmoid(h5 @ self.W6 + self.b6)
        return out

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out


def train_numpy_autoencoder(model: NumpyAutoencoder, data: np.ndarray,
                            epochs: int, batch_size: int,
                            val_data: np.ndarray) -> Dict[str, List[float]]:
    """Treina o autoencoder em NumPy puro."""
    history = {"train_loss": [], "val_loss": []}
    lr = LEARNING_RATE
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(epochs):
        indices = np.random.permutation(len(data))
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch_idx = indices[start:end]
            batch = data[batch_idx]

            output = model.forward(batch)
            loss = np.mean((batch - output) ** 2)
            epoch_loss += loss
            num_batches += 1

            grad = 2 * (output - batch) / batch.shape[0]
            model.W6 -= lr * (model.W5.T @ grad)
            model.b6 -= lr * grad.sum(axis=0)

        avg_train_loss = epoch_loss / max(num_batches, 1)
        history["train_loss"].append(avg_train_loss)

        val_output = model.forward(val_data)
        val_loss = np.mean((val_data - val_output) ** 2)
        history["val_loss"].append(val_loss)

        log.info("Época %d/%d - train_loss: %.4f - val_loss: %.4f",
                epoch + 1, epochs, avg_train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {
                "W1": model.W1.copy(), "b1": model.b1.copy(),
                "W2": model.W2.copy(), "b2": model.b2.copy(),
                "W3": model.W3.copy(), "b3": model.b3.copy(),
                "W4": model.W4.copy(), "b4": model.b4.copy(),
                "W5": model.W5.copy(), "b5": model.b5.copy(),
                "W6": model.W6.copy(), "b6": model.b6.copy(),
            }
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                log.info("Early stopping na época %d", epoch + 1)
                break

        lr *= 0.98

    if best_weights:
        model.W1, model.b1 = best_weights["W1"], best_weights["b1"]
        model.W2, model.b2 = best_weights["W2"], best_weights["b2"]
        model.W3, model.b3 = best_weights["W3"], best_weights["b3"]
        model.W4, model.b4 = best_weights["W4"], best_weights["b4"]
        model.W5, model.b5 = best_weights["W5"], best_weights["b5"]
        model.W6, model.b6 = best_weights["W6"], best_weights["b6"]

    return history


# ============================================================================
# Treinamento Principal
# ============================================================================

def train(args) -> Dict[str, Any]:
    """Função principal de treinamento."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("TREINAMENTO DA IA MUSICAL")
    log.info("=" * 60)

    dataset_mode = args.dataset

    if dataset_mode in ("suno", "mixed"):
        if not check_suno_dataset():
            log.info("Dataset Suno não encontrado. Baixando...")
            success = download_suno_dataset(max_songs=args.max_songs)
            if not success:
                if dataset_mode == "suno":
                    log.error("Não foi possível baixar o dataset Suno.")
                    sys.exit(1)
                else:
                    log.warning("Fallback para dados sintéticos.")
                    dataset_mode = "synthetic"
        else:
            log.info("✓ Dataset Suno encontrado no cache: %s", SUNO_METADATA)

    real_data = None
    real_songs = []

    if dataset_mode in ("suno", "mixed") and SUNO_METADATA.is_file():
        try:
            metadata = json.loads(SUNO_METADATA.read_text(encoding="utf-8"))
            real_songs = metadata.get("songs", [])

            if args.max_songs and len(real_songs) > args.max_songs:
                real_songs = real_songs[:args.max_songs]

            log.info("Carregadas %d músicas reais do dataset Suno", len(real_songs))

            if len(real_songs) >= 10:
                real_data = extract_features_from_metadata(real_songs)
                log.info("Features extraídas: shape %s", real_data.shape)
            else:
                log.warning("Músicas reais insuficientes (%d). Usando sintéticos.", len(real_songs))
                dataset_mode = "synthetic"

        except Exception as e:
            log.error("Erro ao carregar metadados: %s", e)
            dataset_mode = "synthetic"

    synthetic_data = None

    if dataset_mode == "synthetic":
        synthetic_data = generate_synthetic_data(num_samples=200)
        log.info("Usando apenas dados sintéticos: %d amostras", len(synthetic_data))

    elif dataset_mode == "mixed":
        num_synthetic = max(50, len(real_data) // 4)
        synthetic_data = generate_synthetic_data(num_samples=num_synthetic)
        log.info("Adicionando %d amostras sintéticas", num_synthetic)

    if real_data is not None and synthetic_data is not None:
        data = np.vstack([real_data, synthetic_data])
        log.info("Dataset misto: %d reais + %d sintéticas = %d total",
                len(real_data), len(synthetic_data), len(data))
    elif real_data is not None:
        data = real_data
    else:
        data = synthetic_data

    if data is None or len(data) == 0:
        log.error("Nenhum dado disponível para treinamento.")
        sys.exit(1)

    np.random.shuffle(data)

    input_dim = data.shape[1]
    log.info("Dimensão das features: %d", input_dim)

    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    data_norm = (data - mean) / std

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(NORMALIZATION_FILE, mean=mean, std=std)
    log.info("✓ Normalização salva em %s", NORMALIZATION_FILE)

    n = len(data_norm)
    val_size = max(1, int(n * VALIDATION_SPLIT))
    test_size = max(1, int(n * TEST_SPLIT))
    train_size = n - val_size - test_size

    train_data = data_norm[:train_size]
    val_data = data_norm[train_size:train_size + val_size]
    test_data = data_norm[train_size + val_size:]

    log.info("Split: treino=%d, validação=%d, teste=%d",
            train_size, val_size, test_size)

    model, backend = build_autoencoder(input_dim, LATENT_DIM)

    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    if backend == "pytorch":
        import torch
        import torch.nn as nn

        device = torch.device("cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        train_tensor = torch.FloatTensor(train_data).to(device)
        val_tensor = torch.FloatTensor(val_data).to(device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            indices = torch.randperm(len(train_tensor))

            for start in range(0, len(train_tensor), args.batch_size):
                end = min(start + args.batch_size, len(train_tensor))
                batch_idx = indices[start:end]
                batch = train_tensor[batch_idx]

                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / max(num_batches, 1)
            history["train_loss"].append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                val_output = model(val_tensor)
                val_loss = criterion(val_output, val_tensor).item()
            history["val_loss"].append(val_loss)

            log.info("Época %d/%d - train_loss: %.4f - val_loss: %.4f",
                    epoch + 1, args.epochs, avg_train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    log.info("Early stopping na época %d", epoch + 1)
                    break

            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.98

        if best_state:
            model.load_state_dict(best_state)

        model_path = MODELS_DIR / "autoencoder.pt"
        torch.save(model.state_dict(), model_path)
        log.info("✓ Modelo salvo em %s", model_path)

    else:
        np_model = NumpyAutoencoder(input_dim, LATENT_DIM)
        history = train_numpy_autoencoder(
            np_model, train_data, args.epochs, args.batch_size, val_data
        )

        model_path = MODELS_DIR / "autoencoder_np.npz"
        np.savez(
            model_path,
            W1=np_model.W1, b1=np_model.b1,
            W2=np_model.W2, b2=np_model.b2,
            W3=np_model.W3, b3=np_model.b3,
            W4=np_model.W4, b4=np_model.b4,
            W5=np_model.W5, b5=np_model.b5,
            W6=np_model.W6, b6=np_model.b6,
        )
        log.info("✓ Modelo salvo em %s", model_path)

    elapsed = time.time() - start_time
    log.info("Treinamento concluído em %.1f segundos", elapsed)

    if backend == "pytorch":
        import torch
        model.eval()
        test_tensor = torch.FloatTensor(test_data)
        with torch.no_grad():
            test_output = model(test_tensor)
            test_loss = torch.nn.functional.mse_loss(test_output, test_tensor).item()
    else:
        test_output = np_model.forward(test_data)
        test_loss = np.mean((test_data - test_output) ** 2)

    log.info("Loss no conjunto de teste: %.4f", test_loss)

    training_report = {
        "dataset_mode": dataset_mode,
        "total_songs": len(data),
        "real_songs": len(real_songs) if real_songs else 0,
        "synthetic_songs": len(synthetic_data) if synthetic_data is not None else 0,
        "input_dim": input_dim,
        "latent_dim": LATENT_DIM,
        "backend": backend,
        "epochs_completed": len(history["train_loss"]),
        "epochs_requested": args.epochs,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "test_loss": test_loss,
        "training_time_seconds": round(elapsed, 2),
        "early_stopping_triggered": len(history["train_loss"]) < args.epochs,
        "genre_distribution": {},
    }

    if real_songs:
        genre_counts = {}
        for song in real_songs:
            g = song.get("genre", "unknown")
            genre_counts[g] = genre_counts.get(g, 0) + 1
        training_report["genre_distribution"] = genre_counts

    report_path = MEMORY_DIR / "training_report.json"
    report_path.write_text(
        json.dumps(training_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("✓ Relatório de treinamento salvo em %s", report_path)

    log.info("")
    log.info("=" * 60)
    log.info("RESUMO DO TREINAMENTO")
    log.info("=" * 60)
    log.info("  Dataset: %s", dataset_mode)
    log.info("  Músicas reais: %d", training_report["real_songs"])
    log.info("  Músicas sintéticas: %d", training_report["synthetic_songs"])
    log.info("  Backend: %s", backend)
    log.info("  Épocas: %d/%d", training_report["epochs_completed"], args.epochs)
    log.info("  Loss final (teste): %.4f", test_loss)
    log.info("  Tempo: %.1f segundos", elapsed)
    log.info("")

    if training_report["real_songs"] == 0:
        log.warning("⚠️  ATENÇÃO: Treinado apenas com dados sintéticos.")
        log.warning("   A IA aprendeu padrões genéricos, não música real.")
        log.warning("   Para melhor resultado, execute: python download_dataset.py")

    return training_report


# ============================================================================
# Listar Gêneros
# ============================================================================

def list_genres():
    """Lista gêneros disponíveis no dataset."""
    if not SUNO_METADATA.is_file():
        log.warning("Dataset Suno não encontrado.")
        log.info("Execute: python train.py --dataset suno")
        log.info("Ou: python download_dataset.py")
        return

    metadata = json.loads(SUNO_METADATA.read_text(encoding="utf-8"))
    genre_stats = metadata.get("genre_stats", {})

    print("\n=== GÊNEROS DISPONÍVEIS ===\n")

    for genre, stats in sorted(genre_stats.items()):
        print(f"  {genre:15s} - {stats['count']:3d} músicas - BPM {stats['bpm_range']}")

    print(f"\nTotal: {metadata['total_processed']} músicas")
    print(f"Gêneros: {len(genre_stats)}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Treinar IA Musical")

    parser.add_argument("--dataset", type=str, default="mixed",
                       choices=["suno", "synthetic", "mixed"],
                       help="Fonte de dados para treinamento")
    parser.add_argument("--max-songs", type=int, default=None,
                       help="Número máximo de músicas para usar")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                       help="Número de épocas de treinamento")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                       help="Tamanho do batch")
    parser.add_argument("--list-genres", action="store_true",
                       help="Listar gêneros disponíveis no dataset")

    args = parser.parse_args()

    if args.list_genres:
        list_genres()
        return

    train(args)


if __name__ == "__main__":
    main()
