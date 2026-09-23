#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — Treinamento do autoencoder de estilo do IA Music Pro.

Fluxo:
    1. Lê os metadados do dataset Suno salvos por download_dataset.py
       (datasets/suno-ai-music-dataset/metadata.json).
    2. Converte cada música para um vetor de features simbólicas em um
       SCHEMA FIXO (fonte única de verdade — ver GENRE_SCHEMA abaixo).
    3. Gera amostras sintéticas de complemento no MESMO schema.
    4. Treina um autoencoder denso (PyTorch, com fallback em NumPy puro)
       com early stopping na validação.
    5. Salva modelo + metadados e um relatório em memory/ (feedback loop).

Schema de features (FEATURE_DIM = 17):
    [0]        bpm normalizado (bpm / 200, bpm limitado a [30, 200])
    [1 .. 12]  one-hot de gênero (GENRE_SCHEMA — 12 gêneros)
    [13]       duração normalizada (s / 300, limitada a [0, 300])
    [14]       energia       [0, 1]
    [15]       complexidade  [0, 1]
    [16]       valência      [0, 1]

IMPORTANTE — fonte única de verdade:
    Dados reais E sintéticos são construídos por song_to_features(), então é
    estruturalmente impossível as dimensões divergirem (corrige o ValueError
    do np.vstack reportado no CI: shape (100, 10) vs (50, 11)).
    Se mudar GENRE_SCHEMA ou as features, faça bump da chave de cache
    'models-vN' no workflow train_model.yml — a dimensionalidade muda e o
    modelo em cache fica incompatível (load_autoencoder() detecta e recusa).

Uso:
    python train.py --dataset mixed --max-songs 100 --epochs 20 --batch-size 8
    python train.py --dataset suno --max-songs 0            # dataset completo
    python train.py --dataset synthetic --synthetic-samples 200

Outros módulos (music_generator.py, music_intelligence.py) devem carregar o
modelo via load_autoencoder() — nunca abrir os arquivos diretamente.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# PyTorch é OPCIONAL: sem ele, treinamos com um autoencoder em NumPy puro.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

LOG = logging.getLogger("ia_music_pro.train")


def _setup_logging(verbose: bool = False) -> logging.Logger:
    """Logger no formato do restante do projeto: 'HH:MM:SS [LEVEL] mensagem'."""
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not LOG.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                              datefmt="%H:%M:%S")
        )
        LOG.addHandler(handler)
    LOG.propagate = False
    return LOG


# ---------------------------------------------------------------------------
# Caminhos padrão (relativos à raiz do projeto)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "suno-ai-music-dataset" / "metadata.json"
MODELS_DIR = PROJECT_ROOT / "models"
MEMORY_DIR = PROJECT_ROOT / "memory"

# Complemento sintético no modo 'mixed': com 100 músicas reais gera 50
# sintéticas (mesmo comportamento do log do CI: "Gerando 50 amostras...").
MIN_TRAIN_SAMPLES = 150

# =============================================================================
# SCHEMA DE FEATURES — FONTE ÚNICA DE VERDADE
# =============================================================================

GENRE_SCHEMA: List[str] = [
    "generic", "electronic", "hiphop", "rock", "jazz",
    "ambient", "latin", "pop", "samba", "bossa", "folk", "classical",
]
GENRE_TO_IDX: Dict[str, int] = {g: i for i, g in enumerate(GENRE_SCHEMA)}

_GENRE_ALIASES: Dict[str, str] = {
    "hip hop": "hiphop", "hip-hop": "hiphop", "rap": "hiphop", "trap": "hiphop",
    "edm": "electronic", "house": "electronic", "techno": "electronic",
    "bossa nova": "bossa", "mpb": "bossa",
    "lo-fi": "ambient", "lofi": "ambient",
    "world": "latin", "salsa": "latin",
    "soundtrack": "generic", "other": "generic",
}

_TAIL_FEATURES: Tuple[str, ...] = ("duration", "energy", "complexity", "valence")

FEATURE_NAMES: List[str] = (
    ["bpm"] + [f"genre_{g}" for g in GENRE_SCHEMA] + list(_TAIL_FEATURES)
)
FEATURE_DIM: int = len(FEATURE_NAMES)  # 1 + 12 + 4 = 17
GENRE_SLICE: Tuple[int, int] = (1, 1 + len(GENRE_SCHEMA))


def normalize_genre(raw: Any) -> str:
    """Normaliza (lowercase + aliases) e faz fallback para 'generic' (índice 0).

    Metadados reais podem vir como 'Hip Hop, EDM, Party' — usamos o primeiro
    tag. Gênero desconhecido NÃO quebra: cai no bucket 'generic'.
    """
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else "generic"
    g = str(raw or "").strip().lower()
    for sep in (",", ";", "/"):
        if sep in g:
            g = g.split(sep)[0].strip()
    g = _GENRE_ALIASES.get(g, g)
    return g if g in GENRE_TO_IDX else "generic"


# =============================================================================
# PRIORES MUSICAIS (dados sintéticos e valores default para dados reais)
# =============================================================================

GENRE_PROFILE: Dict[str, Dict[str, Tuple[float, float]]] = {
    # gênero:    bpm         energia      complexidade  valência
    "generic":    {"bpm": (60, 180),  "energy": (0.20, 0.90), "complexity": (0.20, 0.80), "valence": (0.20, 0.90)},
    "electronic": {"bpm": (110, 145), "energy": (0.55, 0.95), "complexity": (0.20, 0.55), "valence": (0.40, 0.95)},
    "hiphop":     {"bpm": (78, 105),  "energy": (0.50, 0.85), "complexity": (0.45, 0.85), "valence": (0.25, 0.80)},
    "rock":       {"bpm": (110, 165), "energy": (0.60, 1.00), "complexity": (0.30, 0.65), "valence": (0.30, 0.85)},
    "jazz":       {"bpm": (85, 175),  "energy": (0.30, 0.75), "complexity": (0.60, 0.95), "valence": (0.30, 0.80)},
    "ambient":    {"bpm": (55, 90),   "energy": (0.05, 0.40), "complexity": (0.10, 0.45), "valence": (0.20, 0.70)},
    "latin":      {"bpm": (90, 135),  "energy": (0.50, 0.90), "complexity": (0.40, 0.75), "valence": (0.50, 0.95)},
    "pop":        {"bpm": (95, 130),  "energy": (0.50, 0.85), "complexity": (0.25, 0.60), "valence": (0.45, 0.90)},
    "samba":      {"bpm": (85, 110),  "energy": (0.55, 0.90), "complexity": (0.50, 0.80), "valence": (0.55, 0.95)},
    "bossa":      {"bpm": (68, 100),  "energy": (0.30, 0.60), "complexity": (0.55, 0.85), "valence": (0.50, 0.85)},
    "folk":       {"bpm": (80, 125),  "energy": (0.25, 0.65), "complexity": (0.35, 0.70), "valence": (0.35, 0.80)},
    "classical":  {"bpm": (60, 130),  "energy": (0.20, 0.70), "complexity": (0.50, 0.95), "valence": (0.25, 0.85)},
}

GENRE_COMPLEXITY_PRIOR: Dict[str, float] = {
    "jazz": 0.75, "classical": 0.65, "bossa": 0.65, "samba": 0.60,
    "hiphop": 0.60, "folk": 0.55, "latin": 0.55, "rock": 0.45,
    "pop": 0.40, "electronic": 0.35, "ambient": 0.30, "generic": 0.50,
}

# (palavras-chave, (energia, valência)) — aplicadas sobre humor/prompt
_MOOD_RULES: List[Tuple[Tuple[str, ...], Tuple[float, float]]] = [
    (("aggressive", "agressivo", "heavy", "pesado", "distortion", "distorção"), (0.95, 0.35)),
    (("epic", "épico", "epico", "cinematic", "cinematográfico", "trailer"),    (0.80, 0.60)),
    (("party", "festa", "club", "dance", "dançante", "dancefloor"),           (0.85, 0.85)),
    (("energetic", "energia", "energético", "upbeat", "animado", "groove"),    (0.80, 0.70)),
    (("happy", "feliz", "alegre", "joyful", "verão", "verao", "sunshine"),     (0.65, 0.90)),
    (("romantic", "romântico", "romantico", "love", "amor", "smooth", "suave"), (0.40, 0.75)),
    (("chill", "relax", "relaxante", "calm", "calma", "calmo", "tranquilo", "lo-fi", "lofi"), (0.30, 0.60)),
    (("melancholic", "melancólico", "melancolico", "nostalgic", "nostálgico", "saudade"),    (0.35, 0.30)),
    (("sad", "triste", "somber", "dark", "sombrio", "escuro", "melancholy"),   (0.40, 0.20)),
    (("ambient", "meditative", "meditativo", "sleep", "drone", "minimal"),    (0.15, 0.50)),
]


def infer_mood_features(mood_text: str, default_energy: float = 0.5,
                        default_valence: float = 0.5) -> Tuple[float, float]:
    """Estima (energia, valência) a partir do texto de humor/prompt via keywords."""
    text = f" {str(mood_text or '').lower()} "
    energies: List[float] = []
    valences: List[float] = []
    for keywords, (energy, valence) in _MOOD_RULES:
        if any(k in text for k in keywords):
            energies.append(energy)
            valences.append(valence)
    if not energies:
        return float(default_energy), float(default_valence)
    return float(np.mean(energies)), float(np.mean(valences))


# =============================================================================
# CONSTRUÇÃO DE FEATURES
# =============================================================================

def _clip01(value: Any, default: float = 0.5) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(v):
        return default
    return min(max(v, 0.0), 1.0)


def _stable_jitter(text: Any, spread: float = 0.05) -> float:
    """Variação determinística em [-spread, +spread] derivada de um texto."""
    if not text:
        return 0.0
    digest = zlib.crc32(str(text).encode("utf-8"))
    return ((digest % 2001) - 1000) / 1000.0 * spread


def song_to_features(bpm: float, genre: Any, duration: float,
                     energy: float, complexity: float, valence: float) -> List[float]:
    """ÚNICO lugar onde o layout do vetor de features é definido.

    Dados reais E sintéticos passam por aqui — é estruturalmente impossível
    as dimensões divergirem (bug do np.vstack antigo).
    """
    try:
        bpm = float(bpm)
    except (TypeError, ValueError):
        bpm = 120.0
    bpm = min(max(bpm, 30.0), 200.0)          # garante feature em [0.15, 1.0]

    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 180.0
    duration = min(max(duration, 0.0), 300.0)  # garante feature em [0, 1.0]

    one_hot = [0.0] * len(GENRE_SCHEMA)
    one_hot[GENRE_TO_IDX[normalize_genre(genre)]] = 1.0
    return [
        bpm / 200.0,
        *one_hot,
        duration / 300.0,
        _clip01(energy),
        _clip01(complexity),
        _clip01(valence),
    ]


# Sanidade: o vetor produzido precisa bater com FEATURE_NAMES.
assert len(song_to_features(120.0, "generic", 180.0, 0.5, 0.5, 0.5)) == FEATURE_DIM, (
    "song_to_features() divergiu de FEATURE_NAMES — ajuste um dos dois."
)


def _first(song: Dict[str, Any], keys: Tuple[str, ...], default: Any) -> Any:
    """Retorna o primeiro campo não-vazio entre os nomes alternativos."""
    for key in keys:
        value = song.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def _extract_bpm(song: Dict[str, Any]) -> float:
    """Extrai BPM aceitando 'bpm', 'tempo' ou 'bpm_range' ([min, max])."""
    for key in ("bpm", "tempo"):
        value = song.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
    bpm_range = song.get("bpm_range") or song.get("tempo_range")
    if isinstance(bpm_range, (list, tuple)):
        for value in bpm_range:
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                return float(value)
    return 120.0


def extract_features_from_metadata(songs: List[Dict[str, Any]]) -> np.ndarray:
    """Converte metadados do dataset em features no schema fixo.

    Tolerante a nomes de campo alternativos e a valores ausentes —
    dados reais vêm sujos do Hugging Face.
    """
    if not songs:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    rows: List[List[float]] = []
    for song in songs:
        genre = normalize_genre(
            _first(song, ("genre", "genres", "style", "tag", "tags"), "generic")
        )
        profile = GENRE_PROFILE.get(genre, GENRE_PROFILE["generic"])

        bpm = _extract_bpm(song)
        duration = _first(
            song, ("duration", "length", "duration_seconds", "duration_secs", "seconds"), 180.0
        )

        mood_text = " ".join(
            str(part) for part in (
                _first(song, ("mood", "humor", "vibe", "moods", "tags"), ""),
                _first(song, ("prompt", "description", "style_prompt"), ""),
            ) if part
        )
        default_energy = sum(profile["energy"]) / 2.0
        default_valence = sum(profile["valence"]) / 2.0
        energy, valence = infer_mood_features(mood_text, default_energy, default_valence)

        title = _first(song, ("title", "name", "id", "prompt"), "")
        complexity = _clip01(GENRE_COMPLEXITY_PRIOR.get(genre, 0.5) + _stable_jitter(title))

        rows.append(song_to_features(bpm, genre, duration, energy, complexity, valence))

    return np.asarray(rows, dtype=np.float32)


def genre_indices_from_features(X: np.ndarray) -> np.ndarray:
    """Índice do gênero (argmax do one-hot) de cada linha de features."""
    if X.size == 0:
        return np.zeros(0, dtype=np.int64)
    start, end = GENRE_SLICE
    return X[:, start:end].argmax(axis=1).astype(np.int64)


def generate_synthetic_data(num_samples: int = 200,
                             genres: Optional[List[str]] = None,
                             seed: Optional[int] = None) -> np.ndarray:
    """Gera amostras sintéticas SEMPRE no layout do schema fixo.

    `genres` restringe o sorteio (modo 'mixed' passa os gêneros presentes nos
    dados reais, para acompanhar a distribuição), mas o one-hot tem sempre
    len(GENRE_SCHEMA) posições — não existe mais caminho pelo qual as
    dimensões divergem.
    """
    rng = np.random.default_rng(seed)
    pool = {normalize_genre(g) for g in (genres if genres is not None else GENRE_SCHEMA)}
    pool = sorted(g for g in pool if g in GENRE_TO_IDX)
    if not pool:
        pool = list(GENRE_SCHEMA)

    rows: List[List[float]] = []
    for _ in range(num_samples):
        genre = pool[int(rng.integers(len(pool)))]
        profile = GENRE_PROFILE.get(genre, GENRE_PROFILE["generic"])
        rows.append(song_to_features(
            bpm=int(rng.integers(profile["bpm"][0], profile["bpm"][1] + 1)),
            genre=genre,
            duration=float(rng.uniform(45.0, 300.0)),
            energy=float(rng.uniform(*profile["energy"])),
            complexity=float(rng.uniform(*profile["complexity"])),
            valence=float(rng.uniform(*profile["valence"])),
        ))
    return np.asarray(rows, dtype=np.float32)


# =============================================================================
# CARREGAMENTO E DIVISÃO DOS DADOS
# =============================================================================

def load_suno_metadata(path: Path, max_songs: int, seed: int,
                       log: logging.Logger) -> List[Dict[str, Any]]:
    """Lê o metadata.json salvo por download_dataset.py.

    Aceita lista pura ou dicionário com 'songs'/'data'/'items'.
    A subamostragem (max_songs) é determinística (seed).
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.error("Não foi possível ler %s: %s", path, exc)
        return []

    if isinstance(raw, dict):
        songs = raw.get("songs") or raw.get("data") or raw.get("items") or []
    elif isinstance(raw, list):
        songs = raw
    else:
        log.warning("Formato inesperado em %s — ignorando dataset.", path)
        return []

    songs = [s for s in songs if isinstance(s, dict)]
    log.info("Dataset carregado: %d músicas em %s", len(songs), path)

    if max_songs and 0 < max_songs < len(songs):
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.permutation(len(songs))[:max_songs].tolist())
        songs = [songs[i] for i in chosen]
        log.info("Subamostragem determinística: usando %d de %d músicas (seed=%d).",
                 len(songs), len(songs) and max_songs, seed)
    return songs


def stratified_split(genre_indices: np.ndarray, val_split: float,
                     seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Divisão treino/validação estratificada por gênero.

    Garante que gêneros presentes nos dados também apareçam na validação
    (quando há amostras suficientes), evitando vieses com datasets pequenos.
    """
    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []

    for g in np.unique(genre_indices):
        idx = np.where(genre_indices == g)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_split))
        if len(idx) >= 2 and n_val >= len(idx):
            n_val = len(idx) - 1  # nunca esvazia o treino de um gênero
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])

    train_idx = (np.concatenate(train_parts) if train_parts
                 else np.array([], dtype=np.int64))
    val_idx = (np.concatenate(val_parts) if val_parts
               else np.array([], dtype=np.int64))

    if len(val_idx) == 0 and len(genre_indices) >= 4:
        perm = rng.permutation(len(genre_indices))
        n_val = max(1, int(len(genre_indices) * val_split))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _stack(parts: List[np.ndarray]) -> np.ndarray:
    parts = [p for p in parts if len(p)]
    return np.vstack(parts) if parts else np.zeros((0, FEATURE_DIM), dtype=np.float32)


# =============================================================================
# MODELOS
# =============================================================================

class NumPyAutoencoder:
    """Autoencoder em NumPy puro — fallback quando PyTorch não está disponível.

    Arquitetura: input -> hidden(ReLU) -> latent -> hidden(ReLU) -> input(Sigmoid)
    Treinamento por gradiente descendente em mini-batches com MSE.
    """

    ACTIVATIONS: Tuple[str, ...] = ("relu", "linear", "relu", "sigmoid")

    def __init__(self, input_dim: int, latent_dim: int = 8,
                 hidden_dim: int = 48, seed: int = 42):
        rng = np.random.default_rng(seed)
        sizes = [input_dim, hidden_dim, latent_dim, hidden_dim, input_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # Inicialização He (adequada para ReLU)
            self.weights.append(
                rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)).astype(np.float32)
            )
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    # -- forward ------------------------------------------------------------
    def _forward(self, X: np.ndarray) -> List[np.ndarray]:
        activations = [X]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ W + b
            act = self.ACTIVATIONS[i]
            if act == "relu":
                a = np.maximum(z, 0.0)
            elif act == "sigmoid":
                z = np.clip(z, -30.0, 30.0)  # evita overflow em exp()
                a = 1.0 / (1.0 + np.exp(-z))
            else:  # linear (camada latente)
                a = z
            activations.append(a.astype(np.float32))
        return activations

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(X, dtype=np.float32))[-1]

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Espaço latente (útil para interpolação de estilos no gerador)."""
        return self._forward(np.asarray(X, dtype=np.float32))[2]

    # -- treinamento ---------------------------------------------------------
    def _train_batch(self, batch: np.ndarray, lr: float) -> float:
        acts = self._forward(batch)
        out = acts[-1]
        # Gradiente do MSE (média sobre o batch)
        delta = (2.0 / len(batch)) * (out - batch)
        for i in range(len(self.weights) - 1, -1, -1):
            a = acts[i + 1]
            act = self.ACTIVATIONS[i]
            if act == "relu":
                delta = delta * (a > 0)
            elif act == "sigmoid":
                delta = delta * a * (1.0 - a)
            # linear: nada a fazer
            grad_w = acts[i].T @ delta
            grad_b = delta.sum(axis=0)
            delta = delta @ self.weights[i].T  # propaga com os pesos ANTIGOS
            self.weights[i] -= lr * grad_w
            self.biases[i] -= lr * grad_b
        return float(np.mean((out - batch) ** 2))

    def fit(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None, *,
            epochs: int = 50, batch_size: int = 8, lr: float = 1e-2,
            patience: int = 5, seed: int = 42,
            log: Optional[logging.Logger] = None) -> Dict[str, Any]:
        log = log or LOG
        rng = np.random.default_rng(seed)
        X_train = np.asarray(X_train, dtype=np.float32)
        n = len(X_train)
        has_val = X_val is not None and len(X_val) > 0

        best_val = float("inf")
        best_weights = [w.copy() for w in self.weights]
        best_biases = [b.copy() for b in self.biases]
        history: List[Dict[str, float]] = []
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                batch = X_train[perm[start:start + batch_size]]
                epoch_loss += self._train_batch(batch, lr) * len(batch)
            train_loss = epoch_loss / n

            if has_val:
                val_loss = float(np.mean((self.reconstruct(X_val) - X_val) ** 2))
            else:
                val_loss = train_loss

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                log.info("Época %3d/%d — treino: %.5f | validação: %.5f",
                         epoch, epochs, train_loss, val_loss)

            if has_val:
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        log.info("Early stopping na época %d (sem melhora por %d épocas).",
                                 epoch, patience)
                        break

        if has_val:  # sem validação, mantém os pesos finais
            self.weights, self.biases = best_weights, best_biases
        return {"backend": "numpy", "history": history,
                "best_val_loss": best_val if has_val else None}


if HAS_TORCH:

    class TorchAutoencoder(nn.Module):
        """Autoencoder denso: input -> 48 (ReLU) -> latent -> 48 (ReLU) -> input (Sigmoid)."""

        def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 48):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid(),  # todas as features estão normalizadas em [0, 1]
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    def _train_torch(X_train: np.ndarray, X_val: Optional[np.ndarray], *,
                     latent_dim: int, epochs: int, batch_size: int, lr: float,
                     patience: int, seed: int,
                     log: logging.Logger) -> Tuple[TorchAutoencoder, Dict[str, Any]]:
        torch.manual_seed(seed)
        model = TorchAutoencoder(input_dim=X_train.shape[1], latent_dim=latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        Xtr = torch.from_numpy(np.ascontiguousarray(X_train)).float()
        Xva = (torch.from_numpy(np.ascontiguousarray(X_val)).float()
               if X_val is not None and len(X_val) else None)
        n = Xtr.shape[0]
        has_val = Xva is not None

        best_val = float("inf")
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        history: List[Dict[str, float]] = []
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                batch = Xtr[perm[start:start + batch_size]]
                optimizer.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * batch.shape[0]
            train_loss = epoch_loss / n

            if has_val:
                model.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(model(Xva), Xva).item())
            else:
                val_loss = train_loss

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                log.info("Época %3d/%d — treino: %.5f | validação: %.5f",
                         epoch, epochs, train_loss, val_loss)

            if has_val:
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        log.info("Early stopping na época %d (sem melhora por %d épocas).",
                                 epoch, patience)
                        break

        if has_val:  # sem validação, mantém os pesos finais
            model.load_state_dict(best_state)
        model.eval()
        return model, {"backend": "torch", "history": history,
                       "best_val_loss": best_val if has_val else None}


def train_autoencoder(X_train: np.ndarray, X_val: Optional[np.ndarray], *,
                      latent_dim: int, epochs: int, batch_size: int, lr: float,
                      patience: int, seed: int,
                      log: logging.Logger) -> Tuple[Any, Dict[str, Any]]:
    """Treina com PyTorch (se disponível) ou com o fallback em NumPy puro."""
    if HAS_TORCH:
        return _train_torch(X_train, X_val, latent_dim=latent_dim, epochs=epochs,
                            batch_size=batch_size, lr=lr, patience=patience,
                            seed=seed, log=log)
    model = NumPyAutoencoder(input_dim=X_train.shape[1], latent_dim=latent_dim, seed=seed)
    info = model.fit(X_train, X_val, epochs=epochs, batch_size=batch_size,
                     lr=lr, patience=patience, seed=seed, log=log)
    return model, info


# =============================================================================
# AVALIAÇÃO E PERSISTÊNCIA
# =============================================================================

def reconstruct(model: Any, X: np.ndarray, backend: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if backend == "torch":
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(X)).numpy()
    return model.reconstruct(X)


def encode(model: Any, X: np.ndarray, backend: str) -> np.ndarray:
    """Retorna o espaço latente (p/ interpolação de estilos no gerador)."""
    X = np.asarray(X, dtype=np.float32)
    if backend == "torch":
        model.eval()
        with torch.no_grad():
            return model.encoder(torch.from_numpy(X)).numpy()
    return model.encode(X)


def evaluate(model: Any, X: np.ndarray, backend: str) -> Dict[str, float]:
    """Métricas interpretáveis: MSE, acurácia de gênero e erro de BPM."""
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        return {"mse": float("nan"), "genre_accuracy": float("nan"),
                "bpm_mae": float("nan")}
    recon = reconstruct(model, X, backend)
    start, end = GENRE_SLICE
    return {
        "mse": float(np.mean((recon - X) ** 2)),
        "genre_accuracy": float(np.mean(
            X[:, start:end].argmax(axis=1) == recon[:, start:end].argmax(axis=1)
        )),
        "bpm_mae": float(np.mean(np.abs(recon[:, 0] - X[:, 0])) * 200.0),
    }


def _save_model(model: Any, backend: str, latent_dim: int,
                metrics: Dict[str, float], models_dir: Path,
                log: logging.Logger) -> Dict[str, Any]:
    models_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "backend": backend,
        "feature_dim": FEATURE_DIM,
        "genre_schema": GENRE_SCHEMA,
        "feature_names": FEATURE_NAMES,
        "latent_dim": latent_dim,
        "val_mse": metrics.get("mse"),
        "val_genre_accuracy": metrics.get("genre_accuracy"),
        "val_bpm_mae": metrics.get("bpm_mae"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if backend == "torch":
        weights_path = models_dir / "autoencoder.pt"
        torch.save(model.state_dict(), weights_path)
    else:
        weights_path = models_dir / "autoencoder_np.npz"
        arrays: Dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(zip(model.weights, model.biases)):
            arrays[f"W{i}"], arrays[f"b{i}"] = w, b
        np.savez_compressed(weights_path, **arrays)

    (models_dir / "autoencoder_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Modelo salvo: %s | metadados: %s",
             weights_path, models_dir / "autoencoder_meta.json")
    return meta


def load_autoencoder(models_dir: Path = MODELS_DIR) -> Tuple[Any, Dict[str, Any], str]:
    """Carrega o modelo + metadados, VALIDANDO o schema (fonte única de verdade).

    Use esta função em music_generator.py / music_intelligence.py em vez de
    abrir os arquivos diretamente — ela recusa modelos treinados com schema
    antigo, evitando erros silenciosos.
    """
    meta_path = models_dir / "autoencoder_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Nenhum modelo treinado em {models_dir}. Rode 'python train.py' primeiro."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if (meta.get("feature_dim") != FEATURE_DIM
            or list(meta.get("genre_schema", [])) != list(GENRE_SCHEMA)):
        raise ValueError(
            "Modelo em cache incompatível com o schema atual "
            f"(treinado com feature_dim={meta.get('feature_dim')}, "
            f"gêneros={meta.get('genre_schema')}). "
            "Retreine com 'python train.py --dataset mixed'."
        )
    latent_dim = int(meta.get("latent_dim", 8))

    if HAS_TORCH and (models_dir / "autoencoder.pt").exists():
        model = TorchAutoencoder(FEATURE_DIM, latent_dim=latent_dim)
        model.load_state_dict(torch.load(models_dir / "autoencoder.pt",
                                         map_location="cpu"))
        model.eval()
        return model, meta, "torch"

    npz_path = models_dir / "autoencoder_np.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            n_layers = sum(1 for key in data.files if key.startswith("W"))
            model = NumPyAutoencoder(FEATURE_DIM, latent_dim=latent_dim)
            model.weights = [data[f"W{i}"].astype(np.float32) for i in range(n_layers)]
            model.biases = [data[f"b{i}"].astype(np.float32) for i in range(n_layers)]
        return model, meta, "numpy"

    raise FileNotFoundError(
        f"Metadados encontrados em {meta_path}, mas sem pesos utilizáveis "
        "(modelos .pt exigem PyTorch instalado). Retreine com 'python train.py'."
    )


def _save_training_report(args: argparse.Namespace, *, info: Dict[str, Any],
                          metrics: Dict[str, float], n_real: int, n_synth: int,
                          n_train: int, n_val: int, log: logging.Logger) -> None:
    args.memory_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": info["backend"],
        "dataset_mode": args.dataset,
        "num_real_songs": n_real,
        "num_synthetic": n_synth,
        "train_samples": n_train,
        "val_samples": n_val,
        "feature_dim": FEATURE_DIM,
        "genre_schema": GENRE_SCHEMA,
        "epochs_requested": args.epochs,
        "epochs_run": len(info["history"]),
        "best_val_loss": info["best_val_loss"],
        "val_mse": metrics.get("mse"),
        "val_genre_accuracy": metrics.get("genre_accuracy"),
        "val_bpm_mae": metrics.get("bpm_mae"),
    }
    (args.memory_dir / "training_report.json").write_text(
        json.dumps({**summary, "history": info["history"]},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Histórico acumulado — alimenta o feedback loop (music_intelligence.py)
    history_path = args.memory_dir / "training_history.json"
    try:
        entries = (json.loads(history_path.read_text(encoding="utf-8"))
                   if history_path.exists() else [])
        if not isinstance(entries, list):
            entries = []
        entries.append(summary)
        history_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Não foi possível atualizar o histórico de treinos: %s", exc)


# =============================================================================
# TREINAMENTO (ENTRY POINT)
# =============================================================================

def train(args: argparse.Namespace) -> int:
    log = _setup_logging(args.verbose)
    log.info("=" * 64)
    log.info("IA Music Pro — Treinamento do autoencoder de estilo")
    log.info("Backend: %s | modo: %s | seed: %d",
             "PyTorch" if HAS_TORCH else "NumPy puro (fallback)",
             args.dataset, args.seed)
    log.info("Schema fixo: %d features (%d gêneros no one-hot)",
             FEATURE_DIM, len(GENRE_SCHEMA))
    log.info("=" * 64)

    if args.epochs < 1 or args.batch_size < 1:
        log.error("--epochs e --batch-size devem ser >= 1.")
        return 1

    np.random.seed(args.seed)
    if HAS_TORCH:
        torch.manual_seed(args.seed)

    # -- 1. Dados reais ------------------------------------------------------
    real_songs: List[Dict[str, Any]] = []
    if args.dataset in ("suno", "mixed"):
        if args.dataset_path.exists():
            real_songs = load_suno_metadata(args.dataset_path, args.max_songs,
                                            args.seed, log)
        else:
            log.warning("Dataset não encontrado em %s — rode "
                        "'python download_dataset.py' antes.", args.dataset_path)
        if not real_songs and args.dataset == "suno":
            log.error("O modo 'suno' exige o dataset real. Abortando.")
            return 1

    real_data = extract_features_from_metadata(real_songs)
    if len(real_data):
        log.info("Features reais: shape %s (%d músicas carregadas)",
                 real_data.shape, len(real_songs))
    elif real_songs:
        log.warning("Músicas carregadas, mas nenhuma feature extraída — "
                    "verifique o formato de %s", args.dataset_path)
    else:
        log.warning("Nenhuma música real disponível.")

    # -- 2. Dados sintéticos ---------------------------------------------------
    synthetic_data = np.zeros((0, FEATURE_DIM), dtype=np.float32)
    if args.dataset == "synthetic":
        log.info("Modo sintético puro: gerando %d amostras.", args.synthetic_samples)
        synthetic_data = generate_synthetic_data(
            num_samples=args.synthetic_samples, seed=args.seed)
    elif args.dataset == "mixed" and len(real_data) < MIN_TRAIN_SAMPLES:
        num_synthetic = MIN_TRAIN_SAMPLES - len(real_data)
        log.warning("Gerando %d amostras sintéticas (complemento)", num_synthetic)
        real_genres = sorted({
            normalize_genre(_first(s, ("genre", "genres", "style", "tag", "tags"), "generic"))
            for s in real_songs
        }) or None
        synthetic_data = generate_synthetic_data(
            num_samples=num_synthetic, genres=real_genres, seed=args.seed)
        log.info("Adicionando %d amostras sintéticas", len(synthetic_data))
    elif args.dataset == "mixed":
        log.info("Reais suficientes (%d >= %d): sem complemento sintético.",
                 len(real_data), MIN_TRAIN_SAMPLES)

    # -- 3. Validação defensiva de dimensões -----------------------------------
    for name, arr in (("real", real_data), ("sintético", synthetic_data)):
        if arr.size and (arr.ndim != 2 or arr.shape[1] != FEATURE_DIM):
            raise ValueError(
                f"Dados {name}: shape {arr.shape} — esperado (n, {FEATURE_DIM}). "
                "Alguém alterou GENRE_SCHEMA ou adicionou feature em um único extrator?"
            )
    if len(real_data) == 0 and len(synthetic_data) == 0:
        log.error("Sem dados para treinar (reais e sintéticos vazios).")
        return 1

    # -- 4. Split treino/validação ANTES de misturar ---------------------------
    empty = np.zeros((0, FEATURE_DIM), dtype=np.float32)
    if len(real_data) >= 4:
        tr, va = stratified_split(genre_indices_from_features(real_data),
                                 args.val_split, args.seed)
        real_train, real_val = real_data[tr], real_data[va]
        # mixed: sintéticos vão SOMENTE para o treino — não contaminam a validação
        synth_train, synth_val = synthetic_data, empty
    elif len(synthetic_data) >= 4:
        log.info("Sem dados reais suficientes: validação sai das amostras sintéticas.")
        tr, va = stratified_split(genre_indices_from_features(synthetic_data),
                                 args.val_split, args.seed)
        synth_train, synth_val = synthetic_data[tr], synthetic_data[va]
        real_train, real_val = real_data, empty
    else:
        real_train, real_val = real_data, empty
        synth_train, synth_val = synthetic_data, empty

    X_train = _stack([real_train, synth_train])
    X_val = _stack([real_val, synth_val])

    rng = np.random.default_rng(args.seed)
    X_train = X_train[rng.permutation(len(X_train))]

    if len(X_val) == 0:
        log.warning("Validação vazia — early stopping desativado; "
                    "métricas finais serão otimistas.")
        X_val_fit: Optional[np.ndarray] = None
    else:
        X_val_fit = X_val

    log.info("Amostras — treino: %d | validação: %d (reais: %d | sintéticas: %d)",
             len(X_train), len(X_val), len(real_data), len(synthetic_data))

    # -- 5. Treinamento ---------------------------------------------------------
    # GD puro (NumPy) precisa de lr maior que o Adam (PyTorch).
    lr = args.lr if HAS_TORCH else max(args.lr, 1e-2)
    model, info = train_autoencoder(
        X_train, X_val_fit,
        latent_dim=args.latent_dim, epochs=args.epochs, batch_size=args.batch_size,
        lr=lr, patience=args.patience, seed=args.seed, log=log,
    )

    # -- 6. Avaliação ------------------------------------------------------------
    X_eval = X_val_fit if X_val_fit is not None else X_train
    if X_val_fit is None:
        log.warning("Avaliando no próprio treino (sem conjunto de validação).")
    metrics = evaluate(model, X_eval, info["backend"])
    log.info("Avaliação — MSE: %.5f | acurácia de gênero: %.1f%% | erro de BPM: ±%.1f",
             metrics["mse"], 100.0 * metrics["genre_accuracy"], metrics["bpm_mae"])

    # -- 7. Persistência + verificação de integridade -----------------------------
    _save_model(model, info["backend"], args.latent_dim, metrics,
                args.models_dir, log)
    try:
        reloaded, _, backend = load_autoencoder(args.models_dir)
        check = evaluate(reloaded, X_eval, backend)
        if np.isclose(check["mse"], metrics["mse"], rtol=1e-6, equal_nan=True):
            log.info("Verificação pós-save OK: modelo recarregado reproduz "
                     "o MSE (%.5f).", check["mse"])
        else:
            log.warning("Verificação pós-save divergiu: %.6f vs %.6f — "
                        "o modelo salvo pode estar corrompido.",
                        check["mse"], metrics["mse"])
    except (OSError, ValueError) as exc:
        log.error("Falha ao recarregar o modelo salvo: %s", exc)
        return 1

    # -- 8. Relatório (alimenta o feedback loop) ----------------------------------
    _save_training_report(args, info=info, metrics=metrics, n_real=len(real_data),
                          n_synth=len(synthetic_data), n_train=len(X_train),
                          n_val=len(X_val), log=log)

    log.info("Treinamento concluído com sucesso.")
    return 0


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina o autoencoder de estilo do IA Music Pro.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=("suno", "synthetic", "mixed"),
                        default="mixed",
                        help="suno: só dataset real | synthetic: só amostras "
                             "sintéticas | mixed: real + complemento sintético")
    parser.add_argument("--max-songs", type=int, default=100,
                        help="Máximo de músicas reais usadas (0 = todas)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Taxa de aprendizado (Adam no torch; GD no fallback)")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5,
                        help="Épocas sem melhora antes do early stopping")
    parser.add_argument("--synthetic-samples", type=int, default=200,
                        help="Amostras no modo 'synthetic'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--memory-dir", type=Path, default=MEMORY_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    sys.exit(train(args))


if __name__ == "__main__":
    main()
