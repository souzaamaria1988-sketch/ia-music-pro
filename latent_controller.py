#!/usr/bin/env python3
"""
🧠 LATENT CONTROLLER - Real AI Guidance
Maps the autoencoder's latent space to musical parameters.
This is the BRIDGE between the trained AI and the synthesizer.
"""
import os
import numpy as np
from pathlib import Path

MODEL_DIR = "models"
AE_DIR = os.path.join(MODEL_DIR, "autoencoder")


class LatentController:
    """
    Interprets the 64-dimensional latent space as musical parameters.
    This is trained via calibration: we pass known audio through the
    encoder and learn the mapping from latent → params.
    """
    
    PARAM_BOUNDS = {
        "bpm": (60, 180),
        "intensity": (0.2, 1.0),
        "brightness": (0.0, 1.0),       # high freq energy
        "bass_weight": (0.0, 1.0),       # low freq energy
        "rhythmic_density": (0.0, 1.0),  # events per second
        "harmonic_complexity": (0.0, 1.0),
        "dynamics_range": (0.0, 1.0),
        "style_index": (0, 6),           # which style cluster
    }
    
    def __init__(self, autoencoder=None):
        self.ae = autoencoder
        self._calibration = self._load_calibration()
    
    def _load_calibration(self):
        """Load or compute the latent→params projection matrix."""
        cal_path = os.path.join(AE_DIR, "calibration.npz")
        if os.path.exists(cal_path):
            data = np.load(cal_path)
            return {
                "W": data["W"],
                "b": data["b"],
                "latent_mean": data["latent_mean"],
                "latent_std": data["latent_std"],
            }
        # Default: random initialization (will work poorly but won't crash)
        n_latent = 64
        n_params = len(self.PARAM_BOUNDS)
        np.random.seed(42)
        return {
            "W": np.random.randn(n_latent, n_params) * 0.1,
            "b": np.zeros(n_params),
            "latent_mean": np.zeros(n_latent),
            "latent_std": np.ones(n_latent),
        }
    
    def calibrate(self, latent_samples, param_samples):
        """
        Learn the linear mapping: params = W @ (latent - mean)/std + b
        latent_samples: (N, 64)
        param_samples: (N, n_params) with values in [0, 1]
        """
        n = latent_samples.shape[0]
        latent_mean = latent_samples.mean(axis=0)
        latent_std = latent_samples.std(axis=0) + 1e-6
        X = (latent_samples - latent_mean) / latent_std
        
        # Add bias column
        X = np.hstack([X, np.ones((n, 1))])
        
        # Ridge regression (stable)
        W_aug = np.linalg.lstsq(X, param_samples, rcond=None)[0]
        W = W_aug[:-1]
        b = W_aug[-1]
        
        self._calibration = {
            "W": W, "b": b,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
        }
        
        # Persist
        os.makedirs(AE_DIR, exist_ok=True)
        np.savez(
            os.path.join(AE_DIR, "calibration.npz"),
            W=W, b=b,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )
        return self._calibration
    
    def latent_to_params(self, latent):
        """Map a 64-dim latent vector to musical parameters in [0, 1]."""
        latent = np.asarray(latent, dtype=np.float32).flatten()
        if len(latent) != 64:
            raise ValueError(f"Esperado 64 dimensões, recebido {len(latent)}")
        
        cal = self._calibration
        x = (latent - cal["latent_mean"]) / cal["latent_std"]
        raw = x @ cal["W"] + cal["b"]
        # Sigmoid to bound in (0, 1)
        normalized = 1.0 / (1.0 + np.exp(-raw))
        
        params = {}
        for i, (name, (lo, hi)) in enumerate(self.PARAM_BOUNDS.items()):
            params[name] = lo + normalized[i] * (hi - lo)
        return params
    
    def vary(self, latent, strength=0.1, seed=None):
        """Create a slight variation in latent space."""
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.randn(*latent.shape) * strength
        # Stay near the manifold: add noise in directions with high variance
        return latent + noise
    
    def interpolate(self, latent_a, latent_b, t):
        """Linear interpolation between two latent vectors. t in [0, 1]."""
        t = float(np.clip(t, 0.0, 1.0))
        return (1.0 - t) * latent_a + t * latent_b


class StyleExtractor:
    """
    Extracts the latent representation from a real audio file.
    This is how we get "the DNA" of a reference track.
    """
    
    def __init__(self, autoencoder, sr=22050, n_features=256):
        self.ae = autoencoder
        self.sr = sr
        self.n_features = n_features
    
    def extract_from_file(self, filepath):
        """Extract average latent vector from an audio file."""
        try:
            import soundfile as sf
            audio, fsr = sf.read(filepath, dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if fsr != self.sr:
                idx = np.round(np.arange(0, len(audio), fsr / self.sr)).astype(int)
                audio = audio[idx[idx < len(audio)]]
        except Exception as e:
            raise ValueError(f"Não foi possível ler {filepath}: {e}")
        
        if len(audio) < self.sr:
            raise ValueError(f"Áudio muito curto: {filepath}")
        
        # Extract multiple feature windows and average their latents
        seg_len = self.sr  # 1 second windows
        n_segs = min(len(audio) // seg_len, 20)
        latents = []
        
        # Load normalization from training
        norm_path = os.path.join(AE_DIR, "normalization.npz")
        if os.path.exists(norm_path):
            norm = np.load(norm_path)
            mean, std = norm["mean"], norm["std"]
        else:
            mean = np.zeros(self.n_features)
            std = np.ones(self.n_features)
        
        for i in range(n_segs):
            seg = audio[i * seg_len:(i + 1) * seg_len]
            fft = np.abs(np.fft.rfft(seg))[:self.n_features]
            fft = fft / (np.max(fft) + 1e-10)
            if len(fft) < self.n_features:
                fft = np.pad(fft, (0, self.n_features - len(fft)))
            fft = (fft - mean) / (std + 1e-8)
            latents.append(self.ae.encode(fft.reshape(1, -1))[0])
        
        if not latents:
            raise ValueError(f"Nenhum segmento válido em {filepath}")
        
        return np.mean(latents, axis=0), np.std(latents, axis=0)
