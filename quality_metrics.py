"""Métricas de qualidade para avaliação de áudio gerado."""
from __future__ import annotations
import numpy as np
import logging

log = logging.getLogger(__name__)


def calculate_rms(audio):
    if len(audio) == 0: return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def calculate_peak(audio):
    if len(audio) == 0: return 0.0
    return float(np.max(np.abs(audio)))


def detect_clipping(audio, threshold=0.99):
    if len(audio) == 0: return 0, 0.0
    clipped = np.sum(np.abs(audio) >= threshold)
    return int(clipped), float(clipped / len(audio))


def calculate_dynamic_range(audio, window_size=1024):
    if len(audio) < window_size: return 0.0
    windows = len(audio) // window_size
    if windows == 0: return 0.0
    rms_values = [calculate_rms(audio[i*window_size:(i+1)*window_size]) for i in range(windows)]
    rms_values = np.array(rms_values)
    rms_values = rms_values[rms_values > 0]
    if len(rms_values) < 2: return 0.0
    max_rms, min_rms = np.max(rms_values), np.min(rms_values)
    if min_rms == 0: return 0.0
    return float(20 * np.log10(max_rms / min_rms))


def calculate_spectral_centroid(audio, sample_rate):
    if len(audio) == 0: return 0.0
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    if np.sum(fft) == 0: return 0.0
    return float(np.sum(freqs * fft) / np.sum(fft))


def analyze_audio_quality(audio, sample_rate):
    issues = []
    score = 1.0
    rms = calculate_rms(audio)
    peak = calculate_peak(audio)
    clipped_count, clipped_ratio = detect_clipping(audio)
    dynamic_range = calculate_dynamic_range(audio)
    spectral_centroid = calculate_spectral_centroid(audio, sample_rate)

    if peak >= 0.99: issues.append("clipping detectado"); score -= 0.3
    if rms < 0.01: issues.append("volume muito baixo"); score -= 0.2
    if rms > 0.5: issues.append("volume muito alto"); score -= 0.1
    if dynamic_range < 6.0: issues.append("faixa dinâmica baixa"); score -= 0.15
    silence_ratio = np.mean(np.abs(audio) < 0.001)
    if silence_ratio > 0.8: issues.append("silêncio excessivo"); score -= 0.3

    return {
        "score": max(0.0, score),
        "issues": issues,
        "metrics": {
            "rms": rms, "peak": peak,
            "clipped_samples": clipped_count, "clipping_ratio": clipped_ratio,
            "dynamic_range_db": dynamic_range,
            "spectral_centroid_hz": spectral_centroid,
            "silence_ratio": float(silence_ratio),
        },
    }
