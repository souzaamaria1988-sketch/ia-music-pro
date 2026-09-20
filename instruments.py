#!/usr/bin/env python3
"""
🎸 INSTRUMENTS LIBRARY - 100+ Instrumentos com Síntese Procedural
Organizados por famílias com parâmetros únicos de síntese
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from scipy.signal import lfilter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# SÍNTESE BASE (métodos fundamentais)
# ============================================================

def karplus_strong(freq, duration, sr=44100, damping=0.996, brightness=0.5):
    """Síntese de cordas dedilhadas (Karplus-Strong)"""
    N = max(2, int(sr / freq))
    n_samples = int(duration * sr)
    x = np.zeros(n_samples)
    x[:min(N, n_samples)] = np.random.uniform(-1, 1, min(N, n_samples))
    if HAS_SCIPY:
        a = np.zeros(N + 2)
        a[0] = 1.0
        a[N] = -damping * brightness
        a[N + 1] = -damping * (1.0 - brightness)
        y = lfilter([1.0], a, x)
    else:
        y = np.zeros(n_samples)
        delay = np.random.uniform(-1, 1, N)
        for i in range(n_samples):
            y[i] = delay[i % N]
            delay[i % N] = damping * 0.5 * (delay[i % N] + delay[(i + 1) % N])
    return y / (np.max(np.abs(y)) + 1e-10)


def additive_synthesis(freq, duration, sr=44100, harmonics=None, amplitudes=None, decays=None):
    """Síntese aditiva com harmônicos personalizados"""
    if harmonics is None:
        harmonics = [1, 2, 3, 4, 5, 6, 7, 8]
    if amplitudes is None:
        amplitudes = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05]
    if decays is None:
        decays = [2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8]
    
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.zeros_like(t)
    
    for h, amp, dec in zip(harmonics, amplitudes, decays):
        signal += amp * np.sin(2 * np.pi * freq * h * t) * np.exp(-t * dec)
    
    return signal / (np.max(np.abs(signal)) + 1e-10)


def fm_synthesis(freq, duration, sr=44100, mod_ratio=2.0, mod_index=3.0):
    """Síntese FM (frequência modulada)"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    mod_freq = freq * mod_ratio
    signal = np.sin(2 * np.pi * freq * t + mod_index * np.sin(2 * np.pi * mod_freq * t))
    return signal / (np.max(np.abs(signal)) + 1e-10)


def subtractive_synthesis(freq, duration, sr=44100, wave_type="saw", filter_cutoff=2000):
    """Síntese subtrativa com filtro"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    
    if wave_type == "saw":
        signal = 2 * (t * freq % 1) - 1
    elif wave_type == "square":
        signal = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave_type == "triangle":
        signal = 2 * np.abs(2 * (t * freq % 1) - 1) - 1
    else:  # sine
        signal = np.sin(2 * np.pi * freq * t)
    
    # Filtro passa-baixa simples
    if HAS_SCIPY and filter_cutoff < sr / 2:
        from scipy.signal import butter, filtfilt
        nyq = sr / 2
        cutoff = min(filter_cutoff / nyq, 0.99)
        b, a = butter(2, cutoff, btype='low')
        signal = filtfilt(b, a, signal)
    
    return signal / (np.max(np.abs(signal)) + 1e-10)


def physical_model(freq, duration, sr=44100, model_type="string"):
    """Modelagem física simplificada"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    
    if model_type == "string":
        # Corda com múltiplos modos
        signal = np.zeros_like(t)
        for mode in range(1, 8):
            mode_freq = freq * mode
            signal += np.sin(2 * np.pi * mode_freq * t) / mode * np.exp(-t * mode * 2)
    elif model_type == "membrane":
        # Membrana (tambor)
        signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 15)
        signal += np.random.randn(len(t)) * 0.1 * np.exp(-t * 20)
    elif model_type == "air_column":
        # Coluna de ar (sopro)
        signal = np.sin(2 * np.pi * freq * t)
        signal += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        signal += np.random.randn(len(t)) * 0.05
    
    return signal / (np.max(np.abs(signal)) + 1e-10)


def apply_envelope(signal, sr, attack=0.01, decay=0.1, sustain=0.7, release=0.2):
    """Aplica envelope ADSR"""
    n = len(signal)
    envelope = np.ones(n)
    
    a_samples = int(attack * sr)
    d_samples = int(decay * sr)
    r_samples = int(release * sr)
    
    if a_samples > 0 and a_samples < n:
        envelope[:a_samples] = np.linspace(0, 1, a_samples)
    
    if d_samples > 0 and a_samples + d_samples < n:
        envelope[a_samples:a_samples+d_samples] = np.linspace(1, sustain, d_samples)
    
    if a_samples + d_samples < n - r_samples:
        envelope[a_samples+d_samples:n-r_samples] = sustain
    
    if r_samples > 0 and r_samples < n:
        envelope[n-r_samples:] = np.linspace(sustain, 0, r_samples)
    
    return signal * envelope


# ============================================================
# FAMÍLIA 1: CORDAS (15 instrumentos)
# ============================================================

def violin(freq, duration, sr=44100, vibrato_rate=5.5, vibrato_depth=0.015):
    """Violino - corda friccionada com vibrato"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    phase = 2 * np.pi * np.cumsum(freq * (1.0 + vibrato)) / sr
    signal = np.zeros_like(t)
    for h in range(1, 12):
        signal += np.sin(h * phase) / (h * 1.2)
    signal = apply_envelope(signal, sr, attack=0.08, decay=0.05, sustain=0.8, release=0.05)
    return signal / (np.max(np.abs(signal)) + 1e-10)


def viola(freq, duration, sr=44100):
    """Viola - similar ao violino, mais grave"""
    return violin(freq, duration, sr, vibrato_rate=5.0, vibrato_depth=0.012)


def cello(freq, duration, sr=44100):
    """Violoncelo - corda grave friccionada"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    vibrato = 0.01 * np.sin(2 * np.pi * 4.5 * t)
    phase = 2 * np.pi * np.cumsum(freq * (1.0 + vibrato)) / sr
    signal = np.zeros_like(t)
    for h in range(1, 10):
        signal += np.sin(h * phase) / (h * 1.5)
    signal = apply_envelope(signal, sr, attack=0.1, decay=0.08, sustain=0.85, release=0.1)
    return signal / (np.max(np.abs(signal)) + 1e-10)


def double_bass(freq, duration, sr=44100):
    """Contrabaixo acústico"""
    return cello(freq / 2, duration, sr)


def acoustic_guitar(freq, duration, sr=44100):
    """Violão - corda dedilhada"""
    return karplus_strong(freq, duration, sr, damping=0.995, brightness=0.6)


def electric_guitar(freq, duration, sr=44100, distortion=0.0):
    """Guitarra elétrica com distorção opcional"""
    signal = karplus_strong(freq, duration, sr, damping=0.997, brightness=0.7)
    if distortion > 0:
        signal = np.tanh(signal * (1 + distortion * 3))
    return signal


def electric_bass(freq, duration, sr=44100):
    """Baixo elétrico"""
    return karplus_strong(freq / 2, duration, sr, damping=0.998, brightness=0.3)


def harp(freq, duration, sr=44100):
    """Harpa - corda dedilhada suave"""
    signal = karplus_strong(freq, duration, sr, damping=0.999, brightness=0.4)
    return apply_envelope(signal, sr, attack=0.005, decay=0.3, sustain=0.3, release=0.5)


def ukulele(freq, duration, sr=44100):
    """Ukulele - corda pequena e brilhante"""
    return karplus_strong(freq * 2, duration, sr, damping=0.994, brightness=0.8)


def banjo(freq, duration, sr=44100):
    """Banjo - corda com ressonância metálica"""
    signal = karplus_strong(freq, duration, sr, damping=0.993, brightness=0.9)
    # Adicionar ressonância metálica
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    metallic = np.sin(2 * np.pi * freq * 3 * t) * 0.2 * np.exp(-t * 10)
    return (signal + metallic) / (np.max(np.abs(signal + metallic)) + 1e-10)


def mandolin(freq, duration, sr=44100):
    """Mandolim - cordas duplas"""
    s1 = karplus_strong(freq, duration, sr, damping=0.994, brightness=0.7)
    s2 = karplus_strong(freq * 1.003, duration, sr, damping=0.994, brightness=0.7)
    return (s1 + s2) / 2


def cavaquinho(freq, duration, sr=44100):
    """Cavaquinho - instrumento brasileiro"""
    return karplus_strong(freq * 2, duration, sr, damping=0.993, brightness=0.85)


def sitar(freq, duration, sr=44100):
    """Sitar - instrumento indiano com simpáticas"""
    signal = karplus_strong(freq, duration, sr, damping=0.996, brightness=0.6)
    # Cordas simpáticas
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    sympathetic = np.sin(2 * np.pi * freq * 2 * t) * 0.15 * np.exp(-t * 3)
    return (signal + sympathetic) / (np.max(np.abs(signal + sympathetic)) + 1e-10)


def koto(freq, duration, sr=44100):
    """Koto - instrumento japonês"""
    return karplus_strong(freq, duration, sr, damping=0.997, brightness=0.5)


def balalaika(freq, duration, sr=44100):
    """Balalaika - instrumento russo"""
    return karplus_strong(freq, duration, sr, damping=0.994, brightness=0.75)


def rabeca(freq, duration, sr=44100):
    """Rabeca - violino popular brasileiro"""
    return violin(freq, duration, sr, vibrato_rate=6.0, vibrato_depth=0.02)


# ============================================================
# FAMÍLIA 2: TECLAS (12 instrumentos)
# ============================================================

def piano(freq, duration, sr=44100, velocity=1.0):
    """Piano acústico"""
    signal = additive_synthesis(freq, duration, sr,
        harmonics=[1,2,3,4,5,6,7,8],
        amplitudes=[1,.6,.35,.25,.18,.14,.11,.09],
        decays=[2.5,2.2,2,1.8,1.6,1.4,1.3,1.2])
    signal = apply_envelope(signal, sr, attack=0.003, decay=0.1, sustain=0.3, release=0.3)
    return signal * velocity


def electric_piano(freq, duration, sr=44100):
    """Piano elétrico (Rhodes)"""
    signal = fm_synthesis(freq, duration, sr, mod_ratio=2.0, mod_index=2.0)
    return apply_envelope(signal, sr, attack=0.005, decay=0.2, sustain=0.5, release=0.3)


def organ(freq, duration, sr=44100):
    """Órgão - múltiplos harmônicos sustentados"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.zeros_like(t)
    for h, amp in [(1, 1.0), (2, 0.8), (3, 0.6), (4, 0.4), (6, 0.3), (8, 0.2)]:
        signal += amp * np.sin(2 * np.pi * freq * h * t)
    return apply_envelope(signal, sr, attack=0.01, decay=0.0, sustain=1.0, release=0.1)


def harpsichord(freq, duration, sr=44100):
    """Cravo - corda beliscada"""
    signal = karplus_strong(freq, duration, sr, damping=0.998, brightness=0.9)
    return apply_envelope(signal, sr, attack=0.001, decay=0.5, sustain=0.2, release=0.1)


def accordion(freq, duration, sr=44100):
    """Acordeão/Sanfona"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    signal += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
    signal += 0.3 * np.sin(2 * np.pi * freq * 3 * t)
    # Vibrato suave
    vibrato = 0.02 * np.sin(2 * np.pi * 6 * t)
    signal *= (1 + vibrato)
    return apply_envelope(signal, sr, attack=0.05, decay=0.0, sustain=1.0, release=0.1)


def synthesizer(freq, duration, sr=44100, wave_type="saw"):
    """Sintetizador genérico"""
    signal = subtractive_synthesis(freq, duration, sr, wave_type=wave_type, filter_cutoff=3000)
    return apply_envelope(signal, sr, attack=0.01, decay=0.1, sustain=0.8, release=0.2)


def celesta(freq, duration, sr=44100):
    """Celesta - som de sino suave"""
    signal = additive_synthesis(freq, duration, sr,
        harmonics=[1, 2.76, 5.4, 8.9],
        amplitudes=[1.0, 0.6, 0.3, 0.15],
        decays=[3.0, 4.0, 5.0, 6.0])
    return apply_envelope(signal, sr, attack=0.001, decay=0.5, sustain=0.1, release=1.0)


def glockenspiel(freq, duration, sr=44100):
    """Glockenspiel - metalofone brilhante"""
    signal = additive_synthesis(freq * 2, duration, sr,
        harmonics=[1, 3.0, 6.5],
        amplitudes=[1.0, 0.5, 0.2],
        decays=[2.0, 3.0, 4.0])
    return apply_envelope(signal, sr, attack=0.001, decay=0.3, sustain=0.1, release=0.5)


def xylophone(freq, duration, sr=44100):
    """Xilofone - madeira"""
    signal = additive_synthesis(freq, duration, sr,
        harmonics=[1, 3.0, 6.0],
        amplitudes=[1.0, 0.4, 0.15],
        decays=[4.0, 5.0, 6.0])
    return apply_envelope(signal, sr, attack=0.001, decay=0.2, sustain=0.05, release=0.3)


def marimba(freq, duration, sr=44100):
    """Marimba - madeira grave e quente"""
    signal = additive_synthesis(freq, duration, sr,
        harmonics=[1, 4.0],
        amplitudes=[1.0, 0.3],
        decays=[2.5, 3.5])
    return apply_envelope(signal, sr, attack=0.002, decay=0.4, sustain=0.2, release=0.5)


def vibraphone(freq, duration, sr=44100):
    """Vibrafone - metal com vibrato"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = additive_synthesis(freq, duration, sr,
        harmonics=[1, 4.0],
        amplitudes=[1.0, 0.3],
        decays=[1.5, 2.0])
    # Vibrato (motor do vibrafone)
    vibrato = 0.3 * np.sin(2 * np.pi * 5 * t)
    signal *= (1 + vibrato)
    return apply_envelope(signal, sr, attack=0.002, decay=0.5, sustain=0.4, release=0.5)


def keyboard(freq, duration, sr=44100):
    """Teclado eletrônico"""
    return synthesizer(freq, duration, sr, wave_type="triangle")


# ============================================================
# FAMÍLIA 3: SOPROS (18 instrumentos)
# ============================================================

def flute(freq, duration, sr=44100):
    """Flauta transversal"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    signal += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    signal += 0.1 * np.sin(2 * np.pi * freq * 3 * t)
    # Ruído de ar
    breath = np.random.randn(len(t)) * 0.05
    signal += breath
    return apply_envelope(signal, sr, attack=0.04, decay=0.0, sustain=1.0, release=0.06)


def clarinet(freq, duration, sr=44100):
    """Clarinete - harmônicos ímpares"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.zeros_like(t)
    for h in [1, 3, 5, 7, 9]:
        signal += np.sin(2 * np.pi * freq * h * t) / h
    return apply_envelope(signal, sr, attack=0.03, decay=0.0, sustain=1.0, release=0.05)


def oboe(freq, duration, sr=44100):
    """Oboé - palheta dupla"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.zeros_like(t)
    for h in range(1, 15):
        signal += np.sin(2 * np.pi * freq * h * t) / (h * 0.8)
    return apply_envelope(signal, sr, attack=0.02, decay=0.0, sustain=1.0, release=0.04)


def bassoon(freq, duration, sr=44100):
    """Fagote - grave de palheta dupla"""
    return oboe(freq / 2, duration, sr)


def soprano_sax(freq, duration, sr=44100):
    """Saxofone soprano"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.zeros_like(t)
    for h in range(1, 12):
        signal += np.sin(2 * np.pi * freq * h * t) / h
    vibrato = 0.02 * np.sin(2 * np.pi * 5 * t)
    signal *= (1 + vibrato)
    return apply_envelope(signal, sr, attack=0.03, decay=0.0, sustain=1.0, release=0.05)


def alto_sax(freq, duration, sr=44100):
    """Saxofone alto"""
    return soprano_sax(freq * 0.75, duration, sr)


def tenor_sax(freq, duration, sr=44100):
    """Saxofone tenor"""
    return soprano_sax(freq * 0.5, duration, sr)


def baritone_sax(freq, duration, sr=44100):
    """Saxofone barítono"""
    return soprano_sax(freq * 0.35, duration, sr)


def trumpet(freq, duration, sr=44100):
    """Trompete"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.zeros_like(t)
    for h in range(1, 16):
        signal += np.sin(2 * np.pi * freq * h * t) / h
    signal = apply_envelope(signal, sr, attack=0.02, decay=0.0, sustain=1.0, release=0.05)
    # Brilho metálico
    signal += np.random.randn(len(t)) * 0.02
    return signal / (np.max(np.abs(signal)) + 1e-10)


def trombone(freq, duration, sr=44100):
    """Trombone"""
    return trumpet(freq * 0.6, duration, sr)


def tuba(freq, duration, sr=44100):
    """Tuba - grave dos metais"""
    return trumpet(freq * 0.3, duration, sr)


def french_horn(freq, duration, sr=44100):
    """Trompa francesa"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.zeros_like(t)
    for h in range(1, 20):
        signal += np.sin(2 * np.pi * freq * h * t) / (h * 1.2)
    return apply_envelope(signal, sr, attack=0.05, decay=0.0, sustain=1.0, release=0.1)


def cornet(freq, duration, sr=44100):
    """Corneta"""
    return trumpet(freq * 1.1, duration, sr)


def flugelhorn(freq, duration, sr=44100):
    """Fliscorne - som mais suave que trompete"""
    signal = trumpet(freq, duration, sr)
    # Suavizar
    return np.convolve(signal, np.ones(5)/5, mode='same')


def euphonium(freq, duration, sr=44100):
    """Bombardino"""
    return french_horn(freq * 0.7, duration, sr)


def harmonica(freq, duration, sr=44100):
    """Gaita de boca"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    signal += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
    vibrato = 0.05 * np.sin(2 * np.pi * 8 * t)
    signal *= (1 + vibrato)
    return apply_envelope(signal, sr, attack=0.01, decay=0.0, sustain=1.0, release=0.03)


def recorder(freq, duration, sr=44100):
    """Flauta doce"""
    return flute(freq * 1.5, duration, sr)


def piccolo(freq, duration, sr=44100):
    """Piccolo - flauta aguda"""
    return flute(freq * 2, duration, sr)


def ocarina(freq, duration, sr=44100):
    """Ocarina"""
    signal = flute(freq, duration, sr)
    return np.convolve(signal, np.ones(3)/3, mode='same')


# ============================================================
# FAMÍLIA 4: PERCUSSÃO (25 instrumentos)
# ============================================================

def kick_drum(sr=44100, velocity=1.0):
    """Bumbo/Bombo"""
    t = np.linspace(0, 0.35, int(0.35 * sr), endpoint=False)
    freq_curve = 160 * np.exp(-t * 25) + 45
    phase = 2 * np.pi * np.cumsum(freq_curve) / sr
    signal = np.sin(phase) * np.exp(-t * 12) * velocity
    return signal / (np.max(np.abs(signal)) + 1e-10)


def snare_drum(sr=44100, velocity=1.0):
    """Caixa"""
    t = np.linspace(0, 0.22, int(0.22 * sr), endpoint=False)
    tone = np.sin(2 * np.pi * 195 * t) * np.exp(-t * 35)
    tone += 0.5 * np.sin(2 * np.pi * 330 * t) * np.exp(-t * 40)
    noise = np.random.randn(len(t)) * np.exp(-t * 22)
    signal = (0.4 * tone + 0.6 * noise) * velocity
    return signal / (np.max(np.abs(signal)) + 1e-10)


def hihat_closed(sr=44100, velocity=1.0):
    """Chimbal fechado"""
    t = np.linspace(0, 0.06, int(0.06 * sr), endpoint=False)
    noise = np.random.randn(len(t))
    filtered = np.diff(noise, prepend=noise[0])
    signal = filtered * np.exp(-t * 45) * velocity
    return signal / (np.max(np.abs(signal)) + 1e-10)


def hihat_open(sr=44100, velocity=1.0):
    """Chimbal aberto"""
    t = np.linspace(0, 0.25, int(0.25 * sr), endpoint=False)
    noise = np.random.randn(len(t))
    filtered = np.diff(noise, prepend=noise[0])
    signal = filtered * np.exp(-t * 10) * velocity
    return signal / (np.max(np.abs(signal)) + 1e-10)


def crash_cymbal(sr=44100, velocity=1.0):
    """Prato de ataque"""
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
    noise = np.random.randn(len(t))
    metallic = 0
    for f in [5000, 6500, 8000, 9500, 11000, 13000]:
        metallic += 0.15 * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
    signal = (noise * 0.4 + metallic * 0.4) * np.exp(-t * 3) * velocity
    return signal / (np.max(np.abs(signal)) + 1e-10)


def ride_cymbal(sr=44100, velocity=1.0):
    """Prato de condução"""
    t = np.linspace(0, 1.0, int(1.0 * sr), endpoint=False)
    noise = np.random.randn(len(t)) * 0.3
    metallic = 0
    for f in [6000, 7500, 9000]:
        metallic += 0.2 * np.sin(2 * np.pi * f * t)
    signal = (noise + metallic) * np.exp(-t * 5) * velocity
    return signal / (np.max(np.abs(signal)) + 1e-10)


def tom_drum(freq=120, sr=44100, velocity=1.0):
    """Tom-tom"""
    t = np.linspace(0, 0.3, int(0.3 * sr), endpoint=False)
    freq_curve = freq * np.exp(-t * 8) + freq * 0.7
    phase = 2 * np.pi * np.cumsum(freq_curve) / sr
    signal = np.sin(phase) * np.exp(-t * 10) * velocity
    return signal / (np.max(np.abs(signal)) + 1e-10)


def timpani(freq=80, sr=44100, velocity=1.0):
    """Tímpano"""
    t = np.linspace(0, 1.0, int(1.0 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 5)
    signal += np.random.randn(len(t)) * 0.1 * np.exp(-t * 10)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def pandeiro(sr=44100, velocity=1.0):
    """Pandeiro - instrumento brasileiro"""
    t = np.linspace(0, 0.15, int(0.15 * sr), endpoint=False)
    # Som de pele
    tone = np.sin(2 * np.pi * 250 * t) * np.exp(-t * 30)
    # Platinelas
    noise = np.random.randn(len(t)) * np.exp(-t * 40) * 0.5
    signal = (tone + noise) * velocity
    return signal / (np.max(np.abs(signal)) + 1e-10)


def surdo(freq=70, sr=44100, velocity=1.0):
    """Surdo - tambor grave brasileiro"""
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 8)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def tamborim(sr=44100, velocity=1.0):
    """Tamborim - pequeno tambor brasileiro"""
    t = np.linspace(0, 0.08, int(0.08 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * 800 * t) * np.exp(-t * 50)
    signal += np.random.randn(len(t)) * 0.3 * np.exp(-t * 60)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def cuica(sr=44100, velocity=1.0):
    """Cuíca - tambor de fricção brasileiro"""
    t = np.linspace(0, 0.3, int(0.3 * sr), endpoint=False)
    freq_curve = 400 * np.exp(-t * 10) + 200
    phase = 2 * np.pi * np.cumsum(freq_curve) / sr
    signal = np.sin(phase) * np.exp(-t * 15)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def berimbau(freq=200, sr=44100, velocity=1.0):
    """Berimbau - arco musical brasileiro"""
    signal = karplus_strong(freq, 0.3, sr, damping=0.997, brightness=0.4)
    return signal * velocity


def agogo(freq=600, sr=44100, velocity=1.0):
    """Agogô - sino duplo brasileiro"""
    t = np.linspace(0, 0.3, int(0.3 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 15)
    signal += np.sin(2 * np.pi * freq * 1.5 * t) * 0.5 * np.exp(-t * 20)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def reco_reco(sr=44100, velocity=1.0):
    """Reco-reco - raspador brasileiro"""
    t = np.linspace(0, 0.1, int(0.1 * sr), endpoint=False)
    noise = np.random.randn(len(t))
    # Efeito de raspagem
    envelope = np.sin(2 * np.pi * 30 * t) * np.exp(-t * 20)
    signal = noise * np.abs(envelope)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def chocalho(sr=44100, velocity=1.0):
    """Chocalho/Chocalho de platinela"""
    t = np.linspace(0, 0.15, int(0.15 * sr), endpoint=False)
    noise = np.random.randn(len(t))
    signal = noise * np.exp(-t * 30)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def triangle(sr=44100, velocity=1.0):
    """Triângulo"""
    t = np.linspace(0, 1.0, int(1.0 * sr), endpoint=False)
    signal = 0
    for f in [4000, 5500, 7000, 8500]:
        signal += np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi)) * 0.25
    signal *= np.exp(-t * 4)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def claves(sr=44100, velocity=1.0):
    """Claves - bastões de madeira"""
    t = np.linspace(0, 0.1, int(0.1 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * 2500 * t) * np.exp(-t * 50)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def woodblock(sr=44100, velocity=1.0):
    """Woodblock - bloco de madeira"""
    t = np.linspace(0, 0.08, int(0.08 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 60)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def castanets(sr=44100, velocity=1.0):
    """Castanholas"""
    t = np.linspace(0, 0.05, int(0.05 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * 3000 * t) * np.exp(-t * 80)
    signal += np.random.randn(len(t)) * 0.2 * np.exp(-t * 100)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def cowbell(sr=44100, velocity=1.0):
    """Cowbell - sino de vaca"""
    t = np.linspace(0, 0.3, int(0.3 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * 800 * t) * np.exp(-t * 15)
    signal += np.sin(2 * np.pi * 1200 * t) * 0.5 * np.exp(-t * 20)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def guiro(sr=44100, velocity=1.0):
    """Güiro - raspador latino"""
    t = np.linspace(0, 0.2, int(0.2 * sr), endpoint=False)
    noise = np.random.randn(len(t))
    envelope = np.abs(np.sin(2 * np.pi * 20 * t)) * np.exp(-t * 10)
    signal = noise * envelope
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def maracas(sr=44100, velocity=1.0):
    """Maracas"""
    t = np.linspace(0, 0.15, int(0.15 * sr), endpoint=False)
    noise = np.random.randn(len(t))
    signal = noise * np.exp(-t * 25)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def bongo_high(sr=44100, velocity=1.0):
    """Bongô agudo"""
    return tom_drum(300, sr, velocity)


def bongo_low(sr=44100, velocity=1.0):
    """Bongô grave"""
    return tom_drum(180, sr, velocity)


def conga(freq=200, sr=44100, velocity=1.0):
    """Conga"""
    t = np.linspace(0, 0.3, int(0.3 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 12)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def timbales(sr=44100, velocity=1.0):
    """Timbales"""
    t = np.linspace(0, 0.2, int(0.2 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * 400 * t) * np.exp(-t * 20)
    signal += np.random.randn(len(t)) * 0.2 * np.exp(-t * 30)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


# ============================================================
# FAMÍLIA 5: ELETRÔNICOS (20 instrumentos)
# ============================================================

def synth_lead(freq, duration, sr=44100):
    """Synth Lead - som principal"""
    signal = subtractive_synthesis(freq, duration, sr, wave_type="saw", filter_cutoff=4000)
    return apply_envelope(signal, sr, attack=0.01, decay=0.1, sustain=0.9, release=0.2)


def synth_pad(freq, duration, sr=44100):
    """Synth Pad - som de fundo"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    saw1 = 2 * (t * freq % 1) - 1
    saw2 = 2 * (t * freq * 1.003 % 1) - 1
    saw3 = 2 * (t * freq * 0.997 % 1) - 1
    signal = (saw1 + saw2 + saw3) / 3
    signal = np.convolve(signal, np.ones(15)/15, mode='same')
    return apply_envelope(signal, sr, attack=0.1, decay=0.0, sustain=1.0, release=0.2)


def synth_bass(freq, duration, sr=44100):
    """Synth Bass"""
    signal = subtractive_synthesis(freq / 2, duration, sr, wave_type="square", filter_cutoff=800)
    return apply_envelope(signal, sr, attack=0.005, decay=0.1, sustain=0.8, release=0.1)


def synth_arp(freq, duration, sr=44100):
    """Synth Arp - arpejo"""
    signal = subtractive_synthesis(freq, duration, sr, wave_type="square", filter_cutoff=3000)
    return apply_envelope(signal, sr, attack=0.001, decay=0.1, sustain=0.3, release=0.1)


def synth_pluck(freq, duration, sr=44100):
    """Synth Pluck - dedilhado sintético"""
    signal = karplus_strong(freq, duration, sr, damping=0.99, brightness=0.8)
    return signal


def synth_organ(freq, duration, sr=44100):
    """Synth Organ"""
    return organ(freq, duration, sr)


def synth_strings(freq, duration, sr=44100):
    """Synth Strings"""
    signal = synth_pad(freq, duration, sr)
    # Adicionar ataque mais definido
    return apply_envelope(signal, sr, attack=0.05, decay=0.1, sustain=0.9, release=0.3)


def synth_brass(freq, duration, sr=44100):
    """Synth Brass"""
    signal = subtractive_synthesis(freq, duration, sr, wave_type="saw", filter_cutoff=5000)
    return apply_envelope(signal, sr, attack=0.02, decay=0.0, sustain=1.0, release=0.1)


def synth_choir(freq, duration, sr=44100):
    """Synth Choir"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    signal += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
    signal += 0.3 * np.sin(2 * np.pi * freq * 3 * t)
    vibrato = 0.02 * np.sin(2 * np.pi * 5 * t)
    signal *= (1 + vibrato)
    return apply_envelope(signal, sr, attack=0.2, decay=0.0, sustain=1.0, release=0.5)


def synth_bell(freq, duration, sr=44100):
    """Synth Bell"""
    signal = additive_synthesis(freq, duration, sr,
        harmonics=[1, 2.76, 5.4, 8.9],
        amplitudes=[1.0, 0.6, 0.3, 0.15],
        decays=[2.0, 3.0, 4.0, 5.0])
    return apply_envelope(signal, sr, attack=0.001, decay=0.3, sustain=0.2, release=0.5)


def synth_fx(freq, duration, sr=44100):
    """Synth FX - efeito sonoro"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    freq_curve = np.linspace(freq, freq * 2, len(t))
    phase = 2 * np.pi * np.cumsum(freq_curve) / sr
    signal = np.sin(phase) * np.exp(-t * 5)
    return signal / (np.max(np.abs(signal)) + 1e-10)


def synth_noise(duration, sr=44100):
    """Synth Noise - ruído"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.random.randn(len(t)) * np.exp(-t * 10)
    return signal / (np.max(np.abs(signal)) + 1e-10)


def synth_sweep(duration, sr=44100):
    """Synth Sweep - varredura de frequência"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    freq_curve = np.linspace(100, 5000, len(t))
    phase = 2 * np.pi * np.cumsum(freq_curve) / sr
    signal = np.sin(phase)
    return signal / (np.max(np.abs(signal)) + 1e-10)


def synth_riser(duration, sr=44100):
    """Synth Riser - subida"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    freq_curve = np.linspace(200, 2000, len(t))
    phase = 2 * np.pi * np.cumsum(freq_curve) / sr
    signal = np.sin(phase) * np.linspace(0, 1, len(t))
    return signal / (np.max(np.abs(signal)) + 1e-10)


def synth_impact(sr=44100):
    """Synth Impact - impacto"""
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 10)
    signal += np.random.randn(len(t)) * 0.3 * np.exp(-t * 20)
    return signal / (np.max(np.abs(signal)) + 1e-10)


def sub_bass(freq, duration, sr=44100):
    """Sub Bass - grave profundo"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 3)
    return signal / (np.max(np.abs(signal)) + 1e-10)


def wobble_bass(freq, duration, sr=44100):
    """Wobble Bass - dubstep"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    lfo = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))  # LFO 4Hz
    signal = np.sin(2 * np.pi * freq * t) * lfo
    return signal / (np.max(np.abs(signal)) + 1e-10)


def fm_bass(freq, duration, sr=44100):
    """FM Bass"""
    return fm_synthesis(freq / 2, duration, sr, mod_ratio=1.5, mod_index=4.0)


def am_synth(freq, duration, sr=44100):
    """AM Synth - amplitude modulada"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    carrier = np.sin(2 * np.pi * freq * t)
    modulator = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
    signal = carrier * modulator
    return signal / (np.max(np.abs(signal)) + 1e-10)


def ring_mod(freq, duration, sr=44100):
    """Ring Modulator"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    carrier = np.sin(2 * np.pi * freq * t)
    modulator = np.sin(2 * np.pi * freq * 1.5 * t)
    signal = carrier * modulator
    return signal / (np.max(np.abs(signal)) + 1e-10)


# ============================================================
# FAMÍLIA 6: ÉTNICOS (12 instrumentos)
# ============================================================

def tabla(freq=200, sr=44100, velocity=1.0):
    """Tabla - percussão indiana"""
    t = np.linspace(0, 0.2, int(0.2 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 20)
    signal += np.random.randn(len(t)) * 0.2 * np.exp(-t * 30)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def djembe(freq=150, sr=44100, velocity=1.0):
    """Djembe - tambor africano"""
    t = np.linspace(0, 0.3, int(0.3 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 15)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def kalimba(freq=500, sr=44100, velocity=1.0):
    """Kalimba - piano de polegar africano"""
    signal = additive_synthesis(freq, 0.5, sr,
        harmonics=[1, 3.0],
        amplitudes=[1.0, 0.3],
        decays=[2.0, 3.0])
    return signal * velocity


def mbira(freq=400, sr=44100, velocity=1.0):
    """Mbira - similar à kalimba"""
    return kalimba(freq * 0.8, sr, velocity)


def didgeridoo(freq=80, sr=44100, velocity=1.0):
    """Didgeridoo - instrumento aborígene australiano"""
    t = np.linspace(0, 1.0, int(1.0 * sr), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    signal += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
    signal += 0.3 * np.sin(2 * np.pi * freq * 3 * t)
    # Drone contínuo
    return apply_envelope(signal, sr, attack=0.1, decay=0.0, sustain=1.0, release=0.2) * velocity


def shakuhachi(freq=300, sr=44100, velocity=1.0):
    """Shakuhachi - flauta japonesa"""
    signal = flute(freq, 1.0, sr)
    # Adicionar respiração
    t = np.linspace(0, 1.0, int(1.0 * sr), endpoint=False)
    breath = np.random.randn(len(t)) * 0.1
    return (signal + breath) * velocity


def erhu(freq=400, sr=44100, velocity=1.0):
    """Erhu - violino chinês"""
    return violin(freq, 1.0, sr, vibrato_rate=6.0, vibrato_depth=0.02) * velocity


def guzheng(freq=300, sr=44100, velocity=1.0):
    """Guzheng - cítara chinesa"""
    return karplus_strong(freq, 1.0, sr, damping=0.997, brightness=0.5) * velocity


def gamelan(freq=250, sr=44100, velocity=1.0):
    """Gamelan - metalofone indonésio"""
    signal = additive_synthesis(freq, 1.0, sr,
        harmonics=[1, 2.76, 5.4],
        amplitudes=[1.0, 0.5, 0.2],
        decays=[2.0, 3.0, 4.0])
    return signal * velocity


def steel_drum(freq=400, sr=44100, velocity=1.0):
    """Steel Drum - tambor de aço caribenho"""
    signal = additive_synthesis(freq, 0.5, sr,
        harmonics=[1, 2.0, 3.0],
        amplitudes=[1.0, 0.6, 0.3],
        decays=[3.0, 4.0, 5.0])
    return signal * velocity


def handpan(freq=300, sr=44100, velocity=1.0):
    """Handpan - instrumento moderno"""
    signal = additive_synthesis(freq, 1.0, sr,
        harmonics=[1, 2.0, 3.0],
        amplitudes=[1.0, 0.5, 0.2],
        decays=[2.0, 3.0, 4.0])
    return signal * velocity


def bouzouki(freq=350, sr=44100, velocity=1.0):
    """Bouzouki - instrumento grego"""
    return karplus_strong(freq, 1.0, sr, damping=0.995, brightness=0.7) * velocity


# ============================================================
# REGISTRO DE INSTRUMENTOS
# ============================================================

INSTRUMENTS = {
    # Cordas
    "violin": violin, "viola": viola, "cello": cello, "double_bass": double_bass,
    "acoustic_guitar": acoustic_guitar, "electric_guitar": electric_guitar,
    "electric_bass": electric_bass, "harp": harp, "ukulele": ukulele,
    "banjo": banjo, "mandolin": mandolin, "cavaquinho": cavaquinho,
    "sitar": sitar, "koto": koto, "balalaika": balalaika, "rabeca": rabeca,
    
    # Teclas
    "piano": piano, "electric_piano": electric_piano, "organ": organ,
    "harpsichord": harpsichord, "accordion": accordion, "synthesizer": synthesizer,
    "celesta": celesta, "glockenspiel": glockenspiel, "xylophone": xylophone,
    "marimba": marimba, "vibraphone": vibraphone, "keyboard": keyboard,
    
    # Sopros
    "flute": flute, "clarinet": clarinet, "oboe": oboe, "bassoon": bassoon,
    "soprano_sax": soprano_sax, "alto_sax": alto_sax, "tenor_sax": tenor_sax,
    "baritone_sax": baritone_sax, "trumpet": trumpet, "trombone": trombone,
    "tuba": tuba, "french_horn": french_horn, "cornet": cornet,
    "flugelhorn": flugelhorn, "euphonium": euphonium, "harmonica": harmonica,
    "recorder": recorder, "piccolo": piccolo, "ocarina": ocarina,
    
    # Percussão
    "kick": kick_drum, "snare": snare_drum, "hihat_closed": hihat_closed,
    "hihat_open": hihat_open, "crash": crash_cymbal, "ride": ride_cymbal,
    "tom": tom_drum, "timpani": timpani, "pandeiro": pandeiro, "surdo": surdo,
    "tamborim": tamborim, "cuica": cuica, "berimbau": berimbau, "agogo": agogo,
    "reco_reco": reco_reco, "chocalho": chocalho, "triangle": triangle,
    "claves": claves, "woodblock": woodblock, "castanets": castanets,
    "cowbell": cowbell, "guiro": guiro, "maracas": maracas,
    "bongo_high": bongo_high, "bongo_low": bongo_low, "conga": conga,
    "timbales": timbales,
    
    # Eletrônicos
    "synth_lead": synth_lead, "synth_pad": synth_pad, "synth_bass": synth_bass,
    "synth_arp": synth_arp, "synth_pluck": synth_pluck, "synth_organ": synth_organ,
    "synth_strings": synth_strings, "synth_brass": synth_brass,
    "synth_choir": synth_choir, "synth_bell": synth_bell, "synth_fx": synth_fx,
    "synth_noise": synth_noise, "synth_sweep": synth_sweep, "synth_riser": synth_riser,
    "synth_impact": synth_impact, "sub_bass": sub_bass, "wobble_bass": wobble_bass,
    "fm_bass": fm_bass, "am_synth": am_synth, "ring_mod": ring_mod,
    
    # Étnicos
    "tabla": tabla, "djembe": djembe, "kalimba": kalimba, "mbira": mbira,
    "didgeridoo": didgeridoo, "shakuhachi": shakuhachi, "erhu": erhu,
    "guzheng": guzheng, "gamelan": gamelan, "steel_drum": steel_drum,
    "handpan": handpan, "bouzouki": bouzouki,
}


def get_instrument(name):
    """Retorna função de instrumento pelo nome"""
    return INSTRUMENTS.get(name.lower())


def list_instruments():
    """Lista todos os instrumentos disponíveis"""
    return sorted(INSTRUMENTS.keys())


def get_instruments_by_family(family):
    """Retorna instrumentos por família"""
    families = {
        "strings": ["violin", "viola", "cello", "double_bass", "acoustic_guitar", 
                   "electric_guitar", "electric_bass", "harp", "ukulele", "banjo",
                   "mandolin", "cavaquinho", "sitar", "koto", "balalaika", "rabeca"],
        "keys": ["piano", "electric_piano", "organ", "harpsichord", "accordion",
                "synthesizer", "celesta", "glockenspiel", "xylophone", "marimba",
                "vibraphone", "keyboard"],
        "winds": ["flute", "clarinet", "oboe", "bassoon", "soprano_sax", "alto_sax",
                 "tenor_sax", "baritone_sax", "trumpet", "trombone", "tuba",
                 "french_horn", "cornet", "flugelhorn", "euphonium", "harmonica",
                 "recorder", "piccolo", "ocarina"],
        "percussion": ["kick", "snare", "hihat_closed", "hihat_open", "crash",
                      "ride", "tom", "timpani", "pandeiro", "surdo", "tamborim",
                      "cuica", "berimbau", "agogo", "reco_reco", "chocalho",
                      "triangle", "claves", "woodblock", "castanets", "cowbell",
                      "guiro", "maracas", "bongo_high", "bongo_low", "conga", "timbales"],
        "electronic": ["synth_lead", "synth_pad", "synth_bass", "synth_arp",
                      "synth_pluck", "synth_organ", "synth_strings", "synth_brass",
                      "synth_choir", "synth_bell", "synth_fx", "synth_noise",
                      "synth_sweep", "synth_riser", "synth_impact", "sub_bass",
                      "wobble_bass", "fm_bass", "am_synth", "ring_mod"],
        "ethnic": ["tabla", "djembe", "kalimba", "mbira", "didgeridoo",
                  "shakuhachi", "erhu", "guzheng", "gamelan", "steel_drum",
                  "handpan", "bouzouki"],
    }
    return families.get(family.lower(), [])


if __name__ == "__main__":
    print(f"🎸 Total de instrumentos: {len(INSTRUMENTS)}")
    print("\nFamílias:")
    for family in ["strings", "keys", "winds", "percussion", "electronic", "ethnic"]:
        instruments = get_instruments_by_family(family)
        print(f"  {family}: {len(instruments)}")
