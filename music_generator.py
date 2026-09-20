#!/usr/bin/env python3
"""
🎵 IA MUSIC GENERATOR - COM VALIDAÇÃO DE PARÂMETROS POR ESTILO
CORREÇÃO: Parâmetros AI agora são validados contra restrições de cada estilo
"""
import os, sys, json, time, gc, argparse
import numpy as np
from pathlib import Path

try:
    from scipy.signal import lfilter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from music_intelligence import FeedbackLoop, AutoMixer, MusicMemory

OUTPUT_DIR = "song_output"
MODEL_DIR = "models"
MUSIC_DIR = "music_input"
AE_DIR = os.path.join(MODEL_DIR, "autoencoder")
SAMPLE_CACHE = {}


# ============================================================
# STYLE CONSTRAINTS - CORREÇÃO CRÍTICA
# ============================================================

STYLE_CONSTRAINTS = {
    "cinematic": {
        "bpm_range": (90, 140),
        "intensity_min": 0.7, "intensity_max": 1.0,
        "brightness_min": 0.5, "brightness_max": 0.9,
        "bass_weight_min": 0.5, "bass_weight_max": 0.9,
        "dynamics_range_min": 0.6, "dynamics_range_max": 1.0,
        "harmonic_complexity_min": 0.5, "harmonic_complexity_max": 0.9,
    },
    "epic": {
        "bpm_range": (100, 160),
        "intensity_min": 0.8, "intensity_max": 1.0,
        "brightness_min": 0.6, "brightness_max": 1.0,
        "bass_weight_min": 0.6, "bass_weight_max": 1.0,
        "dynamics_range_min": 0.7, "dynamics_range_max": 1.0,
        "harmonic_complexity_min": 0.6, "harmonic_complexity_max": 1.0,
    },
    "ambient": {
        "bpm_range": (40, 90),
        "intensity_min": 0.2, "intensity_max": 0.6,
        "brightness_min": 0.1, "brightness_max": 0.5,
        "bass_weight_min": 0.1, "bass_weight_max": 0.5,
        "rhythmic_density_min": 0.1, "rhythmic_density_max": 0.4,
        "dynamics_range_min": 0.3, "dynamics_range_max": 0.7,
    },
    "electronic": {
        "bpm_range": (110, 150),
        "intensity_min": 0.6, "intensity_max": 1.0,
        "brightness_min": 0.5, "brightness_max": 0.9,
        "bass_weight_min": 0.6, "bass_weight_max": 1.0,
        "rhythmic_density_min": 0.6, "rhythmic_density_max": 1.0,
    },
    "rock": {
        "bpm_range": (100, 160),
        "intensity_min": 0.7, "intensity_max": 1.0,
        "brightness_min": 0.5, "brightness_max": 0.9,
        "bass_weight_min": 0.6, "bass_weight_max": 0.9,
        "dynamics_range_min": 0.6, "dynamics_range_max": 1.0,
    },
    "jazz": {
        "bpm_range": (80, 160),
        "intensity_min": 0.4, "intensity_max": 0.8,
        "brightness_min": 0.4, "brightness_max": 0.8,
        "harmonic_complexity_min": 0.7, "harmonic_complexity_max": 1.0,
        "dynamics_range_min": 0.6, "dynamics_range_max": 1.0,
    },
    "classical": {
        "bpm_range": (60, 140),
        "intensity_min": 0.4, "intensity_max": 0.9,
        "brightness_min": 0.4, "brightness_max": 0.9,
        "harmonic_complexity_min": 0.6, "harmonic_complexity_max": 1.0,
        "dynamics_range_min": 0.7, "dynamics_range_max": 1.0,
    },
    "breakcore": {
        "bpm_range": (160, 230),
        "intensity_min": 0.8, "intensity_max": 1.0,
        "brightness_min": 0.6, "brightness_max": 1.0,
        "rhythmic_density_min": 0.8, "rhythmic_density_max": 1.0,
    },
    "pop": {
        "bpm_range": (90, 130),
        "intensity_min": 0.5, "intensity_max": 0.9,
        "brightness_min": 0.5, "brightness_max": 0.9,
        "bass_weight_min": 0.5, "bass_weight_max": 0.8,
    },
    "dark": {
        "bpm_range": (60, 120),
        "intensity_min": 0.5, "intensity_max": 0.8,
        "brightness_min": 0.1, "brightness_max": 0.4,
        "bass_weight_min": 0.6, "bass_weight_max": 1.0,
        "harmonic_complexity_min": 0.4, "harmonic_complexity_max": 0.8,
    },
    "bossfight": {
        "bpm_range": (130, 180),
        "intensity_min": 0.9, "intensity_max": 1.0,
        "brightness_min": 0.6, "brightness_max": 1.0,
        "bass_weight_min": 0.7, "bass_weight_max": 1.0,
        "dynamics_range_min": 0.8, "dynamics_range_max": 1.0,
    },
}


def validate_and_adjust_params(style, ai_params):
    """
    CORREÇÃO CRÍTICA: Ajusta parâmetros AI para serem coerentes com o estilo
    
    ANTES: Música "cinematic" com BPM 112, brightness 0.4 (incoerente)
    DEPOIS: Música "cinematic" com BPM 90-140, brightness 0.5-0.9 (coerente)
    """
    constraints = STYLE_CONSTRAINTS.get(style, STYLE_CONSTRAINTS.get("pop", {}))
    adjusted = ai_params.copy()
    
    # Ajustar BPM
    if "bpm_range" in constraints and "bpm" in adjusted:
        bpm_min, bpm_max = constraints["bpm_range"]
        current_bpm = adjusted["bpm"]
        if current_bpm < bpm_min:
            adjusted["bpm"] = bpm_min + np.random.randint(0, 10)
        elif current_bpm > bpm_max:
            adjusted["bpm"] = bpm_max - np.random.randint(0, 10)
    
    # Ajustar parâmetros numéricos com mínimos/máximos
    for param_name, value in list(adjusted.items()):
        if param_name == "bpm":
            continue
        if not isinstance(value, (int, float)):
            continue
        
        min_key = f"{param_name}_min"
        max_key = f"{param_name}_max"
        
        if min_key in constraints:
            adjusted[param_name] = max(float(value), constraints[min_key])
        if max_key in constraints:
            adjusted[param_name] = min(float(adjusted[param_name]), constraints[max_key])
    
    return adjusted


def get_dynamic_seed():
    return int(time.time() * 1000) % (2 ** 32) ^ np.random.randint(0, 2 ** 31)


def get_next_song_number():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")]
    if not existing: return 1
    numbers = []
    for f in existing:
        try: numbers.append(int(f.replace(".wav", "")))
        except: pass
    return max(numbers, default=0) + 1


def save_song(audio, sr, metadata=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    number = get_next_song_number()
    filepath = os.path.join(OUTPUT_DIR, f"{number}.wav")
    try:
        import soundfile as sf
        sf.write(filepath, audio, sr)
    except ImportError:
        import wave
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(filepath, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())
    if metadata:
        meta_path = os.path.join(OUTPUT_DIR, f"{number}.json")
        metadata["song_number"] = number
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"💾 Salvo: {filepath}")
    return filepath, number


# ============================================================
# INSTRUMENTOS (mantidos)
# ============================================================

def make_kick(sr=44100, velocity=1.0):
    key = ("kick", sr, velocity)
    if key in SAMPLE_CACHE: return SAMPLE_CACHE[key].copy()
    t = np.linspace(0, 0.35, int(0.35 * sr), endpoint=False)
    freq_curve = 160 * np.exp(-t * 25) + 45
    phase = 2 * np.pi * np.cumsum(freq_curve) / sr
    signal = np.sin(phase) * np.exp(-t * 12) * velocity
    result = signal / (np.max(np.abs(signal)) + 1e-10)
    SAMPLE_CACHE[key] = result
    return result.copy()


def make_snare(sr=44100, velocity=1.0):
    key = ("snare", sr, velocity)
    if key in SAMPLE_CACHE: return SAMPLE_CACHE[key].copy()
    t = np.linspace(0, 0.22, int(0.22 * sr), endpoint=False)
    tone = np.sin(2 * np.pi * 195 * t) * np.exp(-t * 35) + 0.5 * np.sin(2 * np.pi * 330 * t) * np.exp(-t * 40)
    noise = np.random.randn(len(t)) * np.exp(-t * 22)
    signal = (0.4 * tone + 0.6 * noise) * velocity
    result = signal / (np.max(np.abs(signal)) + 1e-10)
    SAMPLE_CACHE[key] = result
    return result.copy()


def make_hihat(sr=44100, velocity=1.0):
    key = ("hihat", sr, velocity)
    if key in SAMPLE_CACHE: return SAMPLE_CACHE[key].copy()
    t = np.linspace(0, 0.06, int(0.06 * sr), endpoint=False)
    noise = np.random.randn(len(t))
    filtered = np.diff(noise, prepend=noise[0])
    result = filtered * np.exp(-t * 45) * velocity / (np.max(np.abs(filtered)) + 1e-10)
    SAMPLE_CACHE[key] = result
    return result.copy()


def make_tom(freq=120, sr=44100, velocity=1.0):
    key = ("tom", freq, sr, velocity)
    if key in SAMPLE_CACHE: return SAMPLE_CACHE[key].copy()
    t = np.linspace(0, 0.3, int(0.3 * sr), endpoint=False)
    freq_curve = freq * np.exp(-t * 8) + freq * 0.7
    phase = 2 * np.pi * np.cumsum(freq_curve) / sr
    signal = np.sin(phase) * np.exp(-t * 10) * velocity
    result = signal / (np.max(np.abs(signal)) + 1e-10)
    SAMPLE_CACHE[key] = result
    return result.copy()


def make_crash(sr=44100, velocity=1.0):
    key = ("crash", sr, velocity)
    if key in SAMPLE_CACHE: return SAMPLE_CACHE[key].copy()
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
    noise = np.random.randn(len(t))
    metallic = 0
    for f in [5000, 6500, 8000, 9500, 11000, 13000]:
        metallic += 0.15 * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
    signal = (noise * 0.4 + metallic * 0.4) * np.exp(-t * 3) * velocity
    result = signal / (np.max(np.abs(signal)) + 1e-10)
    SAMPLE_CACHE[key] = result
    return result.copy()


def karplus_strong(freq, duration, sr=44100, damping=0.996, brightness=0.5):
    N = max(2, int(sr / freq)); n_samples = int(duration * sr)
    x = np.zeros(n_samples)
    x[:min(N, n_samples)] = np.random.uniform(-1, 1, min(N, n_samples))
    if HAS_SCIPY:
        a = np.zeros(N + 2); a[0] = 1.0; a[N] = -damping * brightness
        a[N + 1] = -damping * (1.0 - brightness)
        y = lfilter([1.0], a, x)
    else:
        y = np.zeros(n_samples); delay = np.random.uniform(-1, 1, N)
        for i in range(n_samples):
            y[i] = delay[i % N]
            delay[i % N] = damping * 0.5 * (delay[i % N] + delay[(i + 1) % N])
    return y / (np.max(np.abs(y)) + 1e-10)


def piano_note(freq, duration, sr=44100, velocity=1.0):
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.zeros_like(t)
    for h, amp, dec in zip([1,2,3,4,5,6,7,8],[1,.6,.35,.25,.18,.14,.11,.09],[2.5,2.2,2,1.8,1.6,1.4,1.3,1.2]):
        inharmonic = 1.0 + 0.00008 * (h ** 2)
        signal += amp * np.sin(2 * np.pi * freq * h * inharmonic * t) * np.exp(-t * dec)
    atk = int(0.003 * sr)
    if 0 < atk < len(signal):
        signal[:atk] *= np.linspace(0, 1, atk)
    signal += np.random.randn(len(signal)) * 0.03 * np.exp(-t * 30)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)


def synth_pad(freq, duration, sr=44100):
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    saw1 = 2 * (t * freq % 1) - 1
    saw2 = 2 * (t * freq * 1.003 % 1) - 1
    saw3 = 2 * (t * freq * 0.997 % 1) - 1
    signal = (saw1 + saw2 + saw3) / 3
    signal = np.convolve(signal, np.ones(15) / 15, mode="same")
    envelope = np.ones_like(t)
    atk = min(int(0.1 * sr), len(t) // 3)
    rel = min(int(0.2 * sr), len(t) // 3)
    if atk > 0: envelope[:atk] = np.linspace(0, 1, atk)
    if rel > 0: envelope[-rel:] = np.linspace(1, 0, rel)
    return signal * envelope * 0.3 / (np.max(np.abs(signal)) + 1e-10)


def violin_note(freq, duration, sr=44100):
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    vibrato = 0.015 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(freq * (1.0 + vibrato)) / sr
    signal = np.zeros_like(t)
    for h in range(1, 12): signal += np.sin(h * phase) / (h * 1.2)
    envelope = np.ones_like(t)
    atk = int(min(0.08, duration * 0.2) * sr)
    rel = int(min(0.05, duration * 0.1) * sr)
    if 0 < atk < len(t): envelope[:atk] = np.linspace(0, 1, atk) ** 0.5
    if 0 < rel < len(t): envelope[-rel:] = np.linspace(1, 0, rel)
    return (signal * envelope * 0.3) / (np.max(np.abs(signal)) + 1e-10)


def flute_note(freq, duration, sr=44100):
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    signal = np.sin(2*np.pi*freq*t) + 0.3*np.sin(2*np.pi*freq*2*t) + 0.1*np.sin(2*np.pi*freq*3*t)
    breath = np.random.randn(len(t)) * 0.05
    if len(breath) > 10: breath = np.convolve(breath, np.ones(10)/10, mode="same")
    envelope = np.ones_like(t)
    atk = int(0.04 * sr); rel = int(0.06 * sr)
    if 0 < atk < len(t): envelope[:atk] = np.linspace(0, 1, atk)
    if 0 < rel < len(t): envelope[-rel:] = np.linspace(1, 0, rel)
    return (signal*envelope*0.3 + breath*envelope) / (np.max(np.abs(signal)) + 1e-10)


def brass_note(freq, duration, sr=44100):
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    phase = 2 * np.pi * freq * t
    signal = np.zeros_like(t)
    for h in range(1, 16): signal += np.sin(h * phase) / h
    envelope = np.ones_like(t)
    atk = int(0.02 * sr)
    if 0 < atk < len(t):
        env = np.linspace(0, 1.3, atk)
        split = int(atk * 0.7)
        if split < atk: env[split:] = np.linspace(1.3, 1.0, atk - split)
        envelope[:atk] = env
    rel = int(0.05 * sr)
    if 0 < rel < len(t): envelope[-rel:] = np.linspace(1, 0, rel)
    return signal * envelope * 0.25 * 1.5 / (np.max(np.abs(signal)) + 1e-10)


# ============================================================
# MUSIC THEORY
# ============================================================

ALL_PROGRESSIONS = [
    [[0,2,4],[5,0,2],[3,5,0],[4,6,1]],
    [[0,2,4],[3,5,0],[4,6,1],[5,0,2]],
    [[5,0,2],[3,5,0],[0,2,4],[4,6,1]],
    [[0,2,4],[0,2,4],[5,0,2],[4,6,1]],
    [[1,3,5],[4,6,1],[0,2,4],[5,0,2]],
    [[0,2,4,6],[4,6,1,3],[5,0,2,4],[0,2,4,6]],
]

ALL_SCALES = {
    "major": [0,2,4,5,7,9,11],
    "minor": [0,2,3,5,7,8,10],
    "dorian": [0,2,3,5,7,9,10],
    "phrygian": [0,1,3,5,7,8,10],
    "lydian": [0,2,4,6,7,9,11],
    "mixolydian": [0,2,4,5,7,9,10],
    "harmonic_minor": [0,2,3,5,7,8,11],
    "pentatonic_major": [0,2,4,7,9],
}


def note_to_freq(semitone, base_freq=261.63):
    return base_freq * (2 ** (semitone / 12.0))


class SongStructure:
    STRUCTURES = {
        "pop": ["intro","verse","chorus","verse","chorus","bridge","chorus","outro"],
        "rock": ["intro","verse","chorus","verse","chorus","solo","chorus","outro"],
        "electronic": ["intro","buildup","drop","breakdown","buildup","drop","outro"],
        "cinematic": ["intro","theme","development","climax","resolution","outro"],
        "jazz": ["intro","head","solo1","head","solo2","head","outro"],
        "ambient": ["intro","section1","section2","section3","section4","outro"],
        "classical": ["exposition","development","recapitulation","coda"],
        "breakcore": ["intro","chaos1","break","chaos2","break","chaos3","outro"],
        "dark": ["intro","verse","chorus","verse","chorus","bridge","chorus","outro"],
        "epic": ["intro","theme","buildup","climax","resolution","outro"],
        "bossfight": ["intro","phase1","transition","phase2","climax","outro"],
    }
    SECTION_ENERGY = {
        "intro": 0.3, "verse": 0.5, "chorus": 0.9, "bridge": 0.6, "outro": 0.4,
        "buildup": 0.7, "drop": 1.0, "breakdown": 0.2, "solo": 0.7, "head": 0.6,
        "theme": 0.5, "development": 0.7, "climax": 1.0, "resolution": 0.5,
        "exposition": 0.5, "recapitulation": 0.7, "coda": 0.4,
        "chaos1": 0.8, "chaos2": 0.9, "chaos3": 1.0, "break": 0.3,
        "section1": 0.4, "section2": 0.5, "section3": 0.6, "section4": 0.5,
        "phase1": 0.7, "phase2": 0.9, "transition": 0.6,
    }
    
    def __init__(self, style="pop", duration=45, bpm=120):
        self.style = style; self.duration = duration; self.bpm = bpm
        self.sections = self._generate_sections()
        self.section_times = self._calculate_times()
    
    def _generate_sections(self):
        base = self.STRUCTURES.get(self.style, self.STRUCTURES["pop"])
        sections = []
        for section in base:
            if section == "intro": bars = np.random.choice([4, 8])
            elif section in ["verse","head","theme","exposition","phase1"]: bars = np.random.choice([8, 12, 16])
            elif section in ["chorus","climax","drop","recapitulation","phase2"]: bars = np.random.choice([8, 12])
            elif section == "bridge" or section == "transition": bars = np.random.choice([4, 8])
            elif section == "buildup": bars = np.random.choice([4, 8])
            elif section in ["breakdown","break","coda"]: bars = np.random.choice([4, 8])
            elif section in ["solo","development","chaos1","chaos2","chaos3"]: bars = np.random.choice([8, 12, 16])
            elif section == "outro": bars = np.random.choice([4, 8])
            else: bars = 8
            sections.append({"name": section, "bars": bars, "energy": self.SECTION_ENERGY.get(section, 0.5)})
        return sections
    
    def _calculate_times(self):
        beats_per_bar = 4
        beat_duration = 60.0 / self.bpm
        times = []; current_time = 0.0
        for section in self.sections:
            section_duration = section["bars"] * beats_per_bar * beat_duration
            times.append({
                "name": section["name"], "start": current_time,
                "end": current_time + section_duration,
                "duration": section_duration, "energy": section["energy"]
            })
            current_time += section_duration
        total = current_time
        if total > 0:
            scale = self.duration / total
            for t in times:
                t["start"] *= scale; t["end"] *= scale; t["duration"] *= scale
        return times
    
    def get_section_at_time(self, time):
        for s in self.section_times:
            if s["start"] <= time < s["end"]: return s
        return self.section_times[-1]
    
    def get_energy_at_time(self, time):
        s = self.get_section_at_time(time)
        pos = (time - s["start"]) / s["duration"] if s["duration"] > 0 else 0
        base = s["energy"]
        if s["name"] == "buildup": return base * (0.3 + 0.7 * pos)
        elif s["name"] == "drop": return 1.0 if pos < 0.1 else base * (1.0 - 0.2 * (pos - 0.1) / 0.9)
        elif s["name"] == "outro": return base * (1.0 - pos * 0.7)
        else: return base + 0.1 * np.sin(pos * np.pi * 2)


# ============================================================
# GERAÇÃO COM VALIDAÇÃO DE PARÂMETROS
# ============================================================

def generate_with_ai(duration, ai_params, style="pop", sr=44100):
    """
    Geração COM VALIDAÇÃO DE PARÂMETROS POR ESTILO
    """
    seed = get_dynamic_seed()
    np.random.seed(seed)
    
    # VALIDAR E AJUSTAR PARÂMETROS - CORREÇÃO CRÍTICA
    original_params = ai_params.copy()
    ai_params = validate_and_adjust_params(style, ai_params)
    
    # Log das correções
    if original_params != ai_params:
        print("  🔧 Parâmetros ajustados para estilo '" + style + "':")
        for key in ai_params:
            if key in original_params and original_params[key] != ai_params[key]:
                print(f"     {key}: {original_params[key]:.3f} → {ai_params[key]:.3f}")
    
    bpm = int(ai_params.get("bpm", 120))
    intensity = float(ai_params.get("intensity", 0.7))
    brightness = float(ai_params.get("brightness", 0.5))
    bass_weight = float(ai_params.get("bass_weight", 0.5))
    rhythmic_density = float(ai_params.get("rhythmic_density", 0.5))
    harmonic_complexity = float(ai_params.get("harmonic_complexity", 0.5))
    dynamics_range = float(ai_params.get("dynamics_range", 0.7))
    
    print(f"  🧠 Parâmetros AI (após validação):")
    print(f"     BPM: {bpm}")
    print(f"     Intensidade: {intensity:.2f}")
    print(f"     Brilho: {brightness:.2f}")
    print(f"     Peso do grave: {bass_weight:.2f}")
    print(f"     Densidade rítmica: {rhythmic_density:.2f}")
    print(f"     Complexidade: {harmonic_complexity:.2f}")
    print(f"     Estilo: {style}")
    
    freq_options = [196.0, 220.0, 261.63, 293.66, 349.23]
    base_freq = freq_options[min(int(brightness * len(freq_options)), len(freq_options) - 1)]
    
    if harmonic_complexity > 0.7:
        scale = ALL_SCALES["harmonic_minor"]
    elif harmonic_complexity > 0.5:
        scale = ALL_SCALES["dorian"]
    elif harmonic_complexity > 0.3:
        scale = ALL_SCALES["minor"]
    else:
        scale = ALL_SCALES["major"]
    
    progression = ALL_PROGRESSIONS[np.random.randint(0, len(ALL_PROGRESSIONS))]
    structure = SongStructure(style=style, duration=duration, bpm=bpm)
    
    print(f"   🎼 Estrutura: {[s['name'] for s in structure.sections]}")
    
    total_samples = int(duration * sr)
    beat_duration = 60.0 / bpm
    
    drums_track = np.zeros(total_samples)
    bass_track = np.zeros(total_samples)
    chords_track = np.zeros(total_samples)
    melody_track = np.zeros(total_samples)
    
    kick_sample = make_kick(sr); snare_sample = make_snare(sr)
    hihat_sample = make_hihat(sr)
    tom1_sample = make_tom(200, sr); tom2_sample = make_tom(150, sr); tom3_sample = make_tom(100, sr)
    crash_sample = make_crash(sr)
    n_beats = int(duration / beat_duration)
    
    drum_prob = 0.3 + rhythmic_density * 0.5
    for beat in range(n_beats):
        time = beat * beat_duration
        section = structure.get_section_at_time(time)
        sname = section["name"]
        energy = structure.get_energy_at_time(time) * intensity
        vel = 0.3 + 0.6 * energy
        pos = int(beat * beat_duration * sr)
        
        if sname in ["intro", "breakdown", "break"]:
            kick_beats = [0]
        elif sname in ["chorus", "drop", "climax", "phase2"]:
            kick_beats = [0, 2]
        else:
            kick_beats = [0, 2] if np.random.random() < drum_prob else [0]
        
        for kb in kick_beats:
            kp = pos + int(kb * beat_duration * sr / 4 * 2)
            if kp + len(kick_sample) <= total_samples:
                drums_track[kp:kp + len(kick_sample)] += kick_sample * vel * 0.8
        
        if sname not in ["intro", "breakdown", "break"]:
            for sb in [1, 3]:
                sp = pos + int(sb * beat_duration * sr / 4 * 2)
                if sp + len(snare_sample) <= total_samples:
                    drums_track[sp:sp + len(snare_sample)] += snare_sample * vel * 0.7
        
        if sname == "intro":
            hat_subdivs = [0]
        elif sname in ["drop", "climax"] or rhythmic_density > 0.7:
            hat_subdivs = [0, 0.25, 0.5, 0.75]
        elif rhythmic_density > 0.4:
            hat_subdivs = [0, 0.5]
        else:
            hat_subdivs = [0]
        
        for sub in hat_subdivs:
            hp = pos + int(sub * beat_duration * sr)
            if hp + len(hihat_sample) <= total_samples:
                vol = 0.35 if sub == 0 else 0.2
                drums_track[hp:hp + len(hihat_sample)] += hihat_sample * vel * vol
    
    bass_vol = 0.2 + bass_weight * 0.4
    for beat in range(n_beats):
        time = beat * beat_duration
        section = structure.get_section_at_time(time)
        energy = structure.get_energy_at_time(time) * intensity
        chord_idx = (beat // 4) % len(progression)
        root = progression[chord_idx][0]
        root_freq = note_to_freq(scale[root % len(scale)], base_freq) / 2
        vel = (0.3 + 0.7 * energy) * bass_vol
        
        if section["name"] in ["intro", "breakdown", "break"]:
            if beat % 4 == 0:
                note = karplus_strong(root_freq, beat_duration * 3, sr)
                pos = int(beat * beat_duration * sr)
                if pos + len(note) <= total_samples:
                    bass_track[pos:pos + len(note)] += note * vel * 0.5
        elif section["name"] in ["chorus", "drop", "climax", "phase2"]:
            freq = root_freq if beat % 2 == 0 else root_freq * 2
            note = karplus_strong(freq, beat_duration * 0.9, sr)
            pos = int(beat * beat_duration * sr)
            if pos + len(note) <= total_samples:
                bass_track[pos:pos + len(note)] += note * vel * 0.8
        else:
            if beat % 2 == 0:
                note = karplus_strong(root_freq, beat_duration * 1.5, sr)
                pos = int(beat * beat_duration * sr)
                if pos + len(note) <= total_samples:
                    bass_track[pos:pos + len(note)] += note * vel * 0.7
    
    chord_duration = beat_duration * 4
    chord_vol = 0.15 + brightness * 0.2
    for i in range(int(duration / chord_duration)):
        time = i * chord_duration
        section = structure.get_section_at_time(time)
        energy = structure.get_energy_at_time(time) * intensity
        chord = progression[i % len(progression)]
        pos = int(i * chord_duration * sr)
        vel = (0.2 + 0.6 * energy) * chord_vol
        
        for nd in chord:
            freq = note_to_freq(scale[nd % len(scale)], base_freq)
            note = synth_pad(freq, chord_duration * 0.95, sr)
            if pos + len(note) <= total_samples:
                chords_track[pos:pos + len(note)] += note * vel * 0.8
    
    note_duration = beat_duration / 2
    current_degree = 0
    prev_section = None
    melody_prob = 0.4 + intensity * 0.4
    note_octaves = 1 + int(harmonic_complexity * 2)
    
    for i in range(int(duration / note_duration)):
        time = i * note_duration
        section = structure.get_section_at_time(time)
        sname = section["name"]
        energy = structure.get_energy_at_time(time) * intensity
        
        if section != prev_section:
            current_degree = np.random.randint(0, len(scale))
            prev_section = section
        
        if np.random.random() > melody_prob:
            continue
        
        step_range = 1 + int(harmonic_complexity * 3)
        step = np.random.choice(list(range(-step_range, step_range + 1)))
        max_deg = len(scale) * note_octaves
        current_degree = max(0, min(current_degree + step, max_deg - 1))
        
        octave = current_degree // len(scale)
        degree = current_degree % len(scale)
        freq = note_to_freq(scale[degree], base_freq) * (2 ** octave)
        
        pos = int(time * sr)
        nlen = int(note_duration * sr * 1.5)
        
        if brightness > 0.7:
            note = brass_note(freq, nlen / sr, sr)
        elif brightness > 0.4:
            note = piano_note(freq, nlen / sr, sr)
        else:
            note = flute_note(freq, nlen / sr, sr)
        
        if pos + len(note) <= total_samples:
            melody_track[pos:pos + len(note)] += note * energy * 0.3
    
    mixer = AutoMixer(sr)
    tracks = {
        "drums": drums_track,
        "bass": bass_track,
        "chords": chords_track,
        "melody": melody_track,
    }
    mix = mixer.mix_tracks(tracks)
    
    if dynamics_range < 0.5:
        mix = mixer.compress(mix, threshold=0.3, ratio=4.0)
    else:
        mix = mixer.compress(mix, threshold=0.5, ratio=2.5)
    
    mix = mixer.limit(mix, ceiling=0.92)
    
    fade_in = int(0.3 * sr); fade_out = int(1.0 * sr)
    if fade_in < len(mix): mix[:fade_in] *= np.linspace(0, 1, fade_in)
    if fade_out < len(mix): mix[-fade_out:] *= np.linspace(1, 0, fade_out)
    
    metadata = {
        "bpm": bpm, "style": style, "duration": duration, "seed": seed,
        "ai_guided": True,
        "ai_params": {
            "intensity": intensity, "brightness": brightness,
            "bass_weight": bass_weight, "rhythmic_density": rhythmic_density,
            "harmonic_complexity": harmonic_complexity, "dynamics_range": dynamics_range,
        },
        "validated_params": True,
    }
    
    return mix, sr, metadata


# ============================================================
# CLI
# ============================================================

def generate_random_params_for_style(style):
    """Gera parâmetros aleatórios dentro dos limites do estilo"""
    constraints = STYLE_CONSTRAINTS.get(style, STYLE_CONSTRAINTS["pop"])
    params = {}
    
    if "bpm_range" in constraints:
        lo, hi = constraints["bpm_range"]
        params["bpm"] = np.random.randint(lo, hi+1)
    
    param_names = ["intensity", "brightness", "bass_weight", "rhythmic_density", 
                   "harmonic_complexity", "dynamics_range"]
    
    for name in param_names:
        min_key = f"{name}_min"
        max_key = f"{name}_max"
        if min_key in constraints and max_key in constraints:
            params[name] = np.random.uniform(constraints[min_key], constraints[max_key])
        elif min_key in constraints:
            params[name] = np.random.uniform(constraints[min_key], 1.0)
        elif max_key in constraints:
            params[name] = np.random.uniform(0.0, constraints[max_key])
        else:
            params[name] = np.random.uniform(0.3, 0.8)
    
    return params


def cli():
    parser = argparse.ArgumentParser(description="IA Music Generator - With Style Validation")
    parser.add_argument("--duration", type=int, default=45)
    parser.add_argument("--style", type=str, default="pop",
                       choices=list(STYLE_CONSTRAINTS.keys()))
    parser.add_argument("--ai-guided", action="store_true", help="Use AI-guided parameters")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--list-styles", action="store_true", help="List all available styles")
    args = parser.parse_args()
    
    if args.list_styles:
        print("Estilos disponíveis:")
        for style in sorted(STYLE_CONSTRAINTS.keys()):
            print(f"  - {style}")
        return
    
    print("=" * 60)
    print("🎵 IA MUSIC - COM VALIDAÇÃO DE PARÂMETROS POR ESTILO")
    print("=" * 60)
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    style = args.style
    params = generate_random_params_for_style(style)
    print(f"\n🎯 Estilo: {style}")
    
    audio, sr, metadata = generate_with_ai(args.duration, params, style, 44100)
    
    feedback = FeedbackLoop()
    audio = feedback.mixer.fix_bad_mix(audio)
    
    filepath, number = save_song(audio, sr, metadata)
    print(f"\n✅ Música #{number}: {filepath}")
    
    # Mostrar análise de qualidade
    try:
        from quality_metrics import QualityMetrics
        qm = QualityMetrics(sr)
        metrics, score = qm.analyze_comprehensive(audio)
        print(f"\n📊 Qualidade: {score:.2f}/1.00")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"   {k}: {v:.3f}")
    except ImportError:
        pass


if __name__ == "__main__":
    cli()
