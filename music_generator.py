#!/usr/bin/env python3
"""
🎵 IA MUSIC GENERATOR - REAL AI INTEGRATION
Now the trained autoencoder actually GUIDES generation.
"""
import os, sys, json, time, gc, argparse
import numpy as np
from pathlib import Path

try:
    from scipy.signal import lfilter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Imports
from music_intelligence import FeedbackLoop, AutoMixer, MusicMemory
from latent_controller import LatentController, StyleExtractor
from autoencoder import Autoencoder

OUTPUT_DIR = "song_output"
MODEL_DIR = "models"
MUSIC_DIR = "music_input"
AE_DIR = os.path.join(MODEL_DIR, "autoencoder")
SAMPLE_CACHE = {}


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
# AI BRAIN - loads trained autoencoder + controller
# ============================================================

class AIBrain:
    """The actual trained AI that guides generation."""
    
    def __init__(self):
        self.ae = None
        self.controller = None
        self.extractor = None
        self.is_loaded = False
        self._load()
    
    def _load(self):
        ae_path = os.path.join(AE_DIR, "best_autoencoder.npz")
        if not os.path.exists(ae_path):
            print("  ⚠️  Autoencoder não treinado - usando parâmetros aleatórios")
            print("     Rode: python train.py --epochs 10000")
            return
        
        try:
            self.ae = Autoencoder(input_size=256, hidden_sizes=[512, 256, 128], latent_size=64)
            self.ae.load(ae_path)
            self.controller = LatentController(self.ae)
            self.extractor = StyleExtractor(self.ae)
            self.is_loaded = True
            print("  🧠 IA carregada: autoencoder + controller prontos")
        except Exception as e:
            print(f"  ⚠️  Erro ao carregar IA: {e}")
    
    def get_params_from_audio(self, audio_path):
        """Extract musical parameters from a reference audio file."""
        if not self.is_loaded:
            return None
        try:
            latent, latent_std = self.extractor.extract_from_file(audio_path)
            params = self.controller.latent_to_params(latent)
            params["latent"] = latent
            params["latent_std"] = latent_std
            return params
        except Exception as e:
            print(f"  ⚠️  Erro ao extrair estilo: {e}")
            return None
    
    def get_params_from_random(self, seed=None):
        """Generate random params by sampling latent space near origin."""
        if not self.is_loaded:
            return None
        if seed is not None:
            np.random.seed(seed)
        # Sample from a normal distribution centered at 0
        latent = np.random.randn(64) * 0.5
        params = self.controller.latent_to_params(latent)
        params["latent"] = latent
        return params
    
    def vary_params(self, base_params, strength=0.15, seed=None):
        """Create a variation by perturbing the latent space."""
        if not self.is_loaded or "latent" not in base_params:
            return base_params
        varied_latent = self.controller.vary(base_params["latent"], strength, seed)
        params = self.controller.latent_to_params(varied_latent)
        params["latent"] = varied_latent
        return params
    
    def interpolate(self, params_a, params_b, t):
        """Interpolate between two sets of parameters via latent space."""
        if not self.is_loaded:
            return params_a
        lat_a = params_a.get("latent")
        lat_b = params_b.get("latent")
        if lat_a is None or lat_b is None:
            return params_a
        interp_latent = self.controller.interpolate(lat_a, lat_b, t)
        params = self.controller.latent_to_params(interp_latent)
        params["latent"] = interp_latent
        return params


# ============================================================
# INSTRUMENTS (unchanged - they're solid)
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
    }
    SECTION_ENERGY = {
        "intro": 0.3, "verse": 0.5, "chorus": 0.9, "bridge": 0.6, "outro": 0.4,
        "buildup": 0.7, "drop": 1.0, "breakdown": 0.2, "solo": 0.7, "head": 0.6,
        "theme": 0.5, "development": 0.7, "climax": 1.0, "resolution": 0.5,
        "section1": 0.4, "section2": 0.5, "section3": 0.6, "section4": 0.5,
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
            elif section in ["verse","head","theme"]: bars = np.random.choice([8, 12, 16])
            elif section in ["chorus","climax","drop"]: bars = np.random.choice([8, 12])
            elif section == "bridge": bars = np.random.choice([4, 8])
            elif section == "buildup": bars = np.random.choice([4, 8])
            elif section in ["breakdown"]: bars = np.random.choice([4, 8])
            elif section in ["solo","development"]: bars = np.random.choice([8, 12, 16])
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
# AI-GUIDED GENERATION (the new real AI part)
# ============================================================

def generate_with_ai(duration, ai_params, sr=44100):
    """
    Generate music where parameters come from the trained AI.
    This is the REAL integration: the autoencoder's latent space
    controls the synthesizer.
    """
    seed = get_dynamic_seed()
    np.random.seed(seed)
    
    # Extract AI-derived parameters
    bpm = int(ai_params.get("bpm", 120))
    intensity = float(ai_params.get("intensity", 0.7))
    brightness = float(ai_params.get("brightness", 0.5))
    bass_weight = float(ai_params.get("bass_weight", 0.5))
    rhythmic_density = float(ai_params.get("rhythmic_density", 0.5))
    harmonic_complexity = float(ai_params.get("harmonic_complexity", 0.5))
    dynamics_range = float(ai_params.get("dynamics_range", 0.7))
    style_index = int(ai_params.get("style_index", 0))
    
    # Map style_index to a style name
    styles = list(SongStructure.STRUCTURES.keys())
    style = styles[style_index % len(styles)]
    
    print(f"  🧠 Parâmetros da IA:")
    print(f"     BPM: {bpm}")
    print(f"     Intensidade: {intensity:.2f}")
    print(f"     Brilho: {brightness:.2f}")
    print(f"     Peso do grave: {bass_weight:.2f}")
    print(f"     Densidade rítmica: {rhythmic_density:.2f}")
    print(f"     Complexidade harmônica: {harmonic_complexity:.2f}")
    print(f"     Range dinâmico: {dynamics_range:.2f}")
    print(f"     Estilo inferido: {style}")
    
    # Base frequency scales with brightness
    freq_options = [196.0, 220.0, 261.63, 293.66, 349.23]
    base_freq = freq_options[min(int(brightness * len(freq_options)), len(freq_options) - 1)]
    
    # Scale choice influenced by harmonic complexity
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
    
    # SEPARATED TRACKS (for proper mixing)
    drums_track = np.zeros(total_samples)
    bass_track = np.zeros(total_samples)
    chords_track = np.zeros(total_samples)
    melody_track = np.zeros(total_samples)
    
    kick_sample = make_kick(sr); snare_sample = make_snare(sr)
    hihat_sample = make_hihat(sr)
    tom1_sample = make_tom(200, sr); tom2_sample = make_tom(150, sr); tom3_sample = make_tom(100, sr)
    crash_sample = make_crash(sr)
    n_beats = int(duration / beat_duration)
    
    # DRUMS - density controlled by AI
    drum_prob = 0.3 + rhythmic_density * 0.5
    for beat in range(n_beats):
        time = beat * beat_duration
        section = structure.get_section_at_time(time)
        sname = section["name"]
        energy = structure.get_energy_at_time(time) * intensity
        vel = 0.3 + 0.6 * energy
        pos = int(beat * beat_duration * sr)
        
        # Kick
        if sname in ["intro", "breakdown"]:
            kick_beats = [0]
        elif sname in ["chorus", "drop", "climax"]:
            kick_beats = [0, 2]
        else:
            kick_beats = [0, 2] if np.random.random() < drum_prob else [0]
        
        for kb in kick_beats:
            kp = pos + int(kb * beat_duration * sr / 4 * 2)
            if kp + len(kick_sample) <= total_samples:
                drums_track[kp:kp + len(kick_sample)] += kick_sample * vel * 0.8
        
        # Snare on 2 and 4 (backbeat)
        if sname not in ["intro", "breakdown"]:
            for sb in [1, 3]:
                sp = pos + int(sb * beat_duration * sr / 4 * 2)
                if sp + len(snare_sample) <= total_samples:
                    drums_track[sp:sp + len(snare_sample)] += snare_sample * vel * 0.7
        
        # Hi-hats - density from AI
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
    
    # BASS - weight controlled by AI
    bass_vol = 0.2 + bass_weight * 0.4
    for beat in range(n_beats):
        time = beat * beat_duration
        section = structure.get_section_at_time(time)
        energy = structure.get_energy_at_time(time) * intensity
        chord_idx = (beat // 4) % len(progression)
        root = progression[chord_idx][0]
        root_freq = note_to_freq(scale[root % len(scale)], base_freq) / 2
        vel = (0.3 + 0.7 * energy) * bass_vol
        
        if section["name"] in ["intro", "breakdown"]:
            if beat % 4 == 0:
                note = karplus_strong(root_freq, beat_duration * 3, sr)
                pos = int(beat * beat_duration * sr)
                if pos + len(note) <= total_samples:
                    bass_track[pos:pos + len(note)] += note * vel * 0.5
        elif section["name"] in ["chorus", "drop", "climax"]:
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
    
    # CHORDS - brightness controlled by AI
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
    
    # MELODY - complexity and intensity controlled by AI
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
        
        # Instrument choice based on brightness
        if brightness > 0.7:
            note = brass_note(freq, nlen / sr, sr)
        elif brightness > 0.4:
            note = piano_note(freq, nlen / sr, sr)
        else:
            note = flute_note(freq, nlen / sr, sr)
        
        if pos + len(note) <= total_samples:
            melody_track[pos:pos + len(note)] += note * energy * 0.3
    
    # AUTO-MIXING with proper volume balance
    mixer = AutoMixer(sr)
    tracks = {
        "drums": drums_track,
        "bass": bass_track,
        "chords": chords_track,
        "melody": melody_track,
    }
    mix = mixer.mix_tracks(tracks)
    
    # Apply dynamics range from AI
    if dynamics_range < 0.5:
        # Compress more for small dynamic range
        mix = mixer.compress(mix, threshold=0.3, ratio=4.0)
    else:
        mix = mixer.compress(mix, threshold=0.5, ratio=2.5)
    
    mix = mixer.limit(mix, ceiling=0.92)
    
    # Fade
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
            "style_index": style_index,
        },
    }
    
    return mix, sr, metadata


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def cli():
    parser = argparse.ArgumentParser(description="IA Music Generator - AI-Guided")
    parser.add_argument("--duration", type=int, default=45)
    parser.add_argument("--style-from", type=str, default=None,
                        help="Path to audio file to extract style from")
    parser.add_argument("--interpolate", nargs=2, default=None,
                        help="Two audio files to interpolate between")
    parser.add_argument("--interpolate-t", type=float, default=0.5,
                        help="Interpolation point (0.0-1.0)")
    parser.add_argument("--variation-of", type=int, default=None,
                        help="Song number to create a variation of")
    parser.add_argument("--variation-strength", type=float, default=0.15,
                        help="How much to vary (0.0-1.0)")
    parser.add_argument("--ai-guided", action="store_true",
                        help="Use random AI parameters")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--style", type=str, default="pop")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎵 IA MUSIC - AI-GUIDED GENERATION")
    print("=" * 60)
    
    brain = AIBrain()
    
    # Determine where parameters come from
    if args.style_from:
        print(f"\n🎯 Extraindo estilo de: {args.style_from}")
        params = brain.get_params_from_audio(args.style_from)
        if params is None:
            print("  ⚠️  Falha, usando parâmetros aleatórios")
            params = brain.get_params_from_random(args.seed)
    elif args.interpolate:
        print(f"\n🎯 Interpolando entre: {args.interpolate[0]} e {args.interpolate[1]}")
        p_a = brain.get_params_from_audio(args.interpolate[0])
        p_b = brain.get_params_from_audio(args.interpolate[1])
        if p_a and p_b:
            params = brain.interpolate(p_a, p_b, args.interpolate_t)
        else:
            params = brain.get_params_from_random(args.seed)
    elif args.variation_of is not None:
        src_path = os.path.join(OUTPUT_DIR, f"{args.variation_of}.wav")
        print(f"\n🎯 Criando variação da música #{args.variation_of}")
        base_params = brain.get_params_from_audio(src_path)
        if base_params:
            params = brain.vary_params(base_params, args.variation_strength, args.seed)
        else:
            params = brain.get_params_from_random(args.seed)
    elif args.ai_guided or not brain.is_loaded:
        print("\n🎯 Parâmetros aleatórios do espaço latente")
        params = brain.get_params_from_random(args.seed)
    else:
        # Default: AI-guided
        params = brain.get_params_from_random(args.seed)
    
    if params is None:
        # Fallback if AI not loaded
        print("  ⚠️  IA não disponível, gerando proceduralmente")
        from music_generator import generate_with_automix  # legacy
        audio, sr, metadata = generate_with_automix(args.duration, 120, args.style, 44100)
    else:
        audio, sr, metadata = generate_with_ai(args.duration, params, 44100)
    
    # Post-processing via FeedbackLoop (keeps existing quality gate)
    feedback = FeedbackLoop()
    audio = feedback.mixer.fix_bad_mix(audio)
    
    filepath, number = save_song(audio, sr, metadata)
    print(f"\n✅ Música #{number}: {filepath}")


if __name__ == "__main__":
    cli()
