#!/usr/bin/env python3
"""
🎵 IA MUSIC GENERATOR - COM +100 INSTRUMENTOS
Integração completa com a biblioteca de instrumentos
"""
import os, sys, json, time, gc, argparse
import numpy as np
from pathlib import Path

try:
    from scipy.signal import lfilter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Importar biblioteca de instrumentos
from instruments import INSTRUMENTS, get_instrument, list_instruments, get_instruments_by_family
from music_intelligence import FeedbackLoop, AutoMixer, MusicMemory

OUTPUT_DIR = "song_output"
MODEL_DIR = "models"
MUSIC_DIR = "music_input"
SAMPLE_CACHE = {}


# ============================================================
# ESTILOS COM INSTRUMENTOS PADRÃO
# ============================================================

STYLE_INSTRUMENTS = {
    "cinematic": {
        "melody": ["synth_strings", "violin", "french_horn"],
        "chords": ["synth_pad", "piano", "synth_choir"],
        "bass": ["synth_bass", "cello", "sub_bass"],
        "percussion": ["timpani", "crash", "kick", "snare"],
    },
    "epic": {
        "melody": ["synth_brass", "trumpet", "french_horn"],
        "chords": ["synth_strings", "synth_choir", "piano"],
        "bass": ["synth_bass", "sub_bass", "cello"],
        "percussion": ["timpani", "crash", "kick", "snare", "tom"],
    },
    "ambient": {
        "melody": ["synth_pad", "flute", "synth_bell"],
        "chords": ["synth_pad", "synth_strings"],
        "bass": ["sub_bass", "synth_bass"],
        "percussion": [],
    },
    "electronic": {
        "melody": ["synth_lead", "synth_arp"],
        "chords": ["synth_pad", "synth_organ"],
        "bass": ["synth_bass", "sub_bass", "wobble_bass"],
        "percussion": ["kick", "snare", "hihat_closed", "hihat_open", "crash"],
    },
    "rock": {
        "melody": ["electric_guitar"],
        "chords": ["electric_guitar", "organ"],
        "bass": ["electric_bass"],
        "percussion": ["kick", "snare", "hihat_closed", "crash", "tom"],
    },
    "jazz": {
        "melody": ["piano", "soprano_sax", "trumpet"],
        "chords": ["piano", "organ"],
        "bass": ["electric_bass", "double_bass"],
        "percussion": ["kick", "snare", "ride", "hihat_closed"],
    },
    "classical": {
        "melody": ["violin", "flute", "cello"],
        "chords": ["synth_strings", "piano", "harp"],
        "bass": ["cello", "double_bass"],
        "percussion": ["timpani"],
    },
    "samba": {
        "melody": ["cavaquinho", "acoustic_guitar"],
        "chords": ["acoustic_guitar", "cavaquinho"],
        "bass": ["surdo", "electric_bass"],
        "percussion": ["pandeiro", "tamborim", "surdo", "agogo", "chocalho", "reco_reco"],
    },
    "bossa": {
        "melody": ["acoustic_guitar", "piano"],
        "chords": ["acoustic_guitar", "piano"],
        "bass": ["electric_bass"],
        "percussion": ["pandeiro", "shaker"],
    },
    "forro": {
        "melody": ["accordion", "triangle"],
        "chords": ["accordion"],
        "bass": ["surdo", "electric_bass"],
        "percussion": ["triangle", "chocalho", "surdo"],
    },
    "breakcore": {
        "melody": ["synth_lead", "synth_fx"],
        "chords": ["synth_pad", "synth_noise"],
        "bass": ["wobble_bass", "sub_bass", "fm_bass"],
        "percussion": ["kick", "snare", "hihat_closed", "crash", "tom"],
    },
    "pop": {
        "melody": ["synth_lead", "piano"],
        "chords": ["synth_pad", "piano", "synth_strings"],
        "bass": ["synth_bass", "electric_bass"],
        "percussion": ["kick", "snare", "hihat_closed", "hihat_open"],
    },
    "dark": {
        "melody": ["synth_strings", "cello"],
        "chords": ["synth_pad", "organ"],
        "bass": ["sub_bass", "synth_bass"],
        "percussion": ["kick", "tom", "timpani"],
    },
    "indian": {
        "melody": ["sitar", "erhu", "shakuhachi"],
        "chords": ["synth_pad", "harp"],
        "bass": ["sub_bass"],
        "percussion": ["tabla", "djembe", "tamborim"],
    },
    "african": {
        "melody": ["kalimba", "mbira", "balafon"],
        "chords": ["synth_pad"],
        "bass": ["djembe", "sub_bass"],
        "percussion": ["djembe", "conga", "bongo_high", "bongo_low", "maracas", "claves"],
    },
}


def get_style_instruments(style):
    """Retorna instrumentos padrão para um estilo"""
    return STYLE_INSTRUMENTS.get(style, STYLE_INSTRUMENTS["pop"])


def get_instrument_func(name, sr=44100):
    """Retorna função de instrumento com sr fixo"""
    func = get_instrument(name)
    if func is None:
        return None
    
    # Verificar assinatura da função
    import inspect
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    
    # Retornar wrapper com sr fixo
    if 'sr' in params:
        return lambda freq=None, duration=1.0, **kwargs: func(freq or 440, duration, sr=sr, **kwargs) if freq else func(sr=sr, **kwargs)
    else:
        return lambda freq=None, duration=1.0, **kwargs: func(freq or 440, duration, **kwargs) if freq else func(**kwargs)


# ============================================================
# FUNÇÕES DE GERAÇÃO (mantidas do original)
# ============================================================

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
        "samba": ["intro","verse","chorus","verse","chorus","break","chorus","outro"],
        "bossa": ["intro","theme","theme2","outro"],
        "forro": ["intro","verse","chorus","verse","chorus","outro"],
        "breakcore": ["intro","chaos1","break","chaos2","break","chaos3","outro"],
        "dark": ["intro","verse","chorus","verse","chorus","bridge","chorus","outro"],
        "indian": ["alap","jor","jhala","outro"],
        "african": ["intro","theme","variation","theme","outro"],
    }
    SECTION_ENERGY = {
        "intro": 0.3, "verse": 0.5, "chorus": 0.9, "bridge": 0.6, "outro": 0.4,
        "buildup": 0.7, "drop": 1.0, "breakdown": 0.2, "solo": 0.7, "head": 0.6,
        "theme": 0.5, "theme2": 0.6, "development": 0.7, "climax": 1.0, "resolution": 0.5,
        "exposition": 0.5, "recapitulation": 0.7, "coda": 0.4,
        "chaos1": 0.8, "chaos2": 0.9, "chaos3": 1.0, "break": 0.3,
        "section1": 0.4, "section2": 0.5, "section3": 0.6, "section4": 0.5,
        "alap": 0.3, "jor": 0.5, "jhala": 0.8, "variation": 0.7,
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
            elif section in ["verse","head","theme","exposition","alap","jor"]: bars = np.random.choice([8, 12, 16])
            elif section in ["chorus","climax","drop","recapitulation","jhala"]: bars = np.random.choice([8, 12])
            elif section == "bridge" or section == "break": bars = np.random.choice([4, 8])
            elif section == "buildup": bars = np.random.choice([4, 8])
            elif section in ["breakdown","coda","outro"]: bars = np.random.choice([4, 8])
            elif section in ["solo","development","chaos1","chaos2","chaos3","variation"]: bars = np.random.choice([8, 12, 16])
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
# GERAÇÃO COM +100 INSTRUMENTOS
# ============================================================

def generate_with_instruments(duration, style="pop", bpm=120, custom_instruments=None, sr=44100):
    """
    Gera música usando a biblioteca de 100+ instrumentos
    
    Args:
        duration: duração em segundos
        style: estilo musical
        bpm: batidas por minuto
        custom_instruments: dict com listas de instrumentos personalizados
                           {"melody": [...], "chords": [...], "bass": [...], "percussion": [...]}
        sr: sample rate
    """
    seed = get_dynamic_seed()
    np.random.seed(seed)
    
    # Obter instrumentos do estilo ou personalizados
    if custom_instruments:
        style_insts = custom_instruments
    else:
        style_insts = get_style_instruments(style)
    
    print(f"  🎸 Instrumentos usados:")
    for role, instruments in style_insts.items():
        if instruments:
            print(f"     {role}: {', '.join(instruments)}")
    
    # Configuração
    base_freq = np.random.choice([196.0, 220.0, 261.63, 293.66, 349.23])
    scale = ALL_SCALES[np.random.choice(list(ALL_SCALES.keys()))]
    progression = ALL_PROGRESSIONS[np.random.randint(0, len(ALL_PROGRESSIONS))]
    structure = SongStructure(style=style, duration=duration, bpm=bpm)
    
    print(f"   🎼 Estrutura: {[s['name'] for s in structure.sections]}")
    
    total_samples = int(duration * sr)
    beat_duration = 60.0 / bpm
    
    # Tracks separadas
    melody_track = np.zeros(total_samples)
    chords_track = np.zeros(total_samples)
    bass_track = np.zeros(total_samples)
    percussion_track = np.zeros(total_samples)
    
    n_beats = int(duration / beat_duration)
    
    # === MELODIA ===
    melody_instruments = style_insts.get("melody", ["piano"])
    if melody_instruments:
        print("  🎵 Gerando melodia...")
        current_degree = 0
        note_duration = beat_duration / 2
        
        for i in range(int(duration / note_duration)):
            time = i * note_duration
            section = structure.get_section_at_time(time)
            energy = structure.get_energy_at_time(time)
            
            if np.random.random() > 0.6:
                continue
            
            step = np.random.choice(list(range(-3, 4)))
            max_deg = len(scale) * 2
            current_degree = max(0, min(current_degree + step, max_deg - 1))
            
            octave = current_degree // len(scale)
            degree = current_degree % len(scale)
            freq = note_to_freq(scale[degree], base_freq) * (2 ** octave)
            
            pos = int(time * sr)
            nlen = note_duration * 1.5
            
            # Escolher instrumento aleatório da lista
            inst_name = np.random.choice(melody_instruments)
            inst_func = get_instrument_func(inst_name, sr)
            
            if inst_func:
                try:
                    note = inst_func(freq, nlen)
                    if pos + len(note) <= total_samples:
                        melody_track[pos:pos + len(note)] += note * energy * 0.3
                except Exception as e:
                    print(f"     ⚠️ Erro em {inst_name}: {e}")
    
    # === ACORDES ===
    chord_instruments = style_insts.get("chords", ["synth_pad"])
    if chord_instruments:
        print("  🎹 Gerando acordes...")
        chord_duration = beat_duration * 4
        
        for i in range(int(duration / chord_duration)):
            time = i * chord_duration
            section = structure.get_section_at_time(time)
            energy = structure.get_energy_at_time(time)
            chord = progression[i % len(progression)]
            pos = int(i * chord_duration * sr)
            
            inst_name = np.random.choice(chord_instruments)
            inst_func = get_instrument_func(inst_name, sr)
            
            if inst_func:
                try:
                    for nd in chord:
                        freq = note_to_freq(scale[nd % len(scale)], base_freq)
                        note = inst_func(freq, chord_duration * 0.95)
                        if pos + len(note) <= total_samples:
                            chords_track[pos:pos + len(note)] += note * energy * 0.2
                except Exception as e:
                    print(f"     ⚠️ Erro em {inst_name}: {e}")
    
    # === BAIXO ===
    bass_instruments = style_insts.get("bass", ["synth_bass"])
    if bass_instruments:
        print("  🎸 Gerando baixo...")
        for beat in range(n_beats):
            time = beat * beat_duration
            section = structure.get_section_at_time(time)
            energy = structure.get_energy_at_time(time)
            chord_idx = (beat // 4) % len(progression)
            root = progression[chord_idx][0]
            root_freq = note_to_freq(scale[root % len(scale)], base_freq) / 2
            
            if beat % 2 == 0:
                pos = int(beat * beat_duration * sr)
                inst_name = np.random.choice(bass_instruments)
                inst_func = get_instrument_func(inst_name, sr)
                
                if inst_func:
                    try:
                        note = inst_func(root_freq, beat_duration * 1.5)
                        if pos + len(note) <= total_samples:
                            bass_track[pos:pos + len(note)] += note * energy * 0.4
                    except Exception as e:
                        print(f"     ⚠️ Erro em {inst_name}: {e}")
    
    # === PERCUSSÃO ===
    perc_instruments = style_insts.get("percussion", ["kick", "snare", "hihat_closed"])
    if perc_instruments:
        print("  🥁 Gerando percussão...")
        for beat in range(n_beats):
            time = beat * beat_duration
            section = structure.get_section_at_time(time)
            sname = section["name"]
            energy = structure.get_energy_at_time(time)
            pos = int(beat * beat_duration * sr)
            
            # Kick em beats 0 e 2
            if sname not in ["intro", "breakdown"] and beat % 2 == 0:
                if "kick" in perc_instruments or "surdo" in perc_instruments:
                    inst_name = "kick" if "kick" in perc_instruments else "surdo"
                    inst_func = get_instrument_func(inst_name, sr)
                    if inst_func:
                        try:
                            hit = inst_func()
                            if pos + len(hit) <= total_samples:
                                percussion_track[pos:pos + len(hit)] += hit * energy * 0.8
                        except Exception as e:
                            print(f"     ⚠️ Erro em {inst_name}: {e}")
            
            # Snare/caixa em beats 1 e 3
            if sname not in ["intro", "breakdown"] and beat % 4 in [1, 3]:
                snare_insts = [i for i in perc_instruments if i in ["snare", "pandeiro", "tamborim", "djembe", "conga"]]
                if snare_insts:
                    inst_name = np.random.choice(snare_insts)
                    inst_func = get_instrument_func(inst_name, sr)
                    if inst_func:
                        try:
                            hit = inst_func()
                            if pos + len(hit) <= total_samples:
                                percussion_track[pos:pos + len(hit)] += hit * energy * 0.7
                        except Exception as e:
                            print(f"     ⚠️ Erro em {inst_name}: {e}")
            
            # Hi-hat/chocalho em subdivisões
            hihat_insts = [i for i in perc_instruments if i in ["hihat_closed", "hihat_open", "chocalho", "maracas", "shaker"]]
            if hihat_insts and sname != "intro":
                inst_name = np.random.choice(hihat_insts)
                inst_func = get_instrument_func(inst_name, sr)
                if inst_func:
                    for sub in [0, 0.5]:
                        hp = pos + int(sub * beat_duration * sr)
                        try:
                            hit = inst_func()
                            if hp + len(hit) <= total_samples:
                                vol = 0.35 if sub == 0 else 0.2
                                percussion_track[hp:hp + len(hit)] += hit * energy * vol
                        except Exception as e:
                            print(f"     ⚠️ Erro em {inst_name}: {e}")
            
            # Percussão extra (agogo, triangle, etc)
            extra_insts = [i for i in perc_instruments if i in ["agogo", "triangle", "claves", "cowbell", "guiro", "woodblock"]]
            if extra_insts and np.random.random() < 0.3:
                inst_name = np.random.choice(extra_insts)
                inst_func = get_instrument_func(inst_name, sr)
                if inst_func:
                    try:
                        hit = inst_func()
                        if pos + len(hit) <= total_samples:
                            percussion_track[pos:pos + len(hit)] += hit * energy * 0.3
                    except Exception as e:
                        print(f"     ⚠️ Erro em {inst_name}: {e}")
    
    # === MIXAGEM ===
    print("  🎛️ Mixando...")
    mixer = AutoMixer(sr)
    tracks = {
        "melody": melody_track,
        "chords": chords_track,
        "bass": bass_track,
        "percussion": percussion_track,
    }
    
    # Volumes por papel
    volumes = {
        "melody": 0.30,
        "chords": 0.20,
        "bass": 0.25,
        "percussion": 0.25,
    }
    
    mix = mixer.mix_tracks(tracks, volumes)
    mix = mixer.compress(mix, threshold=0.4, ratio=3.0)
    mix = mixer.limit(mix, ceiling=0.92)
    
    # Fade in/out
    fade_in = int(0.3 * sr); fade_out = int(1.0 * sr)
    if fade_in < len(mix): mix[:fade_in] *= np.linspace(0, 1, fade_in)
    if fade_out < len(mix): mix[-fade_out:] *= np.linspace(1, 0, fade_out)
    
    metadata = {
        "bpm": bpm, "style": style, "duration": duration, "seed": seed,
        "instruments_used": style_insts,
        "total_instruments_available": len(INSTRUMENTS),
    }
    
    return mix, sr, metadata


# ============================================================
# CLI
# ============================================================

def cli():
    parser = argparse.ArgumentParser(description="IA Music Generator - 100+ Instruments")
    parser.add_argument("--duration", type=int, default=45)
    parser.add_argument("--style", type=str, default="pop")
    parser.add_argument("--bpm", type=int, default=120)
    parser.add_argument("--instruments", type=str, default=None,
                       help="Lista de instrumentos separados por vírgula (ex: 'piano,violin,cello')")
    parser.add_argument("--list-instruments", action="store_true", help="Lista todos os instrumentos")
    parser.add_argument("--list-families", action="store_true", help="Lista famílias de instrumentos")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    if args.list_instruments:
        print("🎸 Instrumentos disponíveis:")
        for inst in sorted(list_instruments()):
            print(f"  - {inst}")
        print(f"\nTotal: {len(list_instruments())} instrumentos")
        return
    
    if args.list_families:
        print("🎸 Famílias de instrumentos:")
        for family in ["strings", "keys", "winds", "percussion", "electronic", "ethnic"]:
            instruments = get_instruments_by_family(family)
            print(f"\n{family} ({len(instruments)}):")
            for inst in instruments:
                print(f"  - {inst}")
        return
    
    print("=" * 60)
    print("🎵 IA MUSIC - COM +100 INSTRUMENTOS")
    print("=" * 60)
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Parse instrumentos personalizados
    custom_instruments = None
    if args.instruments:
        inst_list = [i.strip() for i in args.instruments.split(",")]
        custom_instruments = {
            "melody": inst_list[:2],
            "chords": inst_list[2:4] if len(inst_list) > 2 else inst_list[:1],
            "bass": inst_list[4:5] if len(inst_list) > 4 else ["synth_bass"],
            "percussion": inst_list[5:] if len(inst_list) > 5 else ["kick", "snare"],
        }
        print(f"\n🎸 Instrumentos personalizados: {inst_list}")
    
    audio, sr, metadata = generate_with_instruments(
        duration=args.duration,
        style=args.style,
        bpm=args.bpm,
        custom_instruments=custom_instruments,
        sr=44100
    )
    
    # Feedback loop
    feedback = FeedbackLoop()
    audio = feedback.mixer.fix_bad_mix(audio)
    
    filepath, number = save_song(audio, sr, metadata)
    print(f"\n✅ Música #{number}: {filepath}")


if __name__ == "__main__":
    cli()
