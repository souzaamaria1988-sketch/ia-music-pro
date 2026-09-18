#!/usr/bin/env python3
"""
🎵 IA MUSIC GENERATOR PRO - RAG + MoE
Retrieval-Augmented Generation + Mixture of Experts
"""
import os, sys, json
import numpy as np
from pathlib import Path

try:
    from scipy.signal import lfilter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

OUTPUT_DIR = "song_output"
MODEL_DIR = "models"
KNOWLEDGE_FILE = "knowledge_base.json"

# ============================================================
# RAG - Retrieval-Augmented Generation
# ============================================================

class MusicRAG:
    def __init__(self, knowledge_file=KNOWLEDGE_FILE):
        self.knowledge = []
        self.scales = {}
        self.progressions = {}
        if os.path.exists(knowledge_file):
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.knowledge = data.get('entries', [])
            self.scales = data.get('scales', {})
            self.progressions = data.get('chord_progressions', {})
            print(f"📚 RAG: {len(self.knowledge)} entradas carregadas")
        else:
            print("⚠️ RAG: knowledge_base.json não encontrado, usando defaults")
            self.scales = {'major': [0,2,4,5,7,9,11], 'minor': [0,2,3,5,7,8,10]}
            self.progressions = {'default': [[0,2,4],[5,0,2],[3,5,0],[4,6,1]]}
    
    def embed_text(self, text):
        """Embedding simples baseado em caracteres e palavras"""
        text_lower = text.lower()
        vocab = {}
        for entry in self.knowledge:
            for tag in entry['tags']:
                vocab[tag] = len(vocab)
        vec = np.zeros(max(len(vocab), 1))
        for tag, idx in vocab.items():
            if tag in text_lower:
                vec[idx] = 1.0
        # Normalizar
        norm = np.linalg.norm(vec)
        if norm > 0: vec = vec / norm
        return vec
    
    def retrieve(self, query, top_k=3):
        """Busca entradas relevantes na base de conhecimento"""
        if not self.knowledge:
            return []
        query_vec = self.embed_text(query)
        scores = []
        for entry in self.knowledge:
            entry_text = ' '.join(entry['tags']) + ' ' + entry.get('description', '')
            entry_vec = self.embed_text(entry_text)
            # Cosine similarity
            dot = np.dot(query_vec, entry_vec)
            norm = np.linalg.norm(query_vec) * np.linalg.norm(entry_vec) + 1e-10
            sim = dot / norm
            # Bonus por match direto de tags
            tag_bonus = sum(1 for tag in entry['tags'] if tag in query.lower()) * 0.3
            scores.append((sim + tag_bonus, entry))
        scores.sort(key=lambda x: x[0], reverse=True)
        results = [(score, entry) for score, entry in scores[:top_k]]
        print(f"  🔍 RAG recuperou {len(results)} entradas relevantes")
        for score, entry in results:
            print(f"     [{score:.2f}] {entry['tags'][:3]}... - {entry['description'][:50]}")
        return results
    
    def get_context(self, query):
        """Retorna contexto musical baseado no RAG"""
        results = self.retrieve(query, top_k=3)
        if not results:
            return {'scale': 'major', 'progression': [[0,2,4],[5,0,2],[3,5,0],[4,6,1]],
                    'bpm': 120, 'instruments': ['piano', 'strings']}
        # Combinar informações das top entradas
        best = results[0][1]
        scale_name = best.get('scale', 'major')
        scale = self.scales.get(scale_name, [0,2,4,5,7,9,11])
        prog_name = best.get('pattern', 'progression_epic')
        progression = self.progressions.get(prog_name, [[0,2,4],[5,0,2],[3,5,0],[4,6,1]])
        bpm_range = best.get('bpm_range', [100, 140])
        bpm = np.random.randint(bpm_range[0], bpm_range[1] + 1)
        instruments = best.get('instruments', ['piano', 'strings'])
        dynamics = best.get('dynamics', 'medium')
        intensity_map = {'very_quiet': 0.3, 'quiet': 0.5, 'medium': 0.7, 'loud': 0.85, 'extreme': 0.95, 'building': 0.8}
        intensity = intensity_map.get(dynamics, 0.7)
        print(f"  📖 RAG contexto: escala={scale_name}, BPM={bpm}, intensity={intensity}")
        return {'scale': scale, 'scale_name': scale_name, 'progression': progression,
                'bpm': bpm, 'instruments': instruments, 'intensity': intensity, 'dynamics': dynamics}

# ============================================================
# OUTPUT
# ============================================================

def get_next_song_number():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
    if not existing: return 1
    numbers = []
    for f in existing:
        try: numbers.append(int(f.replace('.wav', '')))
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
        with wave.open(filepath, 'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())
    if metadata:
        meta_path = os.path.join(OUTPUT_DIR, f"{number}.json")
        metadata['song_number'] = number
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"💾 Salvo: {filepath}")
    return filepath, number

# ============================================================
# INSTRUMENTOS
# ============================================================

def karplus_strong(freq, duration, sr=44100, damping=0.996, brightness=0.5):
    N = max(2, int(sr / freq)); n_samples = int(duration * sr)
    x = np.zeros(n_samples); x[:min(N, n_samples)] = np.random.uniform(-1, 1, min(N, n_samples))
    if HAS_SCIPY:
        a = np.zeros(N + 2); a[0] = 1.0; a[N] = -damping * brightness; a[N+1] = -damping * (1.0 - brightness)
        y = lfilter([1.0], a, x)
    else:
        y = np.zeros(n_samples); delay = np.random.uniform(-1, 1, N)
        for i in range(n_samples):
            y[i] = delay[i % N]; delay[i % N] = damping * 0.5 * (delay[i % N] + delay[(i+1) % N])
    return y / (np.max(np.abs(y)) + 1e-10)

def piano_note(freq, duration, sr=44100, velocity=1.0):
    t = np.linspace(0, duration, int(duration*sr), endpoint=False)
    signal = np.zeros_like(t)
    for h, amp, dec in zip([1,2,3,4,5,6,7,8], [1,.6,.35,.25,.18,.14,.11,.09], [2.5,2.2,2,1.8,1.6,1.4,1.3,1.2]):
        inharmonic = 1.0 + 0.00008 * (h**2)
        signal += amp * np.sin(2*np.pi*freq*h*inharmonic*t) * np.exp(-t*dec)
    atk = int(0.003*sr)
    if 0 < atk < len(signal): signal[:atk] *= np.linspace(0, 1, atk)
    signal += np.random.randn(len(signal)) * 0.03 * np.exp(-t*30)
    return signal * velocity / (np.max(np.abs(signal)) + 1e-10)

def violin_note(freq, duration, sr=44100):
    t = np.linspace(0, duration, int(duration*sr), endpoint=False)
    vibrato = 0.015 * np.sin(2*np.pi*5.5*t)
    phase = 2*np.pi*np.cumsum(freq*(1.0+vibrato))/sr
    signal = np.zeros_like(t)
    for h in range(1, 12): signal += np.sin(h*phase)/(h*1.2)
    envelope = np.ones_like(t)
    atk = int(min(0.08, duration*0.2)*sr); rel = int(min(0.05, duration*0.1)*sr)
    if 0 < atk < len(t): envelope[:atk] = np.linspace(0, 1, atk)**0.5
    if 0 < rel < len(t): envelope[-rel:] = np.linspace(1, 0, rel)
    return (signal*envelope*0.3)/(np.max(np.abs(signal))+1e-10)

def synth_pad(freq, duration, sr=44100):
    t = np.linspace(0, duration, int(duration*sr), endpoint=False)
    saw1 = 2*(t*freq%1)-1; saw2 = 2*(t*freq*1.003%1)-1
    signal = (saw1+saw2)/2
    signal = np.convolve(signal, np.ones(15)/15, mode='same')
    envelope = np.ones_like(t)
    atk = min(int(0.1*sr), len(t)//3); rel = min(int(0.2*sr), len(t)//3)
    if atk > 0: envelope[:atk] = np.linspace(0, 1, atk)
    if rel > 0: envelope[-rel:] = np.linspace(1, 0, rel)
    return signal*envelope*0.3/(np.max(np.abs(signal))+1e-10)

def flute_note(freq, duration, sr=44100):
    t = np.linspace(0, duration, int(duration*sr), endpoint=False)
    signal = np.sin(2*np.pi*freq*t)+0.3*np.sin(2*np.pi*freq*2*t)+0.1*np.sin(2*np.pi*freq*3*t)
    breath = np.random.randn(len(t))*0.05
    if len(breath)>10: breath = np.convolve(breath, np.ones(10)/10, mode='same')
    envelope = np.ones_like(t)
    atk = int(0.04*sr); rel = int(0.06*sr)
    if 0<atk<len(t): envelope[:atk]=np.linspace(0,1,atk)
    if 0<rel<len(t): envelope[-rel:]=np.linspace(1,0,rel)
    return (signal*envelope*0.3+breath*envelope)/(np.max(np.abs(signal))+1e-10)

def brass_note(freq, duration, sr=44100):
    t = np.linspace(0, duration, int(duration*sr), endpoint=False)
    phase = 2*np.pi*freq*t
    signal = np.zeros_like(t)
    for h in range(1, 16): signal += np.sin(h*phase)/h
    envelope = np.ones_like(t)
    atk = int(0.02*sr)
    if 0<atk<len(t):
        env = np.linspace(0, 1.3, atk); split = int(atk*0.7)
        if split < atk: env[split:] = np.linspace(1.3, 1.0, atk-split)
        envelope[:atk] = env
    rel = int(0.05*sr)
    if 0<rel<len(t): envelope[-rel:]=np.linspace(1,0,rel)
    return signal*envelope*0.25*1.5/(np.max(np.abs(signal))+1e-10)

def kick_drum(sr=44100):
    t = np.linspace(0, 0.35, int(0.35*sr), endpoint=False)
    freq_curve = 160*np.exp(-t*25)+45
    phase = 2*np.pi*np.cumsum(freq_curve)/sr
    signal = np.sin(phase)*np.exp(-t*12)
    return signal/(np.max(np.abs(signal))+1e-10)

def snare_drum(sr=44100):
    t = np.linspace(0, 0.22, int(0.22*sr), endpoint=False)
    tone = np.sin(2*np.pi*195*t)*np.exp(-t*35)+0.5*np.sin(2*np.pi*330*t)*np.exp(-t*40)
    noise = np.random.randn(len(t))*np.exp(-t*22)
    signal = 0.4*tone+0.6*noise
    return signal/(np.max(np.abs(signal))+1e-10)

def hihat(sr=44100):
    t = np.linspace(0, 0.06, int(0.06*sr), endpoint=False)
    noise = np.random.randn(len(t))
    filtered = np.diff(noise, prepend=noise[0])
    return filtered*np.exp(-t*45)/(np.max(np.abs(filtered))+1e-10)

def bitcrush(audio, bits=8, rate_div=4):
    if len(audio)==0: return audio
    reduced = audio[::rate_div]; upsampled = np.repeat(reduced, rate_div)
    if len(upsampled)<len(audio): upsampled = np.pad(upsampled, (0, len(audio)-len(upsampled)))
    elif len(upsampled)>len(audio): upsampled = upsampled[:len(audio)]
    return np.round(upsampled*(2**bits))/(2**bits)

def stutter(audio, sr=44100, size=0.03, repeats=4):
    chunk_size = max(1, int(size*sr))
    if len(audio)==0: return audio
    output = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        for _ in range(repeats): output.append(chunk)
    if not output: return audio
    result = np.concatenate(output)
    return result[:len(audio)] if len(result)>=len(audio) else np.pad(result, (0, len(audio)-len(result)))

def reese_bass(freq, duration, sr=44100, detune=0.03):
    t = np.linspace(0, duration, int(duration*sr), endpoint=False)
    saw1 = 2*(t*freq*(1-detune)%1)-1; saw2 = 2*(t*freq*(1+detune)%1)-1
    signal = (saw1+saw2)*0.5
    signal = np.convolve(signal, np.ones(8)/8, mode='same')
    envelope = np.ones_like(t)
    rel = min(int(0.05*sr), len(t))
    if rel>0: envelope[-rel:]=np.linspace(1,0,rel)
    return signal*envelope*0.6

def sub_808(freq, duration, sr=44100):
    t = np.linspace(0, duration, int(duration*sr), endpoint=False)
    freq_curve = freq*2*np.exp(-t*15)+freq
    phase = 2*np.pi*np.cumsum(freq_curve)/sr
    return np.tanh(np.sin(phase)*1.5)*np.exp(-t*4)

def amen_break(sr=44100, tempo_factor=1.0):
    beat_dur = 0.125/tempo_factor
    pattern = [('K',1),('H',.3),('S',.9),('H',.3),('G',.4),('H',.3),('S',.7),('H',.4),
               ('K',.9),('K',.5),('S',.9),('H',.3),('G',.5),('S',.6),('S',.8),('H',.3),
               ('K',1),('H',.3),('S',.9),('G',.4),('K',.8),('H',.4),('S',.8),('H',.3),
               ('K',1),('G',.5),('S',.9),('H',.4),('S',.7),('S',.6),('K',.8),('S',.9)]
    total_samples = int(beat_dur*len(pattern)*sr)
    output = np.zeros(total_samples)
    kick_s = kick_drum(sr)[:int(0.12*sr)]; snare_s = snare_drum(sr)[:int(0.1*sr)]
    ghost_s = snare_drum(sr)[:int(0.06*sr)]*0.4; hat_s = hihat(sr)[:int(0.04*sr)]
    for i, (hit, vel) in enumerate(pattern):
        pos = int(i*beat_dur*sr)
        if hit=='K': sample=kick_s
        elif hit=='S': sample=snare_s
        elif hit=='G': sample=ghost_s
        else: sample=hat_s
        jitter = np.random.randint(-int(0.002*sr), int(0.002*sr)+1)
        pos = max(0, pos+jitter); vel *= np.random.uniform(0.85, 1.1)
        end = min(pos+len(sample), total_samples)
        if pos<total_samples: output[pos:end] += sample[:end-pos]*vel
    return output/(np.max(np.abs(output))+1e-10)

def add_reverb(audio, sr=44100, decay=0.3, mix=0.25):
    delay = int(0.03*sr); reverb = np.zeros_like(audio)
    for d in [1,2,3,4,5,6]:
        pos = delay*d
        if pos<len(audio): reverb[pos:] += audio[:-pos]*(decay**d)
    return audio*(1-mix)+reverb*mix

def add_distortion(audio, gain=3.0, mix=0.7):
    return audio*(1-mix)+np.tanh(audio*gain)*mix

def soft_compress(audio, threshold=0.6, ratio=3.0):
    compressed = audio.copy()
    mask = np.abs(compressed)>threshold
    compressed[mask] = threshold+(compressed[mask]-threshold)/ratio
    return compressed

def note_to_freq(semitone, base_freq=261.63):
    return base_freq*(2**(semitone/12.0))

# ============================================================
# GERAÇÃO COM RAG + MoE
# ============================================================

def generate_breakcore(duration, sr=44100, intensity=1.0):
    total_samples = int(duration*sr); bpm = np.random.randint(180, 230)
    print(f"  💥 BREAKCORE: BPM={bpm}")
    break_source = amen_break(sr, tempo_factor=bpm/180)
    drums_track = np.zeros(total_samples); pos = 0
    while pos < total_samples:
        max_start = max(1, len(break_source)-int(0.15*sr))
        start = np.random.randint(0, max_start)
        length = np.random.choice([int(0.03*sr), int(0.06*sr), int(0.12*sr), int(0.25*sr)])
        chunk = break_source[start:start+length].copy()
        if len(chunk)==0: pos += int(0.05*sr); continue
        effect = np.random.choice(['none','stutter','reverse','crush','pitch_up','pitch_down'])
        if effect=='stutter': chunk = stutter(chunk, sr, 0.015, np.random.randint(2,8))
        elif effect=='reverse': chunk = chunk[::-1]
        elif effect=='crush': chunk = bitcrush(chunk, np.random.randint(4,10), np.random.randint(2,8))
        elif effect=='pitch_up':
            idx = np.round(np.arange(0, len(chunk), np.random.uniform(1.3,2.0))).astype(int)
            chunk = chunk[idx[idx<len(chunk)]]
        elif effect=='pitch_down':
            idx = np.round(np.arange(0, len(chunk), np.random.uniform(0.5,0.8))).astype(int)
            chunk = chunk[idx[idx<len(chunk)]]
        if np.random.random()<0.4: chunk = np.tanh(chunk*np.random.uniform(2,6))
        end = min(pos+len(chunk), total_samples)
        if pos<total_samples and len(chunk)>0:
            drums_track[pos:end] += chunk[:end-pos]*np.random.uniform(0.5,1.0)*intensity*0.8
        pos += len(chunk)
        if np.random.random()<0.24: pos += np.random.randint(int(0.01*sr), int(0.08*sr))
    bass_track = np.zeros(total_samples); beat_dur = 60.0/bpm
    for i in range(int(duration/beat_dur)):
        pos = int(i*beat_dur*sr)
        if np.random.random()<0.6:
            freq = np.random.choice([55.0, 58.27, 65.41, 73.42, 82.41])
            note_dur = beat_dur*np.random.choice([0.5, 1.0, 1.5])
            bass_note = reese_bass(freq, note_dur, sr) if np.random.random()<0.5 else sub_808(freq, note_dur, sr)
            end = min(pos+len(bass_note), total_samples)
            if pos<total_samples: bass_track[pos:end] += bass_note[:end-pos]*0.7
    glitch_track = np.zeros(total_samples)
    for _ in range(int(duration*3)):
        if total_samples<=int(0.1*sr): break
        pos = np.random.randint(0, max(1, total_samples-int(0.1*sr)))
        length = np.random.randint(int(0.005*sr), int(0.08*sr))
        t = np.arange(length)/sr
        gt = np.random.choice(['noise','tone','sweep'])
        if gt=='noise': glitch = np.random.randn(length)*np.exp(-np.arange(length)/(length/3))
        elif gt=='tone': glitch = np.sin(2*np.pi*np.random.uniform(500,6000)*t)*np.exp(-t*25)
        else:
            freqs = np.linspace(np.random.uniform(200,3000), np.random.uniform(200,3000), length)
            glitch = np.sin(2*np.pi*np.cumsum(freqs)/sr)
        glitch = bitcrush(glitch, np.random.randint(3,8), np.random.randint(2,10))
        end = min(pos+len(glitch), total_samples)
        if len(glitch)>0: glitch_track[pos:end] += glitch[:end-pos]*0.15
    mix = drums_track*0.85+bass_track*0.75+glitch_track*0.6
    mix = np.tanh(mix*2.0); mix = soft_compress(mix, 0.35, 5.0)
    mix = mix/(np.max(np.abs(mix))+1e-10)*0.95
    fi, fo = int(0.1*sr), int(0.3*sr)
    if fi<len(mix): mix[:fi] *= np.linspace(0,1,fi)
    if fo<len(mix): mix[-fo:] *= np.linspace(1,0,fo)
    return mix, sr

def generate_with_rag(duration, rag_context, sr=44100):
    """Gera música usando contexto do RAG"""
    total_samples = int(duration*sr)
    bpm = rag_context['bpm']
    beat_duration = 60.0/bpm
    scale = rag_context['scale']
    progression = rag_context['progression']
    intensity = rag_context['intensity']
    instruments = rag_context['instruments']
    base_freq = 220.0 if rag_context.get('scale_name') in ['minor', 'harmonic_minor'] else 261.63
    
    print(f"🎵 Gerando com RAG: BPM={bpm}, escala={rag_context.get('scale_name','major')}")
    print(f"   Instrumentos: {instruments}")
    
    drums = np.zeros(total_samples)
    bass_track = np.zeros(total_samples)
    chords_track = np.zeros(total_samples)
    melody_track = np.zeros(total_samples)
    
    # Drums baseado em instrumentos
    has_drums = any(d in str(instruments) for d in ['drums', 'percussion', 'breakbeat', 'timpani'])
    if has_drums:
        print("  🥁 Bateria...")
        kick, snare, hh = kick_drum(sr), snare_drum(sr), hihat(sr)
        for beat in range(int(duration/beat_duration)):
            pos = int(beat*beat_duration*sr)
            if beat%4 in [0,2] and pos+len(kick)<=total_samples: drums[pos:pos+len(kick)] += kick*0.8*intensity
            if beat%4 in [1,3] and pos+len(snare)<=total_samples: drums[pos:pos+len(snare)] += snare*0.7*intensity
            for sub in [0, 0.5]:
                sp = int((beat+sub)*beat_duration*sr)
                if sp+len(hh)<=total_samples: drums[sp:sp+len(hh)] += hh*(0.4 if sub==0 else 0.25)
    
    # Bass
    has_bass = any(b in str(instruments) for b in ['bass', 'reese'])
    if has_bass:
        print("  🎸 Baixo...")
        for beat in range(int(duration/beat_duration)):
            chord_idx = (beat//4)%len(progression)
            root = progression[chord_idx][0]
            if beat%2==0:
                freq = note_to_freq(scale[root%len(scale)], base_freq)/2
                note = karplus_strong(freq, beat_duration*1.5, sr, damping=0.998, brightness=0.3)
                pos = int(beat*beat_duration*sr)
                if pos+len(note)<=total_samples: bass_track[pos:pos+len(note)] += note*0.4
    
    # Chords baseado em instrumentos
    chord_duration = beat_duration*4
    has_strings = any(s in str(instruments) for s in ['strings', 'cello', 'choir'])
    has_piano = 'piano' in str(instruments)
    has_synth = any(s in str(instruments) for s in ['synth', 'pad'])
    has_guitar = any(g in str(instruments) for g in ['guitar', 'lute'])
    
    if has_piano:
        print("  🎹 Piano...")
        for i in range(int(duration/chord_duration)):
            chord = progression[i%len(progression)]
            pos = int(i*chord_duration*sr)
            for nd in chord:
                freq = note_to_freq(scale[nd%len(scale)], base_freq)
                note = piano_note(freq, chord_duration*0.9, sr)
                if pos+len(note)<=total_samples: chords_track[pos:pos+len(note)] += note*0.2
    
    if has_strings:
        print("  🎻 Cordas...")
        for i in range(int(duration/chord_duration)):
            chord = progression[i%len(progression)]
            pos = int(i*chord_duration*sr)
            for nd in chord:
                freq = note_to_freq(scale[nd%len(scale)], base_freq)
                note = violin_note(freq, chord_duration*0.95, sr)
                if pos+len(note)<=total_samples: chords_track[pos:pos+len(note)] += note*0.15
    
    if has_synth:
        print("  🎹 Synth...")
        for i in range(int(duration/chord_duration)):
            chord = progression[i%len(progression)]
            pos = int(i*chord_duration*sr)
            for nd in chord:
                freq = note_to_freq(scale[nd%len(scale)], base_freq)
                note = synth_pad(freq, chord_duration*0.95, sr)
                if pos+len(note)<=total_samples: chords_track[pos:pos+len(note)] += note*0.2
    
    if has_guitar:
        print("  🎸 Guitarra...")
        for i in range(int(duration/chord_duration)):
            chord = progression[i%len(progression)]
            pos = int(i*chord_duration*sr)
            for nd in chord:
                freq = note_to_freq(scale[nd%len(scale)], base_freq)
                note = karplus_strong(freq, chord_duration*0.8, sr, damping=0.995)
                if pos+len(note)<=total_samples: chords_track[pos:pos+len(note)] += note*0.25
    
    # Melody
    print("  🎶 Melodia...")
    note_duration = beat_duration/2
    np.random.seed(42)
    melody_notes = []
    current_degree = 0
    for i in range(int(duration/note_duration)):
        step = np.random.choice([-2,-1,0,1,2,3], p=[0.15,0.25,0.2,0.25,0.1,0.05])
        current_degree = max(0, min(current_degree+step, len(scale)*2-1))
        octave = current_degree//len(scale)
        degree = current_degree%len(scale)
        freq = note_to_freq(scale[degree], base_freq)*(2**octave)
        if np.random.random()>0.15: melody_notes.append((freq, i*note_duration))
    
    # Escolher instrumento da melodia
    melody_instrument = 'flute'
    if has_piano: melody_instrument = 'piano'
    elif has_strings: melody_instrument = 'violin'
    elif has_synth: melody_instrument = 'synth'
    elif has_guitar: melody_instrument = 'guitar'
    
    for freq, start_time in melody_notes:
        pos = int(start_time*sr); nlen = int(note_duration*sr*1.8)
        if melody_instrument=='piano': note = piano_note(freq, nlen/sr, sr)
        elif melody_instrument=='violin': note = violin_note(freq, nlen/sr, sr)
        elif melody_instrument=='synth': note = synth_pad(freq, nlen/sr, sr)
        elif melody_instrument=='guitar': note = karplus_strong(freq, nlen/sr, sr)
        else: note = flute_note(freq, nlen/sr, sr)
        if pos+len(note)<=total_samples: melody_track[pos:pos+len(note)] += note*0.35*intensity
    
    # Mix
    print("  🎛️ Mixagem...")
    mix = drums + bass_track + chords_track + melody_track
    mix = add_reverb(mix, sr, decay=0.35, mix=0.2)
    if has_strings: mix = add_reverb(mix, sr, decay=0.4, mix=0.15)  # mais reverb pra cordas
    mix = soft_compress(mix)
    mix = mix/(np.max(np.abs(mix))+1e-10)*0.9
    fade = int(0.5*sr)
    if fade<len(mix): mix[:fade]*=np.linspace(0,1,fade); mix[-fade:]*=np.linspace(1,0,fade)
    return mix, sr

# ============================================================
# MAIN
# ============================================================

def generate_music(prompt=None, style=None, duration=45, use_rag=True):
    """Função principal de geração"""
    # Inicializar RAG
    rag = MusicRAG() if use_rag else None
    
    # Determinar query para RAG
    query = prompt if prompt else style
    
    # Verificar se é breakcore
    if query and any(kw in query.lower() for kw in ['breakcore', 'amen', 'glitch', 'jungle']):
        print("💥 Modo BREAKCORE")
        return generate_breakcore(duration)
    
    if use_rag and rag:
        print("📚 Usando RAG para contexto musical...")
        context = rag.get_context(query)
        return generate_with_rag(duration, context)
    else:
        # Fallback sem RAG
        style_map = {
            "epic": {"intensity": 0.9, "bpm": 130, "scale": [0,2,4,5,7,9,11], "progression": [[0,2,4],[5,0,2],[3,5,0],[4,6,1]], "instruments": ["strings", "drums", "brass"]},
            "bossfight": {"intensity": 0.95, "bpm": 150, "scale": [0,2,3,5,7,8,10], "progression": [[0,2,4],[0,2,4],[5,0,2],[4,6,1]], "instruments": ["orchestra", "drums", "guitar"]},
            "dark": {"intensity": 0.7, "bpm": 75, "scale": [0,2,3,5,7,8,10], "progression": [[5,0,2],[3,5,0],[0,2,4],[4,6,1]], "instruments": ["strings", "piano", "pad"]},
            "rock": {"intensity": 0.95, "bpm": 140, "scale": [0,2,3,5,7,8,10], "progression": [[0,2,4],[0,2,4],[3,5,0],[4,6,1]], "instruments": ["guitar", "bass", "drums"]},
            "ambient": {"intensity": 0.4, "bpm": 55, "scale": [0,2,4,5,7,9,11], "progression": [[0,2,4],[3,5,0],[4,6,1],[0,2,4]], "instruments": ["pad", "flute", "bells"]},
            "electronic": {"intensity": 0.8, "bpm": 135, "scale": [0,2,3,5,7,8,10], "progression": [[0,2,4],[0,2,4],[5,0,2],[4,6,1]], "instruments": ["synth", "bass", "drums_machine"]},
            "jazz": {"intensity": 0.7, "bpm": 120, "scale": [0,2,3,5,7,9,10], "progression": [[1,3,5,0],[4,6,1,3],[0,2,4,6],[0,2,4,6]], "instruments": ["piano", "bass", "drums_brush", "sax"]},
            "classical": {"intensity": 0.7, "bpm": 90, "scale": [0,2,4,5,7,9,11], "progression": [[0,2,4],[3,5,0],[1,3,5],[4,6,1]], "instruments": ["strings", "piano", "woodwinds"]},
        }
        params = style_map.get(style, style_map["epic"])
        context = {'bpm': params['bpm'], 'scale': params['scale'], 'progression': params['progression'],
                   'intensity': params['intensity'], 'instruments': params['instruments'], 'scale_name': 'major'}
        return generate_with_rag(duration, context)

def batch_generate():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--style", type=str, default="epic")
    parser.add_argument("--duration", type=int, default=45)
    parser.add_argument("--use-rag", type=str, default="true")
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    
    use_rag = args.use_rag.lower() == "true"
    
    print("=" * 60)
    print("🎵 IA MUSIC GENERATOR PRO - RAG + MoE")
    print(f"   RAG: {'✅ Ativado' if use_rag else '❌ Desativado'}")
    print("=" * 60)
    
    # Verificar modelo MoE
    moe_path = os.path.join(MODEL_DIR, "best_moe_model.npz")
    if os.path.exists(moe_path):
        print("🧠 Modelo MoE encontrado!")
    else:
        print("⚠️ Modelo MoE não encontrado (gere sem modelo treinado)")
    
    prompt = args.prompt if args.prompt else None
    style = args.style if not prompt else None
    
    audio, sr = generate_music(prompt=prompt, style=style, duration=args.duration, use_rag=use_rag)
    
    metadata = {"prompt": args.prompt, "style": args.style, "duration": args.duration,
                "use_rag": use_rag, "model": "MoE+RAG"}
    filepath, number = save_song(audio, sr, metadata)
    print(f"\n✅ Música #{number} gerada: {filepath}")

def clear_screen():
    try:
        if os.name == 'nt': os.system('cls')
        elif os.environ.get('TERM'): os.system('clear')
    except: pass

def menu():
    clear_screen()
    print("=" * 50)
    print("🎵 IA MUSIC GENERATOR PRO - RAG + MoE")
    print(f"   Músicas em: {OUTPUT_DIR}/")
    print("=" * 50)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    songs = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
    print(f"   Geradas: {len(songs)}\n")
    print("1. 🎵 Gerar com PROMPT (RAG)")
    print("2. 🎼 Gerar com ESTILO (RAG)")
    print("3. 💥 Gerar BREAKCORE")
    print("4. 📁 Ver músicas")
    print("5. ❌ Sair")
    while True:
        choice = input("Opção: ").strip()
        if choice in ["1","2","3","4","5"]: return choice

def main():
    while True:
        choice = menu()
        if choice == "1":
            prompt = input("📝 Prompt: ").strip()
            if not prompt: continue
            duration = input("Duração (30/45/60/90): ").strip()
            if duration not in ["30","45","60","90"]: duration = "45"
            audio, sr = generate_music(prompt=prompt, duration=int(duration), use_rag=True)
            fp, num = save_song(audio, sr, {"prompt": prompt, "duration": int(duration), "use_rag": True})
            print(f"✅ Música #{num}: {fp}")
            input("ENTER...")
        elif choice == "2":
            print("1.Epico 2.Boss 3.Dark 4.Rock 5.Ambient 6.Eletronico 7.Jazz 8.Classico 9.Breakcore")
            s = input("Estilo: ").strip()
            styles = {"1":"epic","2":"bossfight","3":"dark","4":"rock","5":"ambient","6":"electronic","7":"jazz","8":"classical","9":"breakcore"}
            style = styles.get(s, "epic")
            duration = input("Duração (30/45/60/90): ").strip()
            if duration not in ["30","45","60","90"]: duration = "45"
            if style == "breakcore":
                audio, sr = generate_breakcore(int(duration))
            else:
                audio, sr = generate_music(style=style, duration=int(duration), use_rag=True)
            fp, num = save_song(audio, sr, {"style": style, "duration": int(duration), "use_rag": True})
            print(f"✅ Música #{num}: {fp}")
            input("ENTER...")
        elif choice == "3":
            duration = input("Duração (30/45/60/90): ").strip()
            if duration not in ["30","45","60","90"]: duration = "45"
            audio, sr = generate_breakcore(int(duration))
            fp, num = save_song(audio, sr, {"style": "breakcore", "duration": int(duration)})
            print(f"✅ Breakcore #{num}: {fp}")
            input("ENTER...")
        elif choice == "4":
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            songs = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')])
            if not songs: print("Nenhuma música.")
            else:
                for s in songs:
                    size = os.path.getsize(os.path.join(OUTPUT_DIR, s))/1024/1024
                    print(f"  🎵 {s} ({size:.1f}MB)")
            input("ENTER...")
        elif choice == "5":
            print("👋 Até mais!")
            break

if __name__ == "__main__":
    if "--batch" in sys.argv: batch_generate()
    else: main()
