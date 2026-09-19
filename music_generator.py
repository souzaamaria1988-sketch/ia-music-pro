#!/usr/bin/env python3
"""
🧠 IA MUSIC GENERATOR PRO - INTELIGÊNCIA MÁXIMA
- Modelo MoE treinado INFLUENCIA a geração
- Análise REAL das músicas de referência
- RAG + Modelo + Análise combinados
- Prompt interpreter com 80+ keywords
- Composição inteligente com teoria musical
"""
import os,sys,json,time
import numpy as np
from pathlib import Path

try:
    from scipy.signal import lfilter, stft
    from scipy.fft import fft, fftfreq
    HAS_SCIPY=True
except ImportError:
    HAS_SCIPY=False

OUTPUT_DIR="song_output"
MODEL_DIR="models"
MUSIC_DIR="music_input"

def get_dynamic_seed():
    return int(time.time()*1000)%(2**32)^np.random.randint(0,2**31)

def get_next_song_number():
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    existing=[f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
    if not existing:return 1
    numbers=[]
    for f in existing:
        try:numbers.append(int(f.replace('.wav','')))
        except:pass
    return max(numbers,default=0)+1

def save_song(audio,sr,metadata=None):
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    number=get_next_song_number()
    filepath=os.path.join(OUTPUT_DIR,f"{number}.wav")
    try:
        import soundfile as sf;sf.write(filepath,audio,sr)
    except ImportError:
        import wave;audio_int16=(audio*32767).astype(np.int16)
        with wave.open(filepath,'w') as wf:
            wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())
    if metadata:
        meta_path=os.path.join(OUTPUT_DIR,f"{number}.json")
        metadata['song_number']=number
        with open(meta_path,'w',encoding='utf-8') as f:json.dump(metadata,f,indent=2,ensure_ascii=False)
    print(f"💾 Salvo: {filepath}")
    return filepath,number

# ============================================================
# ANÁLISE REAL DE ÁUDIO (INTELIGÊNCIA)
# ============================================================

class AudioAnalyzer:
    """Analisa músicas de referência para extrair padrões reais"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.reference_features = None
    
    def analyze_file(self, filepath, sr=22050):
        """Analisa um arquivo de áudio extraindo features completas"""
        try:
            import soundfile as sf
            audio, file_sr = sf.read(filepath, dtype='float32')
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            # Resample
            if file_sr != sr:
                indices = np.round(np.arange(0, len(audio), file_sr/sr)).astype(int)
                audio = audio[indices[indices < len(audio)]]
        except Exception as e:
            print(f"  ⚠️ Erro ao analisar {filepath}: {e}")
            return None
        
        if len(audio) < sr:  # mínimo 1 segundo
            return None
        
        features = {}
        
        # 1. ANÁLISE TEMPORAL
        features['duration'] = len(audio) / sr
        features['rms'] = float(np.sqrt(np.mean(audio**2)))
        features['peak'] = float(np.max(np.abs(audio)))
        features['dynamic_range'] = float(features['peak'] - np.min(np.abs(audio)))
        
        # 2. ANÁLISE ESPECTRAL (FFT)
        fft_result = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        
        # Centroide espectral (brilho)
        total_energy = np.sum(fft_result) + 1e-10
        features['spectral_centroid'] = float(np.sum(freqs * fft_result) / total_energy)
        
        # Largura de banda espectral
        features['spectral_bandwidth'] = float(np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * fft_result) / total_energy))
        
        # Rolloff (85% da energia)
        cumulative = np.cumsum(fft_result)
        rolloff_idx = np.searchsorted(cumulative, 0.85 * total_energy)
        features['spectral_rolloff'] = float(freqs[min(rolloff_idx, len(freqs)-1)])
        
        # 3. BANDAS DE FREQUÊNCIA
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mids': (250, 500),
            'mids': (500, 2000),
            'high_mids': (2000, 6000),
            'highs': (6000, 20000)
        }
        features['frequency_bands'] = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            energy = float(np.sum(fft_result[mask]**2))
            features['frequency_bands'][band_name] = energy
        
        # 4. DETECÇÃO DE BPM (autocorrelação)
        # Usar apenas primeiros 10 segundos
        analysis_audio = audio[:sr*10] if len(audio) > sr*10 else audio
        autocorr = np.correlate(analysis_audio, analysis_audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Procurar picos na autocorrelação
        peaks = []
        min_lag = int(sr * 60 / 200)  # BPM máximo 200
        max_lag = int(sr * 60 / 40)   # BPM mínimo 40
        
        for i in range(min_lag, min(max_lag, len(autocorr)-1)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > 0.1 * autocorr[0]:
                    peaks.append(i)
                    if len(peaks) >= 5:
                        break
        
        if len(peaks) >= 2:
            avg_period = np.mean(np.diff(peaks))
            bpm = 60.0 / (avg_period / sr)
            features['estimated_bpm'] = float(np.clip(bpm, 40, 200))
        else:
            features['estimated_bpm'] = 120.0
        
        # 5. DETECÇÃO DE TONALIDADE (simplificada)
        # Procurar frequência dominante
        dominant_freq = freqs[np.argmax(fft_result)]
        features['dominant_frequency'] = float(dominant_freq)
        
        # Converter para nota musical
        if dominant_freq > 0:
            midi_note = 69 + 12 * np.log2(dominant_freq / 440)
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_idx = int(np.round(midi_note)) % 12
            octave = int(np.round(midi_note)) // 12 - 1
            features['estimated_key'] = f"{note_names[note_idx]}{octave}"
            
            # Determinar se é maior ou menor (simplificado)
            # Baseado na presença de terça maior ou menor
            third_major = dominant_freq * (2 ** (4/12))
            third_minor = dominant_freq * (2 ** (3/12))
            
            # Verificar energia perto dessas frequências
            major_idx = np.argmin(np.abs(freqs - third_major))
            minor_idx = np.argmin(np.abs(freqs - third_minor))
            
            if fft_result[major_idx] > fft_result[minor_idx]:
                features['mode'] = 'major'
            else:
                features['mode'] = 'minor'
        else:
            features['estimated_key'] = 'C4'
            features['mode'] = 'major'
        
        # 6. ZERO CROSSING RATE (textura)
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        features['zero_crossing_rate'] = float(zcr)
        
        # 7. MFCC-LIKE (13 coeficientes)
        n_mfcc = 13
        log_freqs = np.logspace(np.log10(20), np.log10(sr/2), n_mfcc + 1)
        mfcc_like = []
        for i in range(n_mfcc):
            mask = (freqs >= log_freqs[i]) & (freqs < log_freqs[i+1])
            energy = float(np.sum(fft_result[mask]**2))
            mfcc_like.append(np.log(energy + 1e-10))
        features['mfcc_like'] = mfcc_like
        
        # 8. ENERGIA POR SEGUNDO (para estrutura)
        segment_length = sr  # 1 segundo
        n_segments = len(audio) // segment_length
        energy_per_second = []
        for i in range(min(n_segments, 30)):
            segment = audio[i*segment_length:(i+1)*segment_length]
            energy_per_second.append(float(np.sqrt(np.mean(segment**2))))
        features['energy_curve'] = energy_per_second
        
        return features
    
    def analyze_directory(self, music_dir=MUSIC_DIR):
        """Analisa todas as músicas em music_input/"""
        music_path = Path(music_dir)
        if not music_path.exists():
            print("  📁 music_input/ não existe")
            return None
        
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.flac', '*.ogg']:
            audio_files.extend(list(music_path.glob(ext)))
        
        if not audio_files:
            print("  📁 Nenhuma música em music_input/")
            return None
        
        print(f"  🔍 Analisando {len(audio_files)} músicas de referência...")
        
        all_features = []
        for filepath in audio_files[:20]:  # máximo 20 arquivos
            print(f"    🎵 {filepath.name}")
            features = self.analyze_file(filepath)
            if features:
                all_features.append(features)
        
        if not all_features:
            return None
        
        # Calcular estatísticas agregadas
        aggregated = {
            'num_files': len(all_features),
            'avg_bpm': np.mean([f['estimated_bpm'] for f in all_features]),
            'avg_spectral_centroid': np.mean([f['spectral_centroid'] for f in all_features]),
            'avg_spectral_bandwidth': np.mean([f['spectral_bandwidth'] for f in all_features]),
            'avg_rms': np.mean([f['rms'] for f in all_features]),
            'avg_duration': np.mean([f['duration'] for f in all_features]),
            'key_distribution': {},
            'mode_distribution': {'major': 0, 'minor': 0},
            'avg_bands': {},
        }
        
        # Distribuição de tonalidades
        for f in all_features:
            key = f['estimated_key']
            aggregated['key_distribution'][key] = aggregated['key_distribution'].get(key, 0) + 1
            mode = f['mode']
            aggregated['mode_distribution'][mode] += 1
        
        # Médias das bandas
        band_names = ['sub_bass', 'bass', 'low_mids', 'mids', 'high_mids', 'highs']
        for band in band_names:
            values = [f['frequency_bands'].get(band, 0) for f in all_features]
            aggregated['avg_bands'][band] = float(np.mean(values))
        
        # Curva de energia média
        all_energy_curves = [f['energy_curve'] for f in all_features if f['energy_curve']]
        if all_energy_curves:
            max_len = max(len(c) for c in all_energy_curves)
            padded = [c + [0]*(max_len-len(c)) for c in all_energy_curves]
            aggregated['avg_energy_curve'] = np.mean(padded, axis=0).tolist()
        
        self.reference_features = aggregated
        
        print(f"  ✅ Análise completa:")
        print(f"     BPM médio: {aggregated['avg_bpm']:.1f}")
        print(f"     Tonalidade mais comum: {max(aggregated['key_distribution'].items(), key=lambda x: x[1])[0] if aggregated['key_distribution'] else 'N/A'}")
        print(f"     Modo: {'maior' if aggregated['mode_distribution']['major'] > aggregated['mode_distribution']['minor'] else 'menor'}")
        print(f"     Brilho espectral: {aggregated['avg_spectral_centroid']:.0f} Hz")
        
        return aggregated
    
    def get_style_from_analysis(self):
        """Infere estilo musical baseado na análise"""
        if not self.reference_features:
            return 'pop'
        
        bpm = self.reference_features['avg_bpm']
        centroid = self.reference_features['avg_spectral_centroid']
        bands = self.reference_features['avg_bands']
        
        # Calcular proporções de bandas
        total_band_energy = sum(bands.values()) + 1e-10
        bass_ratio = (bands.get('sub_bass', 0) + bands.get('bass', 0)) / total_band_energy
        high_ratio = (bands.get('high_mids', 0) + bands.get('highs', 0)) / total_band_energy
        
        # Inferir estilo
        if bpm > 160:
            return 'breakcore'
        elif bpm > 130 and high_ratio > 0.3:
            return 'electronic'
        elif bpm > 120 and bass_ratio > 0.4:
            return 'rock'
        elif bpm < 80 and centroid < 1000:
            return 'ambient'
        elif 90 < bpm < 140 and centroid > 1500:
            return 'jazz'
        elif bpm < 100:
            return 'classical'
        else:
            return 'pop'

# ============================================================
# CARREGADOR DE MODELO MoE (INTELIGÊNCIA TREINADA)
# ============================================================

class MoELoader:
    """Carrega e usa o modelo MoE treinado"""
    
    def __init__(self):
        self.model_data = None
        self.is_loaded = False
        self.normalization = None
    
    def load_model(self, model_path=None):
        """Carrega modelo MoE treinado"""
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "best_moe_model.npz")
        
        if not os.path.exists(model_path):
            print("  ⚠️ Modelo MoE não encontrado (treine primeiro)")
            return False
        
        try:
            self.model_data = np.load(model_path, allow_pickle=True)
            self.is_loaded = True
            
            # Carregar normalização
            norm_path = os.path.join(MODEL_DIR, "normalization.npz")
            if os.path.exists(norm_path):
                self.normalization = np.load(norm_path)
            
            # Informações do modelo
            num_experts = int(self.model_data.get('num_experts', 8))
            input_size = int(self.model_data.get('input_size', 128))
            
            print(f"  🧠 Modelo MoE carregado:")
            print(f"     Experts: {num_experts}")
            print(f"     Input size: {input_size}")
            
            # Mostrar nomes dos experts se disponível
            if 'expert_names' in self.model_data:
                expert_names = self.model_data['expert_names']
                print(f"     Nomes: {list(expert_names)}")
            
            return True
        except Exception as e:
            print(f"  ❌ Erro ao carregar modelo: {e}")
            return False
    
    def get_expert_weights(self, expert_idx):
        """Retorna pesos de um expert específico"""
        if not self.is_loaded:
            return None
        
        prefix = f'expert_{expert_idx}_'
        weights = {}
        
        for key in self.model_data.files:
            if key.startswith(prefix):
                weights[key[len(prefix):]] = self.model_data[key]
        
        return weights
    
    def get_gate_weights(self):
        """Retorna pesos do gate network"""
        if not self.is_loaded:
            return None
        return {
            'w': self.model_data.get('gate_w'),
            'b': self.model_data.get('gate_b')
        }
    
    def predict_style_affinity(self, features_vector):
        """Usa o gate do modelo para prever afinidade com estilos"""
        if not self.is_loaded:
            return None
        
        gate = self.get_gate_weights()
        if gate['w'] is None:
            return None
        
        # Normalizar features se necessário
        if self.normalization is not None:
            mean = self.normalization.get('mean')
            std = self.normalization.get('std')
            if mean is not None and std is not None:
                features_vector = (features_vector - mean) / (std + 1e-8)
        
        # Ajustar tamanho se necessário
        input_size = gate['w'].shape[0]
        if len(features_vector) < input_size:
            features_vector = np.pad(features_vector, (0, input_size - len(features_vector)))
        elif len(features_vector) > input_size:
            features_vector = features_vector[:input_size]
        
        # Forward pass no gate
        logits = features_vector @ gate['w'] + gate['b']
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return probs
    
    def get_style_parameters(self, style_probs):
        """Converte probabilidades de estilo em parâmetros musicais"""
        if style_probs is None:
            return None
        
        # Mapear experts para estilos
        expert_styles = ['epic', 'dark', 'electronic', 'jazz', 'breakcore', 
                        'ambient', 'rock', 'classical', 'folk', 'latin', 
                        'cinematic', 'experimental']
        
        # Pegar top-2 experts
        top_indices = np.argsort(style_probs)[-2:]
        
        # Parâmetros base por estilo
        style_params = {
            'epic': {'bpm': 130, 'intensity': 0.9, 'scale': 'major', 'instruments': ['strings', 'brass', 'drums']},
            'dark': {'bpm': 75, 'intensity': 0.7, 'scale': 'harmonic_minor', 'instruments': ['strings', 'piano', 'pad']},
            'electronic': {'bpm': 135, 'intensity': 0.8, 'scale': 'minor', 'instruments': ['synth', 'bass', 'drums_machine']},
            'jazz': {'bpm': 120, 'intensity': 0.7, 'scale': 'dorian', 'instruments': ['piano', 'bass', 'drums_brush']},
            'breakcore': {'bpm': 200, 'intensity': 0.95, 'scale': 'minor', 'instruments': ['breakbeat', 'bass_heavy', 'glitch']},
            'ambient': {'bpm': 55, 'intensity': 0.4, 'scale': 'lydian', 'instruments': ['pad', 'flute', 'bells']},
            'rock': {'bpm': 140, 'intensity': 0.95, 'scale': 'phrygian', 'instruments': ['guitar', 'bass', 'drums_heavy']},
            'classical': {'bpm': 90, 'intensity': 0.7, 'scale': 'major', 'instruments': ['strings', 'piano', 'woodwinds']},
            'folk': {'bpm': 110, 'intensity': 0.6, 'scale': 'mixolydian', 'instruments': ['guitar_acoustic', 'flute', 'drums_light']},
            'latin': {'bpm': 100, 'intensity': 0.7, 'scale': 'major', 'instruments': ['guitar', 'percussion', 'bass']},
            'cinematic': {'bpm': 100, 'intensity': 0.8, 'scale': 'harmonic_minor', 'instruments': ['orchestra', 'percussion', 'choir']},
            'experimental': {'bpm': 120, 'intensity': 0.7, 'scale': 'whole_tone', 'instruments': ['synth', 'glitch', 'texture']},
        }
        
        # Combinar parâmetros dos top-2 experts
        combined = {}
        for idx in top_indices:
            if idx < len(expert_styles):
                style = expert_styles[idx]
                params = style_params.get(style, style_params['epic'])
                weight = style_probs[idx]
                
                for key, value in params.items():
                    if key not in combined:
                        combined[key] = value
                    else:
                        # Combinar valores numéricos
                        if isinstance(value, (int, float)) and isinstance(combined[key], (int, float)):
                            combined[key] = combined[key] * (1-weight) + value * weight
        
        return combined

# ============================================================
# PROMPT INTERPRETER AVANÇADO (80+ KEYWORDS)
# ============================================================

class PromptInterpreter:
    """Interpreta prompts em português com NLP avançado"""
    
    KEYWORDS = {
        # Emoções
        "intenso": {"intensity": 0.95, "bpm_mult": 1.3},
        "calmo": {"intensity": 0.4, "bpm_mult": 0.6},
        "épico": {"intensity": 0.9, "bpm_mult": 1.2, "orchestral": 2.0, "style": "cinematic"},
        "epico": {"intensity": 0.9, "bpm_mult": 1.2, "orchestral": 2.0, "style": "cinematic"},
        "sombrio": {"intensity": 0.7, "minor": True, "style": "dark"},
        "dark": {"intensity": 0.7, "minor": True, "style": "dark"},
        "feliz": {"major": True, "bpm_mult": 1.1},
        "triste": {"minor": True, "bpm_mult": 0.7},
        "alegre": {"major": True, "bpm_mult": 1.2},
        "melancólico": {"minor": True, "bpm_mult": 0.8},
        "melancolico": {"minor": True, "bpm_mult": 0.8},
        "energético": {"intensity": 0.95, "bpm_mult": 1.4},
        "energetico": {"intensity": 0.95, "bpm_mult": 1.4},
        "romântico": {"minor": True, "bpm_mult": 0.7},
        "romantico": {"minor": True, "bpm_mult": 0.7},
        "misterioso": {"intensity": 0.5, "minor": True},
        "heroico": {"intensity": 0.9, "bpm_mult": 1.15, "orchestral": 2.0},
        "heróico": {"intensity": 0.9, "bpm_mult": 1.15, "orchestral": 2.0},
        "agressivo": {"intensity": 0.95, "distortion": 2.0, "bpm_mult": 1.3},
        "suave": {"intensity": 0.4, "bpm_mult": 0.7},
        "pesado": {"intensity": 1.1, "drums": 1.5, "bass": 1.8},
        "leve": {"intensity": 0.4, "drums": 0.5},
        "tenso": {"intensity": 0.8, "minor": True, "bpm_mult": 1.2},
        "relaxante": {"intensity": 0.3, "bpm_mult": 0.5, "style": "ambient"},
        "animado": {"intensity": 0.8, "bpm_mult": 1.3},
        "depressivo": {"minor": True, "bpm_mult": 0.5, "intensity": 0.4},
        "esperançoso": {"major": True, "bpm_mult": 1.1},
        "esperancoso": {"major": True, "bpm_mult": 1.1},
        
        # Instrumentos
        "guitarra": {"guitar": 2.0, "distortion": 1.5},
        "violão": {"guitar": 2.0, "distortion": 1.0},
        "violao": {"guitar": 2.0, "distortion": 1.0},
        "piano": {"piano": 2.5},
        "bateria": {"drums": 2.5},
        "tambores": {"drums": 2.5},
        "violino": {"strings": 2.5, "orchestral": 1.8},
        "sintetizador": {"synth": 2.0},
        "synth": {"synth": 2.0},
        "baixo": {"bass": 2.5},
        "flauta": {"flute": 2.0},
        "trompete": {"brass": 2.0},
        "saxofone": {"sax": 2.0},
        "coral": {"choir": 2.5, "orchestral": 2.0},
        "orquestra": {"orchestral": 3.0, "strings": 2.5},
        "cordas": {"strings": 2.5},
        "teclado": {"piano": 2.0, "synth": 1.5},
        "harpa": {"harp": 2.0},
        "órgão": {"organ": 2.0},
        "orgao": {"organ": 2.0},
        
        # Estilos
        "rock": {"guitar": 2.0, "drums": 2.0, "distortion": 1.5, "style": "rock"},
        "metal": {"guitar": 2.5, "drums": 2.5, "distortion": 2.0, "style": "rock"},
        "jazz": {"jazz": 2.5, "piano": 2.0, "style": "jazz"},
        "eletrônica": {"synth": 2.5, "style": "electronic"},
        "eletronica": {"synth": 2.5, "style": "electronic"},
        "clássica": {"orchestral": 2.5, "strings": 2.0, "piano": 2.0, "style": "classical"},
        "classica": {"orchestral": 2.5, "strings": 2.0, "piano": 2.0, "style": "classical"},
        "pop": {"style": "pop"},
        "funk": {"bass": 2.0, "drums": 1.8, "style": "latin"},
        "samba": {"drums": 2.0, "bpm_mult": 1.1, "style": "latin"},
        "bossa": {"bpm_mult": 0.7, "guitar": 1.5, "style": "latin"},
        "reggae": {"bpm_mult": 0.7, "bass": 1.8, "style": "latin"},
        "hiphop": {"bass": 2.0, "drums": 2.0, "bpm_mult": 0.85},
        "rap": {"bass": 2.0, "drums": 2.0, "bpm_mult": 0.85},
        "techno": {"synth": 2.5, "bpm_mult": 1.4, "style": "electronic"},
        "house": {"synth": 2.0, "bpm_mult": 1.2, "style": "electronic"},
        "trance": {"synth": 2.5, "bpm_mult": 1.3, "style": "electronic"},
        "dubstep": {"bass": 2.5, "bpm_mult": 0.7, "style": "electronic"},
        "lofi": {"intensity": 0.4, "bpm_mult": 0.6, "style": "ambient"},
        "ambient": {"intensity": 0.3, "bpm_mult": 0.5, "style": "ambient"},
        "new age": {"intensity": 0.4, "style": "ambient"},
        
        # Contextos
        "batalha": {"intensity": 0.95, "drums": 2.0, "bpm_mult": 1.3},
        "guerra": {"intensity": 0.95, "drums": 2.5, "bpm_mult": 1.35, "orchestral": 2.0},
        "boss": {"intensity": 0.95, "bpm_mult": 1.3},
        "medieval": {"orchestral": 1.8, "strings": 1.5, "style": "cinematic"},
        "fantasia": {"orchestral": 2.0, "style": "cinematic"},
        "espaço": {"synth": 1.5, "intensity": 0.5, "style": "ambient"},
        "espaco": {"synth": 1.5, "intensity": 0.5, "style": "ambient"},
        "cidade": {"intensity": 0.7, "style": "pop"},
        "natureza": {"intensity": 0.4, "style": "ambient"},
        "chuva": {"intensity": 0.3, "style": "ambient"},
        "noite": {"minor": True, "intensity": 0.5},
        "dia": {"major": True, "intensity": 0.7},
        "amanhecer": {"major": True, "intensity": 0.5, "bpm_mult": 0.8},
        "entardecer": {"minor": True, "intensity": 0.6, "bpm_mult": 0.7},
        "festa": {"intensity": 0.9, "bpm_mult": 1.3, "drums": 2.0},
        "dançar": {"bpm_mult": 1.2, "drums": 1.8},
        "dancar": {"bpm_mult": 1.2, "drums": 1.8},
        "estudar": {"intensity": 0.3, "bpm_mult": 0.6, "style": "ambient"},
        "trabalhar": {"intensity": 0.4, "bpm_mult": 0.7},
        "dormir": {"intensity": 0.2, "bpm_mult": 0.4, "style": "ambient"},
        "correr": {"bpm_mult": 1.4, "drums": 2.0},
        "treinar": {"bpm_mult": 1.3, "drums": 1.8},
        "academia": {"bpm_mult": 1.3, "drums": 1.8},
        "viagem": {"intensity": 0.6, "style": "ambient"},
        "praia": {"intensity": 0.5, "style": "latin"},
        "montanha": {"intensity": 0.6, "orchestral": 1.5},
        "floresta": {"intensity": 0.4, "style": "ambient"},
        "deserto": {"intensity": 0.5, "minor": True},
        "oceano": {"intensity": 0.4, "style": "ambient"},
        "tempestade": {"intensity": 0.9, "drums": 2.0, "minor": True},
        
        # Velocidade
        "rápido": {"bpm_mult": 1.4},
        "rapido": {"bpm_mult": 1.4},
        "lento": {"bpm_mult": 0.6},
        "devagar": {"bpm_mult": 0.6},
        "muito rápido": {"bpm_mult": 1.6},
        "muito rapido": {"bpm_mult": 1.6},
        "muito lento": {"bpm_mult": 0.4},
        "muito lento": {"bpm_mult": 0.4},
        
        # Breakcore
        "breakcore": {"breakcore": 3.0, "intensity": 1.0, "bpm_mult": 1.8, "style": "breakcore"},
        "amen": {"breakcore": 2.5, "style": "breakcore"},
        "glitch": {"breakcore": 2.0, "style": "breakcore"},
        "jungle": {"breakcore": 2.0, "bpm_mult": 1.5, "style": "breakcore"},
        "drum and bass": {"breakcore": 1.5, "bpm_mult": 1.4, "style": "breakcore"},
        "dnb": {"breakcore": 1.5, "bpm_mult": 1.4, "style": "breakcore"},
        "caótico": {"breakcore": 2.0, "style": "breakcore"},
        "caotico": {"breakcore": 2.0, "style": "breakcore"},
        "destruído": {"breakcore": 2.0, "distortion": 2.0},
        "destruido": {"breakcore": 2.0, "distortion": 2.0},
        "frenético": {"breakcore": 2.0, "bpm_mult": 1.5},
        "frenetico": {"breakcore": 2.0, "bpm_mult": 1.5},
    }
    
    def interpret(self, prompt):
        """Interpreta prompt e retorna parâmetros"""
        if not prompt:
            return self._default_params()
        
        prompt_lower = prompt.lower()
        params = self._default_params()
        matched = []
        
        # Verificar keywords compostas primeiro
        compound_keywords = ["drum and bass", "new age", "muito rápido", "muito rapido", "muito lento"]
        for kw in compound_keywords:
            if kw in prompt_lower:
                if kw in self.KEYWORDS:
                    effects = self.KEYWORDS[kw]
                    matched.append(kw)
                    self._apply_effects(params, effects)
        
        # Verificar keywords simples
        for keyword, effects in self.KEYWORDS.items():
            if keyword in prompt_lower and keyword not in compound_keywords:
                matched.append(keyword)
                self._apply_effects(params, effects)
        
        # Detectar números (BPM específico)
        import re
        bpm_match = re.search(r'(d+)s*bpm', prompt_lower)
        if bpm_match:
            params['target_bpm'] = int(bpm_match.group(1))
            matched.append(f"{bpm_match.group(1)} bpm")
        
        # Detectar duração
        duration_match = re.search(r'(d+)s*(segundos|seg|s)', prompt_lower)
        if duration_match:
            params['target_duration'] = int(duration_match.group(1))
            matched.append(f"{duration_match.group(1)}s")
        
        print(f"  🧠 Prompt interpretado: {len(matched)} keywords")
        if matched:
            print(f"     Detectadas: {matched[:10]}{'...' if len(matched) > 10 else ''}")
        
        return params
    
    def _default_params(self):
        return {
            "intensity": 0.7, "bpm_mult": 1.0, "orchestral": 1.0,
            "guitar": 1.0, "piano": 1.0, "drums": 1.0, "strings": 1.0,
            "synth": 1.0, "distortion": 1.0, "bass": 1.0,
            "flute": 1.0, "brass": 1.0, "jazz": 1.0, "choir": 1.0,
            "sax": 1.0, "harp": 1.0, "organ": 1.0,
            "minor": False, "major": True, "breakcore": 0.0,
            "style": None, "target_bpm": None, "target_duration": None
        }
    
    def _apply_effects(self, params, effects):
        for key, value in effects.items():
            if isinstance(value, bool):
                params[key] = value
            elif isinstance(value, (int, float)):
                if key in params and isinstance(params[key], (int, float)):
                    params[key] = min(params[key] * value, 3.0)
                else:
                    params[key] = value
            elif isinstance(value, str):
                params[key] = value

# ============================================================
# INSTRUMENTOS (mesmos de antes)
# ============================================================

def karplus_strong(freq,duration,sr=44100,damping=0.996,brightness=0.5):
    N=max(2,int(sr/freq));n_samples=int(duration*sr)
    x=np.zeros(n_samples);x[:min(N,n_samples)]=np.random.uniform(-1,1,min(N,n_samples))
    if HAS_SCIPY:
        a=np.zeros(N+2);a[0]=1.0;a[N]=-damping*brightness;a[N+1]=-damping*(1.0-brightness)
        y=lfilter([1.0],a,x)
    else:
        y=np.zeros(n_samples);delay=np.random.uniform(-1,1,N)
        for i in range(n_samples):
            y[i]=delay[i%N];delay[i%N]=damping*0.5*(delay[i%N]+delay[(i+1)%N])
    return y/(np.max(np.abs(y))+1e-10)

def piano_note(freq,duration,sr=44100,velocity=1.0):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    signal=np.zeros_like(t)
    for h,amp,dec in zip([1,2,3,4,5,6,7,8],[1,.6,.35,.25,.18,.14,.11,.09],[2.5,2.2,2,1.8,1.6,1.4,1.3,1.2]):
        inharmonic=1.0+0.00008*(h**2)
        signal+=amp*np.sin(2*np.pi*freq*h*inharmonic*t)*np.exp(-t*dec)
    atk=int(0.003*sr)
    if 0<atk<len(signal):signal[:atk]*=np.linspace(0,1,atk)
    signal+=np.random.randn(len(signal))*0.03*np.exp(-t*30)
    return signal*velocity/(np.max(np.abs(signal))+1e-10)

def violin_note(freq,duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    vibrato=0.015*np.sin(2*np.pi*5.5*t)
    phase=2*np.pi*np.cumsum(freq*(1.0+vibrato))/sr
    signal=np.zeros_like(t)
    for h in range(1,12):signal+=np.sin(h*phase)/(h*1.2)
    envelope=np.ones_like(t)
    atk=int(min(0.08,duration*0.2)*sr);rel=int(min(0.05,duration*0.1)*sr)
    if 0<atk<len(t):envelope[:atk]=np.linspace(0,1,atk)**0.5
    if 0<rel<len(t):envelope[-rel:]=np.linspace(1,0,rel)
    bow_noise=np.random.randn(len(t))*0.01*envelope
    return (signal*envelope*0.3+bow_noise)/(np.max(np.abs(signal))+1e-10)

def synth_pad(freq,duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    saw1=2*(t*freq%1)-1;saw2=2*(t*freq*1.003%1)-1;saw3=2*(t*freq*0.997%1)-1
    signal=(saw1+saw2+saw3)/3
    signal=np.convolve(signal,np.ones(15)/15,mode='same')
    envelope=np.ones_like(t)
    atk=min(int(0.1*sr),len(t)//3);rel=min(int(0.2*sr),len(t)//3)
    if atk>0:envelope[:atk]=np.linspace(0,1,atk)
    if rel>0:envelope[-rel:]=np.linspace(1,0,rel)
    return signal*envelope*0.3/(np.max(np.abs(signal))+1e-10)

def flute_note(freq,duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    signal=np.sin(2*np.pi*freq*t)+0.3*np.sin(2*np.pi*freq*2*t)+0.1*np.sin(2*np.pi*freq*3*t)
    breath=np.random.randn(len(t))*0.05
    if len(breath)>10:breath=np.convolve(breath,np.ones(10)/10,mode='same')
    envelope=np.ones_like(t)
    atk=int(0.04*sr);rel=int(0.06*sr)
    if 0<atk<len(t):envelope[:atk]=np.linspace(0,1,atk)
    if 0<rel<len(t):envelope[-rel:]=np.linspace(1,0,rel)
    return (signal*envelope*0.3+breath*envelope)/(np.max(np.abs(signal))+1e-10)

def brass_note(freq,duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    phase=2*np.pi*freq*t;signal=np.zeros_like(t)
    for h in range(1,16):signal+=np.sin(h*phase)/h
    envelope=np.ones_like(t);atk=int(0.02*sr)
    if 0<atk<len(t):
        env=np.linspace(0,1.3,atk);split=int(atk*0.7)
        if split<atk:env[split:]=np.linspace(1.3,1.0,atk-split)
        envelope[:atk]=env
    rel=int(0.05*sr)
    if 0<rel<len(t):envelope[-rel:]=np.linspace(1,0,rel)
    return signal*envelope*0.25*1.5/(np.max(np.abs(signal))+1e-10)

def kick_drum(sr=44100,velocity=1.0):
    t=np.linspace(0,0.35,int(0.35*sr),endpoint=False)
    freq_curve=160*np.exp(-t*25)+45;phase=2*np.pi*np.cumsum(freq_curve)/sr
    signal=np.sin(phase)*np.exp(-t*12)*velocity
    return signal/(np.max(np.abs(signal))+1e-10)

def snare_drum(sr=44100,velocity=1.0):
    t=np.linspace(0,0.22,int(0.22*sr),endpoint=False)
    tone=np.sin(2*np.pi*195*t)*np.exp(-t*35)+0.5*np.sin(2*np.pi*330*t)*np.exp(-t*40)
    noise=np.random.randn(len(t))*np.exp(-t*22)
    signal=(0.4*tone+0.6*noise)*velocity
    return signal/(np.max(np.abs(signal))+1e-10)

def hihat(sr=44100,velocity=1.0):
    t=np.linspace(0,0.06,int(0.06*sr),endpoint=False)
    noise=np.random.randn(len(t));filtered=np.diff(noise,prepend=noise[0])
    return filtered*np.exp(-t*45)*velocity/(np.max(np.abs(filtered))+1e-10)

def tom_drum(freq=120,sr=44100,velocity=1.0):
    t=np.linspace(0,0.3,int(0.3*sr),endpoint=False)
    freq_curve=freq*np.exp(-t*8)+freq*0.7;phase=2*np.pi*np.cumsum(freq_curve)/sr
    signal=np.sin(phase)*np.exp(-t*10)*velocity
    return signal/(np.max(np.abs(signal))+1e-10)

def cymbal_crash(sr=44100,velocity=1.0):
    t=np.linspace(0,1.5,int(1.5*sr),endpoint=False)
    noise=np.random.randn(len(t))
    metallic=0
    for f in [5000,6500,8000,9500,11000,13000]:
        metallic+=0.15*np.sin(2*np.pi*f*t+np.random.uniform(0,2*np.pi))
    signal=(noise*0.4+metallic*0.4)*np.exp(-t*3)*velocity
    return signal/(np.max(np.abs(signal))+1e-10)

def riser_sweep(duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    freq_start=200;freq_end=2000
    freqs=np.linspace(freq_start,freq_end,len(t))
    signal=np.sin(2*np.pi*np.cumsum(freqs)/sr)
    signal+=np.random.randn(len(t))*0.3
    envelope=np.linspace(0,1,len(t))**2
    return signal*envelope/(np.max(np.abs(signal))+1e-10)

def noise_sweep(duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    noise=np.random.randn(len(t))
    envelope=np.linspace(0,1,len(t))**2
    return noise*envelope/(np.max(np.abs(noise*envelope))+1e-10)

def bitcrush(audio,bits=8,rate_div=4):
    if len(audio)==0:return audio
    reduced=audio[::rate_div];upsampled=np.repeat(reduced,rate_div)
    if len(upsampled)<len(audio):upsampled=np.pad(upsampled,(0,len(audio)-len(upsampled)))
    elif len(upsampled)>len(audio):upsampled=upsampled[:len(audio)]
    return np.round(upsampled*(2**bits))/(2**bits)

def stutter(audio,sr=44100,size=0.03,repeats=4):
    chunk_size=max(1,int(size*sr))
    if len(audio)==0:return audio
    output=[]
    for i in range(0,len(audio),chunk_size):
        chunk=audio[i:i+chunk_size]
        for _ in range(repeats):output.append(chunk)
    if not output:return audio
    result=np.concatenate(output)
    return result[:len(audio)] if len(result)>=len(audio) else np.pad(result,(0,len(audio)-len(result)))

def reese_bass(freq,duration,sr=44100,detune=0.03):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    saw1=2*(t*freq*(1-detune)%1)-1;saw2=2*(t*freq*(1+detune)%1)-1
    signal=(saw1+saw2)*0.5;signal=np.convolve(signal,np.ones(8)/8,mode='same')
    envelope=np.ones_like(t);rel=min(int(0.05*sr),len(t))
    if rel>0:envelope[-rel:]=np.linspace(1,0,rel)
    return signal*envelope*0.6

def sub_808(freq,duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    freq_curve=freq*2*np.exp(-t*15)+freq;phase=2*np.pi*np.cumsum(freq_curve)/sr
    return np.tanh(np.sin(phase)*1.5)*np.exp(-t*4)

def amen_break(sr=44100,tempo_factor=1.0):
    beat_dur=0.125/tempo_factor
    pattern=[('K',1),('H',.3),('S',.9),('H',.3),('G',.4),('H',.3),('S',.7),('H',.4),
             ('K',.9),('K',.5),('S',.9),('H',.3),('G',.5),('S',.6),('S',.8),('H',.3),
             ('K',1),('H',.3),('S',.9),('G',.4),('K',.8),('H',.4),('S',.8),('H',.3),
             ('K',1),('G',.5),('S',.9),('H',.4),('S',.7),('S',.6),('K',.8),('S',.9)]
    total_samples=int(beat_dur*len(pattern)*sr);output=np.zeros(total_samples)
    kick_s=kick_drum(sr)[:int(0.12*sr)];snare_s=snare_drum(sr)[:int(0.1*sr)]
    ghost_s=snare_drum(sr)[:int(0.06*sr)]*0.4;hat_s=hihat(sr)[:int(0.04*sr)]
    for i,(hit,vel) in enumerate(pattern):
        pos=int(i*beat_dur*sr)
        if hit=='K':sample=kick_s
        elif hit=='S':sample=snare_s
        elif hit=='G':sample=ghost_s
        else:sample=hat_s
        jitter=np.random.randint(-int(0.002*sr),int(0.002*sr)+1)
        pos=max(0,pos+jitter);vel*=np.random.uniform(0.85,1.1)
        end=min(pos+len(sample),total_samples)
        if pos<total_samples:output[pos:end]+=sample[:end-pos]*vel
    return output/(np.max(np.abs(output))+1e-10)

def add_reverb(audio,sr=44100,decay=0.3,mix=0.25):
    delay=int(0.03*sr);reverb=np.zeros_like(audio)
    for d in [1,2,3,4,5,6]:
        pos=delay*d
        if pos<len(audio):reverb[pos:]+=audio[:-pos]*(decay**d)
    return audio*(1-mix)+reverb*mix

def add_distortion(audio,gain=3.0,mix=0.7):
    return audio*(1-mix)+np.tanh(audio*gain)*mix

def soft_compress(audio,threshold=0.6,ratio=3.0):
    compressed=audio.copy()
    mask=np.abs(compressed)>threshold
    compressed[mask]=threshold+(compressed[mask]-threshold)/ratio
    return compressed

def note_to_freq(semitone,base_freq=261.63):
    return base_freq*(2**(semitone/12.0))

# ============================================================
# PROGRESSÕES E ESCALAS
# ============================================================

ALL_PROGRESSIONS=[
    [[0,2,4],[5,0,2],[3,5,0],[4,6,1]],
    [[0,2,4],[3,5,0],[4,6,1],[5,0,2]],
    [[5,0,2],[3,5,0],[0,2,4],[4,6,1]],
    [[0,2,4],[0,2,4],[5,0,2],[4,6,1]],
    [[1,3,5],[4,6,1],[0,2,4],[5,0,2]],
    [[0,2,4,6],[4,6,1,3],[5,0,2,4],[0,2,4,6]],
    [[0,3,5],[5,0,2],[4,6,1],[0,2,4]],
    [[0,2,4],[4,6,1],[0,2,4],[5,0,2]],
    [[0,4,6],[3,5,0],[2,4,6],[5,0,2]],
    [[0,2,4],[5,0,2],[4,6,1],[0,2,4]],
    [[0,3,5],[3,5,0],[5,0,2],[4,6,1]],
    [[0,2,4],[2,4,6],[4,6,1],[5,0,2]],
    [[5,0,2],[4,6,1],[3,5,0],[0,2,4]],
    [[0,2,4],[6,1,3],[5,0,2],[4,6,1]],
    [[0,2,4],[5,0,2],[5,0,2],[0,2,4]],
]

ALL_SCALES={
    'major':[0,2,4,5,7,9,11],'minor':[0,2,3,5,7,8,10],'dorian':[0,2,3,5,7,9,10],
    'phrygian':[0,1,3,5,7,8,10],'lydian':[0,2,4,6,7,9,11],'mixolydian':[0,2,4,5,7,9,10],
    'harmonic_minor':[0,2,3,5,7,8,11],'melodic_minor':[0,2,3,5,7,9,11],
    'pentatonic_major':[0,2,4,7,9],'pentatonic_minor':[0,3,5,7,10],
    'blues':[0,3,5,6,7,10],'whole_tone':[0,2,4,6,8,10],
}
SCALE_NAMES=list(ALL_SCALES.keys())

# ============================================================
# SONG STRUCTURE
# ============================================================

class SongStructure:
    STRUCTURES={
        'pop':['intro','verse','chorus','verse','chorus','bridge','chorus','outro'],
        'rock':['intro','verse','chorus','verse','chorus','solo','chorus','outro'],
        'electronic':['intro','buildup','drop','breakdown','buildup','drop','outro'],
        'cinematic':['intro','theme','development','climax','resolution','outro'],
        'jazz':['intro','head','solo1','head','solo2','head','outro'],
        'ambient':['intro','section1','section2','section3','section4','outro'],
        'breakcore':['intro','chaos1','break','chaos2','break','chaos3','outro'],
        'classical':['exposition','development','recapitulation','coda'],
        'dark':['intro','verse','chorus','verse','chorus','bridge','chorus','outro'],
    }
    
    SECTION_ENERGY={
        'intro':0.3,'verse':0.5,'chorus':0.9,'bridge':0.6,'outro':0.4,
        'buildup':0.7,'drop':1.0,'breakdown':0.2,'solo':0.7,'head':0.6,
        'theme':0.5,'development':0.7,'climax':1.0,'resolution':0.5,
        'exposition':0.5,'recapitulation':0.7,'coda':0.4,
        'chaos1':0.8,'chaos2':0.9,'chaos3':1.0,'break':0.3,
        'section1':0.4,'section2':0.5,'section3':0.6,'section4':0.5,
    }
    
    def __init__(self,style='pop',duration=45,bpm=120):
        self.style=style
        self.duration=duration
        self.bpm=bpm
        self.sections=self._generate_sections()
        self.section_times=self._calculate_times()
    
    def _generate_sections(self):
        base_structure=self.STRUCTURES.get(self.style,self.STRUCTURES['pop'])
        sections=[]
        for section in base_structure:
            if section=='intro':bars=np.random.choice([4,8])
            elif section in ['verse','head','theme','exposition']:bars=np.random.choice([8,12,16])
            elif section in ['chorus','climax','drop','recapitulation']:bars=np.random.choice([8,12])
            elif section=='bridge':bars=np.random.choice([4,8])
            elif section=='buildup':bars=np.random.choice([4,8])
            elif section in ['breakdown','break','coda']:bars=np.random.choice([4,8])
            elif section in ['solo','development']:bars=np.random.choice([8,12,16])
            elif section=='outro':bars=np.random.choice([4,8])
            else:bars=8
            
            sections.append({
                'name':section,'bars':bars,
                'energy':self.SECTION_ENERGY.get(section,0.5)
            })
        return sections
    
    def _calculate_times(self):
        beats_per_bar=4
        beat_duration=60.0/self.bpm
        times=[]
        current_time=0.0
        
        for section in self.sections:
            section_duration=section['bars']*beats_per_bar*beat_duration
            times.append({
                'name':section['name'],
                'start':current_time,
                'end':current_time+section_duration,
                'duration':section_duration,
                'energy':section['energy']
            })
            current_time+=section_duration
        
        total_time=current_time
        if total_time>0:
            scale=self.duration/total_time
            for t in times:
                t['start']*=scale
                t['end']*=scale
                t['duration']*=scale
        
        return times
    
    def get_section_at_time(self,time):
        for section in self.section_times:
            if section['start']<=time<section['end']:
                return section
        return self.section_times[-1]
    
    def get_energy_at_time(self,time):
        section=self.get_section_at_time(time)
        section_start=section['start']
        section_end=section['end']
        section_duration=section['duration']
        position=(time-section_start)/section_duration if section_duration>0 else 0
        base_energy=section['energy']
        
        if section['name']=='buildup':
            return base_energy*(0.3+0.7*position)
        elif section['name']=='drop':
            if position<0.1:return 1.0
            else:return base_energy*(1.0-0.2*(position-0.1)/0.9)
        elif section['name']=='breakdown':
            if position<0.2:return base_energy*(1.0-2.5*position)
            else:return base_energy*(0.5+0.5*(position-0.2)/0.8)
        elif section['name']=='outro':
            return base_energy*(1.0-position*0.7)
        else:
            variation=0.1*np.sin(position*np.pi*2)
            return base_energy+variation
    
    def get_fill_probability(self,time):
        section=self.get_section_at_time(time)
        section_start=section['start']
        section_end=section['end']
        position=(time-section_start)/(section_end-section_start) if (section_end-section_start)>0 else 0
        if position>0.85:return 0.5
        return 0.02

# ============================================================
# RAG
# ============================================================

class MusicRAG:
    def __init__(self):
        self.entries=[
            {"tags":["epic","batalha","heroico"],"scale":"major","dynamics":"loud","style":"cinematic"},
            {"tags":["dark","sombrio","terror"],"scale":"harmonic_minor","dynamics":"quiet","style":"dark"},
            {"tags":["jazz","swing","blues"],"scale":"dorian","dynamics":"medium","style":"jazz"},
            {"tags":["breakcore","glitch","amen"],"scale":"minor","dynamics":"extreme","style":"breakcore"},
            {"tags":["ambient","calmo"],"scale":"lydian","dynamics":"very_quiet","style":"ambient"},
            {"tags":["rock","metal","pesado"],"scale":"phrygian","dynamics":"loud","style":"rock"},
            {"tags":["classical","clássica"],"scale":"major","dynamics":"varied","style":"classical"},
            {"tags":["triste","melancólico"],"scale":"harmonic_minor","dynamics":"quiet","style":"pop"},
            {"tags":["feliz","alegre","pop"],"scale":"major","dynamics":"medium","style":"pop"},
            {"tags":["medieval","fantasia"],"scale":"dorian","dynamics":"medium","style":"cinematic"},
            {"tags":["eletrônica","techno","edm"],"scale":"minor","dynamics":"loud","style":"electronic"},
            {"tags":["boss","intenso"],"scale":"phrygian","dynamics":"extreme","style":"cinematic"},
            {"tags":["romântico","amor"],"scale":"melodic_minor","dynamics":"quiet","style":"pop"},
            {"tags":["tensão","suspense"],"scale":"whole_tone","dynamics":"building","style":"cinematic"},
            {"tags":["vitória","triunfo"],"scale":"lydian","dynamics":"loud","style":"cinematic"},
        ]
    
    def get_context(self,query,style_hint=None):
        if not query and not style_hint:
            return self._random_context()
        
        query_lower=(query or '').lower()
        scores=[]
        for entry in self.entries:
            score=sum(1 for tag in entry['tags'] if tag in query_lower)
            if style_hint and entry['style']==style_hint:
                score+=2
            if score>0:scores.append((score,entry))
        
        if scores:
            scores.sort(key=lambda x:x[0],reverse=True)
            best=scores[0][1]
        else:
            best=np.random.choice(self.entries)
        
        progression=ALL_PROGRESSIONS[np.random.randint(0,len(ALL_PROGRESSIONS))]
        scale_name=best.get('scale','major')
        if np.random.random()<0.3:scale_name=np.random.choice(SCALE_NAMES)
        scale=ALL_SCALES.get(scale_name,ALL_SCALES['major'])
        dynamics=best.get('dynamics','medium')
        bpm_map={'very_quiet':(40,70),'quiet':(60,90),'medium':(90,130),'loud':(120,160),'extreme':(160,230),'building':(100,140),'varied':(70,140)}
        bpm_range=bpm_map.get(dynamics,(90,130))
        bpm=np.random.randint(bpm_range[0],bpm_range[1]+1)
        intensity_map={'very_quiet':0.35,'quiet':0.55,'medium':0.7,'loud':0.85,'extreme':0.95,'building':0.75,'varied':0.65}
        intensity=intensity_map.get(dynamics,0.7)+np.random.uniform(-0.1,0.1)
        intensity=np.clip(intensity,0.2,0.98)
        style=style_hint or best.get('style','pop')
        
        print(f"  📚 RAG: estilo={style}, escala={scale_name}, BPM={bpm}")
        return {'scale':scale,'scale_name':scale_name,'progression':progression,'bpm':bpm,'intensity':intensity,'style':style,'seed':get_dynamic_seed()}
    
    def _random_context(self):
        scale_name=np.random.choice(SCALE_NAMES)
        progression=ALL_PROGRESSIONS[np.random.randint(0,len(ALL_PROGRESSIONS))]
        style=np.random.choice(list(SongStructure.STRUCTURES.keys()))
        return {'scale':ALL_SCALES[scale_name],'scale_name':scale_name,'progression':progression,'bpm':np.random.randint(50,200),'intensity':np.random.uniform(0.3,0.95),'style':style,'seed':get_dynamic_seed()}

# ============================================================
# GERAÇÃO COM INTELIGÊNCIA COMBINADA
# ============================================================

def generate_with_intelligence(duration, prompt=None, style=None, use_rag=True, sr=44100):
    """Geração com inteligência máxima: RAG + Modelo + Análise"""
    
    print("="*60)
    print("🧠 INTELIGÊNCIA MÁXIMA ATIVADA")
    print("="*60)
    
    # 1. Interpretar prompt
    interpreter = PromptInterpreter()
    prompt_params = interpreter.interpret(prompt)
    
    # 2. Analisar músicas de referência
    analyzer = AudioAnalyzer()
    reference_analysis = analyzer.analyze_directory()
    
    # 3. Carregar modelo MoE
    moe_loader = MoELoader()
    model_loaded = moe_loader.load_model()
    
    # 4. Determinar estilo final
    final_style = style
    
    # Prioridade: prompt > modelo > análise > RAG
    if prompt_params.get('style'):
        final_style = prompt_params['style']
        print(f"  🎯 Estilo do prompt: {final_style}")
    elif model_loaded and reference_analysis:
        # Usar modelo para inferir estilo
        # Criar vetor de features da análise
        feature_vector = np.array([
            reference_analysis['avg_bpm'] / 200,
            reference_analysis['avg_spectral_centroid'] / 5000,
            reference_analysis['avg_rms'],
            reference_analysis['avg_duration'] / 300,
        ] + [reference_analysis['avg_bands'].get(b, 0) / 1e6 for b in ['sub_bass', 'bass', 'low_mids', 'mids', 'high_mids', 'highs']])
        
        style_probs = moe_loader.predict_style_affinity(feature_vector)
        if style_probs is not None:
            style_params = moe_loader.get_style_parameters(style_probs)
            if style_params:
                final_style = style_params.get('style', style)
                print(f"  🧠 Estilo do modelo MoE: {final_style}")
    
    if not final_style:
        if reference_analysis:
            final_style = analyzer.get_style_from_analysis()
            print(f"  📊 Estilo da análise: {final_style}")
        else:
            final_style = 'pop'
    
    # 5. RAG com contexto
    rag = MusicRAG() if use_rag else None
    if use_rag and rag:
        rag_context = rag.get_context(prompt, style_hint=final_style)
    else:
        rag_context = rag._random_context() if rag else {
            'scale':[0,2,4,5,7,9,11],'scale_name':'major',
            'progression':ALL_PROGRESSIONS[0],'bpm':120,
            'intensity':0.7,'style':final_style,'seed':get_dynamic_seed()
        }
    
    # 6. Ajustar BPM baseado em análise
    if reference_analysis:
        ref_bpm = reference_analysis['avg_bpm']
        # Misturar BPM do RAG com BPM da referência
        rag_context['bpm'] = int(rag_context['bpm'] * 0.6 + ref_bpm * 0.4)
        print(f"  🎵 BPM ajustado pela referência: {rag_context['bpm']}")
    
    # 7. Ajustar BPM do prompt se especificado
    if prompt_params.get('target_bpm'):
        rag_context['bpm'] = prompt_params['target_bpm']
        print(f"  🎯 BPM do prompt: {rag_context['bpm']}")
    
    # 8. Ajustar duração do prompt se especificado
    if prompt_params.get('target_duration'):
        duration = prompt_params['target_duration']
        print(f"  🎯 Duração do prompt: {duration}s")
    
    # 9. Aplicar parâmetros do prompt
    if prompt_params.get('intensity'):
        rag_context['intensity'] = prompt_params['intensity']
    
    # 10. Verificar se é breakcore
    if prompt_params.get('breakcore', 0) > 1.5 or final_style == 'breakcore':
        print("💥 Modo BREAKCORE")
        return generate_breakcore(duration)
    
    # 11. Gerar com estrutura
    print(f"\n🎵 Gerando música final...")
    return generate_with_structure(duration, rag_context, sr)

def generate_with_structure(duration,rag_context,sr=44100):
    """Gera música com estrutura completa"""
    np.random.seed(rag_context['seed'])
    
    bpm=rag_context['bpm']
    style=rag_context.get('style','pop')
    base_freq_options=[220.0,246.94,261.63,293.66,329.63,349.23]
    base_freq=np.random.choice(base_freq_options)
    scale=rag_context['scale']
    progression=rag_context['progression']
    
    structure=SongStructure(style=style,duration=duration,bpm=bpm)
    
    print(f"   Estrutura: {[s['name'] for s in structure.sections]}")
    
    total_samples=int(duration*sr)
    beat_duration=60.0/bpm
    
    drums_track=np.zeros(total_samples)
    bass_track=np.zeros(total_samples)
    chords_track=np.zeros(total_samples)
    melody_track=np.zeros(total_samples)
    fx_track=np.zeros(total_samples)
    
    kick=kick_drum(sr);snare=snare_drum(sr);hihat=hihat(sr)
    tom1=tom_drum(200,sr);tom2=tom_drum(150,sr);tom3=tom_drum(100,sr)
    crash=cymbal_crash(sr)
    
    n_beats=int(duration/beat_duration)
    
    # Bateria
    print("  🥁 Bateria...")
    for beat in range(n_beats):
        time=beat*beat_duration
        section=structure.get_section_at_time(time)
        section_name=section['name']
        energy=structure.get_energy_at_time(time)
        vel=0.4+0.6*energy
        pos=int(beat*beat_duration*sr)
        
        if section_name in ['intro','breakdown','break']:
            if beat%4==0 and pos+len(kick)<=total_samples:drums_track[pos:pos+len(kick)]+=kick*vel*0.6
            if beat%4==2 and pos+len(snare)<=total_samples:drums_track[pos:pos+len(snare)]+=snare*vel*0.5
            if pos+len(hihat)<=total_samples:drums_track[pos:pos+len(hihat)]+=hihat*vel*0.3
        elif section_name in ['verse','head','theme','exposition']:
            if beat%4 in [0,2] and pos+len(kick)<=total_samples:drums_track[pos:pos+len(kick)]+=kick*vel*0.8
            if beat%4 in [1,3] and pos+len(snare)<=total_samples:drums_track[pos:pos+len(snare)]+=snare*vel*0.7
            if pos+len(hihat)<=total_samples:drums_track[pos:pos+len(hihat)]+=hihat*vel*0.4
        elif section_name in ['chorus','drop','climax','chaos1','chaos2','chaos3']:
            if beat%2==0 and pos+len(kick)<=total_samples:drums_track[pos:pos+len(kick)]+=kick*vel*0.9
            if beat%4 in [1,3] and pos+len(snare)<=total_samples:drums_track[pos:pos+len(snare)]+=snare*vel*0.8
            for sub in [0,0.5]:
                sp=int((beat+sub)*beat_duration*sr)
                if sp+len(hihat)<=total_samples:drums_track[sp:sp+len(hihat)]+=hihat*vel*(0.4 if sub==0 else 0.25)
        elif section_name=='buildup':
            section_start=section['start'];section_duration=section['duration']
            position=(time-section_start)/section_duration if section_duration>0 else 0
            if position<0.5:
                if beat%4==0 and pos+len(kick)<=total_samples:drums_track[pos:pos+len(kick)]+=kick*vel*0.7
                if pos+len(hihat)<=total_samples:drums_track[pos:pos+len(hihat)]+=hihat*vel*0.3
            elif position<0.75:
                if beat%2==0 and pos+len(kick)<=total_samples:drums_track[pos:pos+len(kick)]+=kick*vel*0.8
                if pos+len(hihat)<=total_samples:drums_track[pos:pos+len(hihat)]+=hihat*vel*0.4
            else:
                for sub in [0,0.25,0.5,0.75]:
                    sp=int((beat+sub)*beat_duration*sr)
                    if sp+len(hihat)<=total_samples:drums_track[sp:sp+len(hihat)]+=hihat*vel*0.5
                    if sp+len(snare)<=total_samples:drums_track[sp:sp+len(snare)]+=snare*vel*0.4
        else:
            if beat%4 in [0,2] and pos+len(kick)<=total_samples:drums_track[pos:pos+len(kick)]+=kick*vel*0.7
            if beat%4 in [1,3] and pos+len(snare)<=total_samples:drums_track[pos:pos+len(snare)]+=snare*vel*0.6
            if pos+len(hihat)<=total_samples:drums_track[pos:pos+len(hihat)]+=hihat*vel*0.35
        
        fill_prob=structure.get_fill_probability(time)
        if np.random.random()<fill_prob:
            fill_start=pos
            fill_duration=int(beat_duration*sr)
            if fill_start+fill_duration<=total_samples:
                toms=[tom1,tom2,tom3,tom2]
                for i,tom in enumerate(toms):
                    tom_pos=fill_start+int(i*beat_duration*sr/4)
                    if tom_pos+len(tom)<=total_samples:drums_track[tom_pos:tom_pos+len(tom)]+=tom*vel*0.7
                crash_pos=fill_start+int(beat_duration*sr*0.9)
                if crash_pos+len(crash)<=total_samples:drums_track[crash_pos:crash_pos+len(crash)]+=crash*vel*0.8
    
    # Baixo
    print("  🎸 Baixo...")
    for beat in range(n_beats):
        time=beat*beat_duration
        section=structure.get_section_at_time(time)
        energy=structure.get_energy_at_time(time)
        chord_idx=(beat//4)%len(progression)
        root=progression[chord_idx][0]
        root_freq=note_to_freq(scale[root%len(scale)],base_freq)/2
        vel=0.3+0.7*energy
        
        if section['name'] in ['intro','breakdown','break']:
            if beat%4==0:
                note=karplus_strong(root_freq,beat_duration*3,sr,damping=0.998)
                pos=int(beat*beat_duration*sr)
                if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*vel*0.3
        elif section['name'] in ['verse','head','theme']:
            if beat%2==0:
                note=karplus_strong(root_freq,beat_duration*1.5,sr,damping=0.997)
                pos=int(beat*beat_duration*sr)
                if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*vel*0.4
        elif section['name'] in ['chorus','drop','climax','chaos1','chaos2','chaos3']:
            freq=root_freq if beat%2==0 else root_freq*2
            note=karplus_strong(freq,beat_duration*0.9,sr)
            pos=int(beat*beat_duration*sr)
            if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*vel*0.45
        else:
            if beat%2==0:
                note=karplus_strong(root_freq,beat_duration*1.2,sr)
                pos=int(beat*beat_duration*sr)
                if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*vel*0.4
    
    # Acordes
    print("  🎹 Acordes...")
    chord_duration=beat_duration*4
    for i in range(int(duration/chord_duration)):
        time=i*chord_duration
        section=structure.get_section_at_time(time)
        energy=structure.get_energy_at_time(time)
        chord=progression[i%len(progression)]
        pos=int(i*chord_duration*sr)
        vel=0.2+0.6*energy
        
        if section['name'] in ['intro','breakdown','break','outro']:
            for nd in chord:
                freq=note_to_freq(scale[nd%len(scale)],base_freq)
                note=synth_pad(freq,chord_duration*0.95,sr)
                if pos+len(note)<=total_samples:chords_track[pos:pos+len(note)]+=note*vel*0.15
        elif section['name'] in ['verse','head','theme','exposition']:
            for nd in chord:
                freq=note_to_freq(scale[nd%len(scale)],base_freq)
                note=piano_note(freq,chord_duration*0.9,sr)
                if pos+len(note)<=total_samples:chords_track[pos:pos+len(note)]+=note*vel*0.2
        elif section['name'] in ['chorus','drop','climax','recapitulation']:
            for nd in chord:
                freq=note_to_freq(scale[nd%len(scale)],base_freq)
                note_p=piano_note(freq,chord_duration*0.9,sr)
                if pos+len(note_p)<=total_samples:chords_track[pos:pos+len(note_p)]+=note_p*vel*0.15
                note_s=violin_note(freq,chord_duration*0.95,sr)
                if pos+len(note_s)<=total_samples:chords_track[pos:pos+len(note_s)]+=note_s*vel*0.12
        else:
            for nd in chord:
                freq=note_to_freq(scale[nd%len(scale)],base_freq)
                note=piano_note(freq,chord_duration*0.9,sr)
                if pos+len(note)<=total_samples:chords_track[pos:pos+len(note)]+=note*vel*0.18
    
    # Melodia
    print("  🎶 Melodia...")
    note_duration=beat_duration/2
    current_degree=0
    prev_section=None
    
    for i in range(int(duration/note_duration)):
        time=i*note_duration
        section=structure.get_section_at_time(time)
        section_name=section['name']
        energy=structure.get_energy_at_time(time)
        
        if section!=prev_section:
            current_degree=np.random.randint(0,len(scale))
            prev_section=section
        
        if section_name in ['intro','breakdown','break']:
            if np.random.random()<0.3:
                step=np.random.choice([-1,0,1])
                current_degree=max(0,min(current_degree+step,len(scale)-1))
                freq=note_to_freq(scale[current_degree],base_freq)
                pos=int(time*sr)
                nlen=int(note_duration*sr*3)
                note=flute_note(freq,nlen/sr,sr)
                if pos+len(note)<=total_samples:melody_track[pos:pos+len(note)]+=note*energy*0.25
        elif section_name in ['verse','head','theme','exposition']:
            if np.random.random()<0.7:
                step=np.random.choice([-2,-1,0,1,2],p=[0.1,0.25,0.3,0.25,0.1])
                current_degree=max(0,min(current_degree+step,len(scale)*2-1))
                octave=current_degree//len(scale)
                degree=current_degree%len(scale)
                freq=note_to_freq(scale[degree],base_freq)*(2**octave)
                pos=int(time*sr)
                nlen=int(note_duration*sr*1.5)
                note=piano_note(freq,nlen/sr,sr)
                if pos+len(note)<=total_samples:melody_track[pos:pos+len(note)]+=note*energy*0.3
        elif section_name in ['chorus','drop','climax','chaos1','chaos2','chaos3']:
            if np.random.random()<0.85:
                step=np.random.choice([-3,-2,-1,0,1,2,3],p=[0.05,0.15,0.2,0.2,0.2,0.15,0.05])
                current_degree=max(0,min(current_degree+step,len(scale)*3-1))
                octave=current_degree//len(scale)
                degree=current_degree%len(scale)
                freq=note_to_freq(scale[degree],base_freq)*(2**octave)
                pos=int(time*sr)
                dur_mult=np.random.choice([0.5,1.0,1.5,2.0],p=[0.3,0.4,0.2,0.1])
                nlen=int(note_duration*sr*1.8*dur_mult)
                inst_choice=np.random.choice(['piano','violin','brass','synth'])
                if inst_choice=='piano':note=piano_note(freq,nlen/sr,sr)
                elif inst_choice=='violin':note=violin_note(freq,nlen/sr,sr)
                elif inst_choice=='brass':note=brass_note(freq,nlen/sr,sr)
                else:note=synth_pad(freq,nlen/sr,sr)
                if pos+len(note)<=total_samples:melody_track[pos:pos+len(note)]+=note*energy*0.35
        else:
            if np.random.random()<0.6:
                step=np.random.choice([-1,0,1,2])
                current_degree=max(0,min(current_degree+step,len(scale)*2-1))
                octave=current_degree//len(scale)
                degree=current_degree%len(scale)
                freq=note_to_freq(scale[degree],base_freq)*(2**octave)
                pos=int(time*sr)
                nlen=int(note_duration*sr*1.5)
                note=piano_note(freq,nlen/sr,sr)
                if pos+len(note)<=total_samples:melody_track[pos:pos+len(note)]+=note*energy*0.3
    
    # Efeitos
    print("  ✨ Efeitos...")
    for section in structure.section_times:
        if section['name']=='buildup':
            riser_start=int((section['end']-2.0)*sr)
            riser_duration=2.0
            if riser_start>0 and riser_start+int(riser_duration*sr)<=total_samples:
                riser=riser_sweep(riser_duration,sr)
                fx_track[riser_start:riser_start+len(riser)]+=riser*0.15
    
    for i in range(len(structure.section_times)-1):
        transition_time=structure.section_times[i]['end']
        sweep_start=int((transition_time-0.5)*sr)
        sweep_duration=0.5
        if sweep_start>0 and sweep_start+int(sweep_duration*sr)<=total_samples:
            sweep=noise_sweep(sweep_duration,sr)
            fx_track[sweep_start:sweep_start+len(sweep)]+=sweep*0.1
    
    # Mixagem
    print("  🎛️ Mixagem...")
    mix=drums_track+bass_track+chords_track+melody_track+fx_track
    reverb_amount=np.random.uniform(0.15,0.3)
    mix=add_reverb(mix,sr,decay=0.35,mix=reverb_amount)
    mix=soft_compress(mix,threshold=0.5,ratio=3.0)
    mix=mix/(np.max(np.abs(mix))+1e-10)*0.9
    fade_in=int(0.5*sr);fade_out=int(1.5*sr)
    if fade_in<len(mix):mix[:fade_in]*=np.linspace(0,1,fade_in)
    if fade_out<len(mix):mix[-fade_out:]*=np.linspace(1,0,fade_out)
    
    return mix,sr

def generate_breakcore(duration,sr=44100,intensity=1.0):
    seed=get_dynamic_seed()
    np.random.seed(seed)
    total_samples=int(duration*sr);bpm=np.random.randint(180,230)
    print(f"  💥 BREAKCORE (seed={seed}): BPM={bpm}")
    break_source=amen_break(sr,tempo_factor=bpm/180)
    drums_track=np.zeros(total_samples);pos=0
    while pos<total_samples:
        max_start=max(1,len(break_source)-int(0.15*sr))
        start=np.random.randint(0,max_start)
        length=np.random.choice([int(0.03*sr),int(0.06*sr),int(0.12*sr),int(0.25*sr)])
        chunk=break_source[start:start+length].copy()
        if len(chunk)==0:pos+=int(0.05*sr);continue
        effect=np.random.choice(['none','stutter','reverse','crush','pitch_up','pitch_down'])
        if effect=='stutter':chunk=stutter(chunk,sr,0.015,np.random.randint(2,8))
        elif effect=='reverse':chunk=chunk[::-1]
        elif effect=='crush':chunk=bitcrush(chunk,np.random.randint(4,10),np.random.randint(2,8))
        elif effect=='pitch_up':
            idx=np.round(np.arange(0,len(chunk),np.random.uniform(1.3,2.0))).astype(int)
            chunk=chunk[idx[idx<len(chunk)]]
        elif effect=='pitch_down':
            idx=np.round(np.arange(0,len(chunk),np.random.uniform(0.5,0.8))).astype(int)
            chunk=chunk[idx[idx<len(chunk)]]
        if np.random.random()<0.4:chunk=np.tanh(chunk*np.random.uniform(2,6))
        end=min(pos+len(chunk),total_samples)
        if pos<total_samples and len(chunk)>0:
            drums_track[pos:end]+=chunk[:end-pos]*np.random.uniform(0.5,1.0)*intensity*0.8
        pos+=len(chunk)
        if np.random.random()<0.24:pos+=np.random.randint(int(0.01*sr),int(0.08*sr))
    bass_track=np.zeros(total_samples);beat_dur=60.0/bpm
    for i in range(int(duration/beat_dur)):
        pos=int(i*beat_dur*sr)
        if np.random.random()<0.6:
            freq=np.random.choice([55.0,58.27,65.41,73.42,82.41])
            note_dur=beat_dur*np.random.choice([0.5,1.0,1.5])
            bass_note=reese_bass(freq,note_dur,sr) if np.random.random()<0.5 else sub_808(freq,note_dur,sr)
            end=min(pos+len(bass_note),total_samples)
            if pos<total_samples:bass_track[pos:end]+=bass_note[:end-pos]*0.7
    glitch_track=np.zeros(total_samples)
    for _ in range(int(duration*3)):
        if total_samples<=int(0.1*sr):break
        pos=np.random.randint(0,max(1,total_samples-int(0.1*sr)))
        length=np.random.randint(int(0.005*sr),int(0.08*sr))
        t=np.arange(length)/sr
        gt=np.random.choice(['noise','tone','sweep'])
        if gt=='noise':glitch=np.random.randn(length)*np.exp(-np.arange(length)/(length/3))
        elif gt=='tone':glitch=np.sin(2*np.pi*np.random.uniform(500,6000)*t)*np.exp(-t*25)
        else:
            freqs=np.linspace(np.random.uniform(200,3000),np.random.uniform(200,3000),length)
            glitch=np.sin(2*np.pi*np.cumsum(freqs)/sr)
        glitch=bitcrush(glitch,np.random.randint(3,8),np.random.randint(2,10))
        end=min(pos+len(glitch),total_samples)
        if len(glitch)>0:glitch_track[pos:end]+=glitch[:end-pos]*0.15
    mix=drums_track*0.85+bass_track*0.75+glitch_track*0.6
    mix=np.tanh(mix*2.0);mix=soft_compress(mix,0.35,5.0)
    mix=mix/(np.max(np.abs(mix))+1e-10)*0.95
    fi,fo=int(0.1*sr),int(0.3*sr)
    if fi<len(mix):mix[:fi]*=np.linspace(0,1,fi)
    if fo<len(mix):mix[-fo:]*=np.linspace(1,0,fo)
    return mix,sr

# ============================================================
# MAIN
# ============================================================

def batch_generate():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--prompt",type=str,default="")
    parser.add_argument("--style",type=str,default="epic")
    parser.add_argument("--duration",type=int,default=45)
    parser.add_argument("--use-rag",type=str,default="true")
    parser.add_argument("--batch",action="store_true")
    args=parser.parse_args()
    
    use_rag=args.use_rag.lower()=="true"
    print("="*60)
    print("🎵 IA MUSIC GENERATOR PRO - INTELIGÊNCIA MÁXIMA")
    print("="*60)
    
    prompt=args.prompt if args.prompt else None
    style=args.style if not prompt else None
    audio,sr=generate_with_intelligence(args.duration,prompt=prompt,style=style,use_rag=use_rag)
    metadata={"prompt":args.prompt,"style":args.style,"duration":args.duration,"use_rag":use_rag,"intelligence":"max"}
    filepath,number=save_song(audio,sr,metadata)
    print(f"\n✅ Música #{number}: {filepath}")

def main():
    while True:
        print("="*50);print("🎵 IA MUSIC PRO - Inteligência Máxima");print("="*50)
        print("1. Prompt  2. Estilo  3. Breakcore  4. Ver  5. Sair")
        choice=input("> ").strip()
        if choice=='1':
            prompt=input("Prompt: ").strip()
            duration=input("Duração (30/45/60/90): ").strip()
            if duration not in ["30","45","60","90"]:duration="45"
            audio,sr=generate_with_intelligence(int(duration),prompt=prompt)
            fp,num=save_song(audio,sr,{"prompt":prompt,"duration":int(duration),"intelligence":"max"})
            print(f"✅ #{num}: {fp}")
        elif choice=='2':
            print("1.Epico 2.Boss 3.Dark 4.Rock 5.Ambient 6.Eletronico 7.Jazz 8.Classico")
            s=input("Estilo: ").strip()
            styles={"1":"epic","2":"bossfight","3":"dark","4":"rock","5":"ambient","6":"electronic","7":"jazz","8":"classical"}
            duration=input("Duração: ").strip()
            if duration not in ["30","45","60","90"]:duration="45"
            audio,sr=generate_with_intelligence(int(duration),style=styles.get(s,"epic"))
            fp,num=save_song(audio,sr,{"style":styles.get(s,"epic"),"intelligence":"max"})
            print(f"✅ #{num}: {fp}")
        elif choice=='3':
            duration=input("Duração: ").strip()
            if duration not in ["30","45","60","90"]:duration="45"
            audio,sr=generate_breakcore(int(duration))
            fp,num=save_song(audio,sr,{"style":"breakcore"})
            print(f"✅ #{num}: {fp}")
        elif choice=='4':
            songs=sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')])
            for s in songs:print(f"  {s}")
        elif choice=='5':break
        input("ENTER...")

if __name__=="__main__":
    if "--batch" in sys.argv:batch_generate()
    else:main()
