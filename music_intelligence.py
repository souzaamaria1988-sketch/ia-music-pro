#!/usr/bin/env python3
"""
🧠 MUSIC INTELLIGENCE - CÉREBRO DA IA
- Feedback Loop (aprende com próprias músicas)
- Memória de sucessos/erros
- Análise de qualidade automática
- Auto-mixing (compressor, limiter, EQ)
- Normalização por instrumento
"""
import os,sys,json,time,gc
import numpy as np
from pathlib import Path

MEMORY_DIR="memory"
OUTPUT_DIR="song_output"

def ensure_memory_dir():
    os.makedirs(MEMORY_DIR, exist_ok=True)

class MusicMemory:
    """Memória de longo prazo da IA"""
    
    def __init__(self):
        ensure_memory_dir()
        self.success_file = os.path.join(MEMORY_DIR, "successes.json")
        self.errors_file = os.path.join(MEMORY_DIR, "errors.json")
        self.stats_file = os.path.join(MEMORY_DIR, "stats.json")
        self.successes = self._load(self.success_file, [])
        self.errors = self._load(self.errors_file, [])
        self.stats = self._load(self.stats_file, {
            "total_generated": 0,
            "total_accepted": 0,
            "total_rejected": 0,
            "average_quality": 0.0,
            "best_quality": 0.0,
            "worst_quality": 1.0
        })
    
    def _load(self, path, default):
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                return default
        return default
    
    def _save(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def remember_success(self, song_info):
        """Lembra de uma música boa"""
        self.successes.append({
            **song_info,
            "timestamp": time.time()
        })
        # Manter só últimos 100 sucessos
        self.successes = self.successes[-100:]
        self._save(self.success_file, self.successes)
    
    def remember_error(self, song_info, reason):
        """Lembra de um erro"""
        self.errors.append({
            **song_info,
            "reason": reason,
            "timestamp": time.time()
        })
        # Manter só últimos 50 erros
        self.errors = self.errors[-50:]
        self._save(self.errors_file, self.errors)
    
    def update_stats(self, quality, accepted):
        self.stats["total_generated"] += 1
        if accepted:
            self.stats["total_accepted"] += 1
        else:
            self.stats["total_rejected"] += 1
        
        # Média móvel
        n = self.stats["total_generated"]
        self.stats["average_quality"] = (
            self.stats["average_quality"] * (n-1) + quality
        ) / n
        self.stats["best_quality"] = max(self.stats["best_quality"], quality)
        self.stats["worst_quality"] = min(self.stats["worst_quality"], quality)
        self._save(self.stats_file, self.stats)
    
    def get_learned_params(self):
        """Retorna parâmetros aprendidos com base nas memórias"""
        params = {
            "preferred_bpm_range": (90, 140),
            "preferred_intensity": 0.7,
            "avoid_patterns": [],
            "successful_instruments": {},
            "volume_corrections": {}
        }
        
        if len(self.successes) >= 5:
            bpms = [s.get('bpm', 120) for s in self.successes if 'bpm' in s]
            if bpms:
                params["preferred_bpm_range"] = (int(min(bpms)), int(max(bpms)))
            
            intensities = [s.get('intensity', 0.7) for s in self.successes if 'intensity' in s]
            if intensities:
                params["preferred_intensity"] = float(np.mean(intensities))
            
            # Instrumentos que funcionaram
            for s in self.successes:
                inst = s.get('main_instrument', 'unknown')
                params["successful_instruments"][inst] =                     params["successful_instruments"].get(inst, 0) + 1
        
        if len(self.errors) > 0:
            # Evitar padrões de erro
            for e in self.errors:
                pattern = e.get('error_pattern')
                if pattern:
                    params["avoid_patterns"].append(pattern)
                
                # Correções de volume aprendidas
                vol_issue = e.get('volume_issue')
                if vol_issue:
                    params["volume_corrections"][vol_issue] = 0.7
        
        return params
    
    def get_report(self):
        """Relatório de aprendizado"""
        return {
            "sucessos": len(self.successes),
            "erros": len(self.errors),
            "total_geradas": self.stats["total_generated"],
            "taxa_sucesso": (
                self.stats["total_accepted"] / max(1, self.stats["total_generated"]) * 100
            ),
            "qualidade_media": self.stats["average_quality"],
            "melhor_qualidade": self.stats["best_quality"]
        }


class QualityAnalyzer:
    """Analisa qualidade de músicas geradas"""
    
    def __init__(self, sr=44100):
        self.sr = sr
    
    def analyze(self, audio):
        """Retorna score de 0 a 1"""
        if len(audio) == 0:
            return 0.0, {"error": "empty"}
        
        scores = {}
        
        # 1. RMS (energia geral)
        rms = np.sqrt(np.mean(audio**2))
        scores['rms'] = min(1.0, rms / 0.3)  # Ideal ~0.2-0.3
        
        # 2. Pico (não deve clipar)
        peak = np.max(np.abs(audio))
        scores['peak'] = 1.0 if peak < 0.95 else 0.3
        
        # 3. Clipping (% de samples acima de 0.99)
        clipping = np.mean(np.abs(audio) > 0.99)
        scores['clipping'] = max(0.0, 1.0 - clipping * 100)
        
        # 4. Dinâmica (diferença entre loud e quiet)
        frame_size = self.sr // 10  # 100ms frames
        if len(audio) > frame_size:
            frames = [audio[i:i+frame_size] for i in range(0, len(audio)-frame_size, frame_size)]
            rms_per_frame = [np.sqrt(np.mean(f**2)) for f in frames]
            dynamic_range = max(rms_per_frame) / (min(rms_per_frame) + 1e-6)
            scores['dynamic'] = min(1.0, dynamic_range / 10.0)  # Ideal ~10:1
        else:
            scores['dynamic'] = 0.5
        
        # 5. Zero crossing rate (textura)
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        scores['zcr'] = min(1.0, zcr * 10)  # Alguma textura é boa
        
        # 6. Variação espectral
        fft = np.abs(np.fft.rfft(audio))
        spectral_variance = np.std(fft) / (np.mean(fft) + 1e-6)
        scores['spectral'] = min(1.0, spectral_variance / 5.0)
        
        # 7. Comprimento mínimo
        scores['length'] = min(1.0, len(audio) / (self.sr * 30))  # Mínimo 30s
        
        # Score total (média ponderada)
        weights = {
            'rms': 0.15,
            'peak': 0.25,  # Muito importante: não clipar!
            'clipping': 0.20,
            'dynamic': 0.15,
            'zcr': 0.10,
            'spectral': 0.10,
            'length': 0.05
        }
        
        total_score = sum(scores[k] * weights[k] for k in weights)
        
        # Diagnóstico de problemas
        issues = []
        if scores['peak'] < 0.5:
            issues.append("CLIPPING")
        if scores['rms'] < 0.3:
            issues.append("MUITO_QUIETO")
        if scores['rms'] > 0.9:
            issues.append("MUITO_ALTO")
        if scores['dynamic'] < 0.3:
            issues.append("SEM_DINAMICA")
        
        return total_score, {
            'scores': scores,
            'issues': issues,
            'rms': float(rms),
            'peak': float(peak),
            'clipping': float(clipping)
        }
    
    def is_acceptable(self, audio, min_score=0.55):
        score, details = self.analyze(audio)
        return score >= min_score, score, details


class AutoMixer:
    """Mixagem automática com compressor, limiter, EQ"""
    
    def __init__(self, sr=44100):
        self.sr = sr
    
    def normalize_instrument(self, audio, target_rms=0.15):
        """Normaliza um instrumento para RMS específico"""
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-6:
            return audio
        return audio * (target_rms / rms)
    
    def compress(self, audio, threshold=0.4, ratio=3.0, attack_ms=5, release_ms=50):
        """Compressor dinâmico"""
        output = audio.copy()
        attack = int(self.sr * attack_ms / 1000)
        release = int(self.sr * release_ms / 1000)
        
        envelope = np.ones(len(audio))
        current_gain = 1.0
        
        for i in range(len(audio)):
            level = abs(audio[i])
            
            if level > threshold:
                # Compressão
                target_gain = threshold + (level - threshold) / ratio
                target_gain /= level
            else:
                target_gain = 1.0
            
            # Attack/Release suave
            if target_gain < current_gain:
                current_gain += (target_gain - current_gain) / max(1, attack)
            else:
                current_gain += (target_gain - current_gain) / max(1, release)
            
            envelope[i] = current_gain
        
        return output * envelope
    
    def limit(self, audio, ceiling=0.95):
        """Limiter: nunca ultrapassa ceiling"""
        peak = np.max(np.abs(audio))
        if peak > ceiling:
            audio = audio * (ceiling / peak)
        
        # Soft clip nos extremos
        mask = np.abs(audio) > ceiling * 0.9
        audio[mask] = np.tanh(audio[mask] * 2) * ceiling
        
        return audio
    
    def eq_auto(self, audio):
        """EQ automático baseado em análise espectral"""
        # FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        magnitude = np.abs(fft)
        
        # Identificar frequências problemáticas
        # Boost graves se muito fracos
        bass_mask = freqs < 200
        bass_energy = np.sum(magnitude[bass_mask]**2)
        total_energy = np.sum(magnitude**2)
        bass_ratio = bass_energy / (total_energy + 1e-10)
        
        if bass_ratio < 0.1:  # Graves fracos
            fft[bass_mask] *= 1.3
        
        # Cortar frequências muito altas (acima de 15kHz) que causam harshness
        high_mask = freqs > 15000
        fft[high_mask] *= 0.7
        
        # Transformada inversa
        return np.fft.irfft(fft, n=len(audio))
    
    def stereo_width(self, audio, width=1.2):
        """Alarga estéreo (se for estéreo)"""
        if len(audio.shape) == 1:
            return audio  # Mono
        mid = (audio[:, 0] + audio[:, 1]) / 2
        side = (audio[:, 0] - audio[:, 1]) / 2
        side *= width
        output = np.zeros_like(audio)
        output[:, 0] = mid + side
        output[:, 1] = mid - side
        return output
    
    def mix_tracks(self, tracks, volumes=None):
        """
        Mixa múltiplas tracks com balanceamento automático
        
        tracks: dict com {'drums': array, 'bass': array, 'chords': array, 'melody': array, ...}
        volumes: dict opcional com volumes personalizados
        """
        if not tracks:
            return np.array([])
        
        # Volumes padrão por tipo de instrumento (balanceados!)
        default_volumes = {
            'drums': 0.35,
            'bass': 0.40,
            'chords': 0.25,
            'melody': 0.30,
            'fx': 0.15,
            'strings': 0.20,
            'pad': 0.18
        }
        
        if volumes is None:
            volumes = default_volumes
        
        # Determinar tamanho (o maior)
        max_len = max(len(t) for t in tracks.values())
        mix = np.zeros(max_len)
        
        for name, track in tracks.items():
            if len(track) == 0:
                continue
            
            # Normalizar cada instrumento
            normalized = self.normalize_instrument(track, target_rms=0.15)
            
            # Aplicar volume
            vol = volumes.get(name, 0.3)
            
            # Pad para tamanho igual
            if len(normalized) < max_len:
                normalized = np.pad(normalized, (0, max_len - len(normalized)))
            
            mix += normalized * vol
        
        # Processamento final
        mix = self.eq_auto(mix)
        mix = self.compress(mix, threshold=0.4, ratio=3.0)
        mix = self.limit(mix, ceiling=0.95)
        
        # Normalização final
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = mix * (0.9 / peak)
        
        return mix
    
    def fix_bad_mix(self, audio):
        """Corrige mix problemático"""
        # Detectar clipping
        clipping = np.mean(np.abs(audio) > 0.95)
        
        if clipping > 0.01:
            # Muito clipping - comprimir agressivamente
            audio = self.compress(audio, threshold=0.3, ratio=4.0)
            audio = self.limit(audio, ceiling=0.9)
        else:
            # Apenas limitar
            audio = self.limit(audio, ceiling=0.95)
        
        return audio


class FeedbackLoop:
    """Loop de feedback: gera, analisa, aprende, melhora"""
    
    def __init__(self):
        self.memory = MusicMemory()
        self.analyzer = QualityAnalyzer()
        self.mixer = AutoMixer()
    
    def generate_with_feedback(self, generate_func, max_attempts=3, min_quality=0.55):
        """
        Gera música com feedback loop
        
        generate_func: função que retorna (audio, sr, metadata)
        max_attempts: máximo de tentativas
        min_quality: qualidade mínima aceitável
        """
        best_audio = None
        best_quality = 0.0
        best_metadata = None
        last_issues = []
        
        for attempt in range(max_attempts):
            print(f"  🎵 Tentativa {attempt + 1}/{max_attempts}...")
            
            # Gerar
            try:
                audio, sr, metadata = generate_func()
            except Exception as e:
                print(f"    ❌ Erro na geração: {e}")
                continue
            
            # Analisar
            acceptable, quality, details = self.analyzer.is_acceptable(audio, min_quality)
            
            print(f"    📊 Qualidade: {quality:.2f}")
            if details.get('issues'):
                print(f"    ⚠️ Problemas: {', '.join(details['issues'])}")
                last_issues = details['issues']
            
            if quality > best_quality:
                best_audio = audio
                best_quality = quality
                best_metadata = metadata
            
            if acceptable:
                print(f"    ✅ Aceita!")
                # Lembrar sucesso
                self.memory.remember_success({
                    **metadata,
                    "quality": quality
                })
                self.memory.update_stats(quality, accepted=True)
                
                # Auto-mixing final
                audio = self.mixer.fix_bad_mix(audio)
                
                return audio, sr, metadata
        
        # Nenhuma tentativa passou - usa a melhor e aprende com erro
        print(f"  ⚠️ Nenhuma tentativa atingiu qualidade mínima")
        print(f"  📝 Registrando erro para aprendizado futuro")
        
        self.memory.remember_error(
            best_metadata or {},
            reason=f"Qualidade {best_quality:.2f} < {min_quality}",
        )
        self.memory.update_stats(best_quality, accepted=False)
        
        # Ainda assim aplicar mixagem
        if best_audio is not None:
            best_audio = self.mixer.fix_bad_mix(best_audio)
        
        return best_audio, sr, best_metadata
    
    def get_report(self):
        """Relatório de aprendizado"""
        return self.memory.get_report()
