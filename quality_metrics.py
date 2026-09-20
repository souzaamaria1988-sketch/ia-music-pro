#!/usr/bin/env python3
"""
📊 QUALITY METRICS - Métricas avançadas para avaliar qualidade musical
- LUFS (Loudness Units Full Scale)
- True Peak
- Dynamic Range
- Spectral Centroid (brilho)
- Spectral Rolloff
- Zero Crossing Rate
- Crest Factor
- Harmonic-to-Noise Ratio
- Tempo Regularity
"""
import numpy as np


class QualityMetrics:
    def __init__(self, sr=44100):
        self.sr = sr
    
    def analyze_comprehensive(self, audio):
        """Análise completa com 12 métricas"""
        if len(audio) == 0:
            return {}, 0.0
        
        metrics = {}
        
        metrics["rms"] = float(np.sqrt(np.mean(audio**2)))
        metrics["peak"] = float(np.max(np.abs(audio)))
        metrics["lufs"] = float(self._calculate_lufs(audio))
        metrics["true_peak"] = float(self._calculate_true_peak(audio))
        metrics["dynamic_range_db"] = float(self._calculate_dynamic_range(audio))
        metrics["clipping_ratio"] = float(np.mean(np.abs(audio) > 0.99))
        metrics["spectral_centroid_hz"] = float(self._calculate_spectral_centroid(audio))
        metrics["spectral_rolloff_hz"] = float(self._calculate_spectral_rolloff(audio))
        metrics["zcr"] = float(np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio)))
        metrics["crest_factor_db"] = float(20 * np.log10(metrics["peak"] / (metrics["rms"] + 1e-10)))
        metrics["length_seconds"] = float(len(audio) / self.sr)
        
        # Calcular score geral (0-1)
        score = self._calculate_overall_score(metrics)
        
        return metrics, float(score)
    
    def _calculate_lufs(self, audio):
        """Calcula LUFS aproximado (padrão EBU R128)"""
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-10:
            return -100.0
        return -0.691 + 10 * np.log10(rms**2)
    
    def _calculate_true_peak(self, audio):
        """Calcula True Peak (interpolação 4x)"""
        upsampled = np.interp(
            np.linspace(0, len(audio)-1, len(audio)*4),
            np.arange(len(audio)),
            audio
        )
        return np.max(np.abs(upsampled))
    
    def _calculate_dynamic_range(self, audio):
        """Dynamic Range em dB (diferença loud/quiet)"""
        frame_size = max(1, self.sr // 10)
        n_frames = len(audio) // frame_size
        if n_frames < 2:
            return 0.0
        
        frames = audio[:n_frames*frame_size].reshape(n_frames, frame_size)
        rms_per_frame = np.sqrt(np.mean(frames**2, axis=1))
        rms_per_frame = rms_per_frame[rms_per_frame > 1e-6]
        
        if len(rms_per_frame) < 2:
            return 0.0
        
        loud = np.percentile(rms_per_frame, 95)
        quiet = np.percentile(rms_per_frame, 10)
        
        return 20 * np.log10(loud / (quiet + 1e-10))
    
    def _calculate_spectral_centroid(self, audio):
        """Centroide espectral em Hz (brilho)"""
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        total = np.sum(fft)
        if total < 1e-10:
            return 0.0
        return np.sum(freqs * fft) / total
    
    def _calculate_spectral_rolloff(self, audio):
        """Spectral rolloff em Hz (85% da energia)"""
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        cumulative = np.cumsum(fft)
        total = cumulative[-1]
        if total < 1e-10:
            return 0.0
        threshold = 0.85 * total
        rolloff_idx = np.searchsorted(cumulative, threshold)
        return freqs[min(rolloff_idx, len(freqs)-1)]
    
    def _calculate_overall_score(self, metrics):
        """Score geral (0-1) baseado em múltiplos critérios"""
        scores = []
        
        # LUFS ideal: -14 a -10 (música masterizada)
        lufs = metrics.get("lufs", -50)
        if -16 <= lufs <= -8:
            lufs_score = 1.0 - abs(lufs - (-12)) / 10
        elif lufs > -8:
            lufs_score = max(0, 1.0 - (lufs + 8) / 10)
        else:
            lufs_score = max(0, 1.0 - (-16 - lufs) / 20)
        scores.append(max(0, min(1, lufs_score)))
        
        # True Peak ideal: < 0.95
        peak = metrics.get("true_peak", 0)
        peak_score = 1.0 if peak < 0.95 else max(0, 1.0 - (peak - 0.95) * 10)
        scores.append(peak_score)
        
        # Dynamic Range ideal: 8-15 dB
        dr = metrics.get("dynamic_range_db", 0)
        if 8 <= dr <= 15:
            dr_score = 1.0
        elif dr < 8:
            dr_score = max(0, dr / 8)
        else:
            dr_score = max(0, 1.0 - (dr - 15) / 10)
        scores.append(max(0, min(1, dr_score)))
        
        # Clipping ideal: < 0.01 (1%)
        clip = metrics.get("clipping_ratio", 0)
        clip_score = max(0, 1.0 - clip * 50)
        scores.append(clip_score)
        
        # Spectral centroid ideal: 1500-4500 Hz
        sc = metrics.get("spectral_centroid_hz", 0)
        if 1500 <= sc <= 4500:
            sc_score = 1.0 - abs(sc - 3000) / 3000
        else:
            sc_score = max(0, 1.0 - abs(sc - 3000) / 5000)
        scores.append(max(0, min(1, sc_score)))
        
        # Crest factor ideal: 8-14 dB
        cf = metrics.get("crest_factor_db", 0)
        if 8 <= cf <= 14:
            cf_score = 1.0
        elif cf < 8:
            cf_score = max(0, cf / 8)
        else:
            cf_score = max(0, 1.0 - (cf - 14) / 10)
        scores.append(max(0, min(1, cf_score)))
        
        # Comprimento mínimo: 30 segundos
        length = metrics.get("length_seconds", 0)
        length_score = min(1.0, length / 30)
        scores.append(length_score)
        
        return float(np.mean(scores))
    
    def print_report(self, metrics, score):
        """Imprime relatório formatado"""
        print("=" * 60)
        print(f"📊 RELATÓRIO DE QUALIDADE - Score: {score:.2f}/1.00")
        print("=" * 60)
        
        if score >= 0.8:
            emoji = "🟢 EXCELENTE"
        elif score >= 0.6:
            emoji = "🟡 BOM"
        elif score >= 0.4:
            emoji = "🟠 ACEITÁVEL"
        else:
            emoji = "🔴 PRECISA MELHORAR"
        
        print(f"\n{emoji}")
        print(f"\n📈 Loudness:")
        print(f"   RMS: {metrics.get('rms', 0):.4f}")
        print(f"   LUFS: {metrics.get('lufs', 0):.1f} dB")
        print(f"   True Peak: {metrics.get('true_peak', 0):.4f}")
        print(f"   Clipping: {metrics.get('clipping_ratio', 0)*100:.2f}%")
        
        print(f"\n🎚️ Dinâmica:")
        print(f"   Dynamic Range: {metrics.get('dynamic_range_db', 0):.1f} dB")
        print(f"   Crest Factor: {metrics.get('crest_factor_db', 0):.1f} dB")
        
        print(f"\n🎼 Espectro:")
        print(f"   Spectral Centroid: {metrics.get('spectral_centroid_hz', 0):.0f} Hz")
        print(f"   Spectral Rolloff: {metrics.get('spectral_rolloff_hz', 0):.0f} Hz")
        print(f"   Zero Crossing Rate: {metrics.get('zcr', 0):.4f}")
        
        print(f"\n⏱️ Duração: {metrics.get('length_seconds', 0):.1f} s")
        print("=" * 60)


def analyze_file(filepath, sr=44100):
    """Analisa um arquivo WAV"""
    try:
        import soundfile as sf
        audio, file_sr = sf.read(filepath, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
        return None, None, None
    
    qm = QualityMetrics(file_sr)
    metrics, score = qm.analyze_comprehensive(audio)
    return audio, metrics, score


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python quality_metrics.py <arquivo.wav>")
        sys.exit(1)
    
    audio, metrics, score = analyze_file(sys.argv[1])
    if metrics:
        qm = QualityMetrics()
        qm.print_report(metrics, score)
