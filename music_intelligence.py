"""Inteligência musical: memória, análise de qualidade, mixagem e feedback loop."""
from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)
MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(exist_ok=True)


class MusicMemory:
    def __init__(self, memory_file="memory/music_memory.json"):
        self.memory_file = Path(memory_file)
        self.data = self._load()

    def _load(self):
        if self.memory_file.is_file():
            try:
                return json.loads(self.memory_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("Arquivo de memória corrompido")
        return {"generations": [], "successes": [], "failures": []}

    def save(self):
        self.memory_file.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def record_generation(self, metadata, quality_score):
        self.data["generations"].append({**metadata, "quality_score": quality_score})
        if quality_score > 0.7:
            self.data["successes"].append(metadata)
        else:
            self.data["failures"].append(metadata)
        self.save()


class QualityAnalyzer:
    def __init__(self):
        self.issues = []

    def analyze_wav(self, wav_path):
        try:
            import soundfile as sf
            data, samplerate = sf.read(str(wav_path))
        except Exception as e:
            log.error("Não foi possível ler o arquivo WAV: %s", e)
            return {"score": 0.0, "issues": ["arquivo ilegível"]}

        self.issues = []
        score = 1.0
        if len(data) > 0:
            max_val = np.max(np.abs(data))
            if max_val >= 0.99:
                self.issues.append("clipping detectado"); score -= 0.3
            rms = np.sqrt(np.mean(data ** 2))
            if rms < 0.01:
                self.issues.append("volume muito baixo"); score -= 0.2
            silence_ratio = np.mean(np.abs(data) < 0.001)
            if silence_ratio > 0.8:
                self.issues.append("silêncio excessivo"); score -= 0.3
        return {
            "score": max(0.0, score),
            "issues": self.issues.copy(),
            "duration": len(data) / samplerate if len(data) > 0 else 0,
            "sample_rate": samplerate,
        }


class AutoMixer:
    def normalize(self, audio_data, target_db=-3.0):
        if len(audio_data) == 0:
            return audio_data
        max_val = np.max(np.abs(audio_data))
        if max_val == 0:
            return audio_data
        target_linear = 10 ** (target_db / 20.0)
        gain = target_linear / max_val
        return audio_data * gain


class FeedbackLoop:
    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts
        self.memory = MusicMemory()
        self.analyzer = QualityAnalyzer()
        self.mixer = AutoMixer()
        self.history = []

    def run(self, generate_func, base_params):
        best_result = None
        best_score = -1.0
        for attempt in range(self.max_attempts):
            log.info("Tentativa %d/%d", attempt + 1, self.max_attempts)
            params = base_params.copy()
            try:
                result = generate_func(params)
            except Exception as e:
                log.error("Erro na geração: %s", e)
                continue
            wav_path = result.get("wav_file")
            if wav_path and Path(wav_path).is_file():
                quality = self.analyzer.analyze_wav(wav_path)
            else:
                quality = {"score": 0.0, "issues": ["WAV não gerado"]}
            result["quality"] = quality
            self.history.append(result)
            self.memory.record_generation(result, quality["score"])
            if quality["score"] > best_score:
                best_score = quality["score"]
                best_result = result
            if quality["score"] >= 0.8:
                break
        return best_result or {}
