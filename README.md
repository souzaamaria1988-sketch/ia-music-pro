# 🎵 IA Music Generator Pro - TREINO EXTREMO

## Modelo
- **50 Camadas Neurais** (blocos residuais)
- **5.000 Épocas** de treinamento
- **~15-20 milhões de parâmetros**
- Skip connections (ResNet-like)
- Adam Optimizer + Learning Rate Decay

## Como Usar

### 1. TREINAR (primeiro!)
```
Actions → "Treinar Modelo (5000 épocas)" → Run workflow
```
⚠️ **Treino leva 2-5 horas!**

### 2. GERAR MÚSICAS
```
Actions → "Gerar Música" → Run workflow
```
Músicas aparecem em `song_output/` numeradas (1.wav, 2.wav, 3.wav...)

## Local (Termux)
```bash
pip install numpy scipy soundfile
python train.py --epochs 5000 --layers 50
python music_generator.py
```

## Estrutura
```
├── .github/workflows/
│   ├── train_model.yml (treino 5000 épocas)
│   └── generate_music.yml (geração)
├── models/ (modelos treinados)
│   ├── best_model.npz
│   ├── final_model.npz
│   └── training_log.json
├── song_output/ (músicas geradas)
│   ├── 1.wav + 1.json
│   ├── 2.wav + 2.json
│   └── ...
├── music_input/ (coloque músicas aqui)
├── train.py (treinamento 50 camadas)
└── music_generator.py (geração)
```

## Estilos
epic, bossfight, dark, rock, ambient, electronic, jazz, classical, breakcore
