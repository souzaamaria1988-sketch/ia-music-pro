<p align="center">
  <img src="https://i.ibb.co/PGnjSZpM/2afe7a8b-8a3c-44d1-a091-e10e569a0d71.png" alt="main image" width="800">
</p>


[![Gerar Música (macOS MAX)](https://github.com/souzaamaria1988-sketch/ia-music-pro/actions/workflows/generate_music.yml/badge.svg?branch=main&event=workflow_run)](https://github.com/souzaamaria1988-sketch/ia-music-pro/actions/workflows/generate_music.yml)



[![Treinar MoE (macOS MAX - 14GB)](https://github.com/souzaamaria1988-sketch/ia-music-pro/actions/workflows/train_moe.yml/badge.svg?branch=main)](https://github.com/souzaamaria1988-sketch/ia-music-pro/actions/workflows/train_moe.yml)




# 🎵 IA Music Generator Pro - MoE + RAG

## Arquitetura

### 🧠 MoE (Mixture of Experts)
- **8 Experts** especializados (epic, dark, electronic, jazz, breakcore, ambient, rock, classical)
- **Gate Network** com Top-2 Routing
- **6 Blocos Residuais por Expert** (= 12 camadas por expert)
- **5000 Épocas** de treinamento
- Adam Optimizer + Dropout + Gradient Clipping

### 📚 RAG (Retrieval-Augmented Generation)
- **Base de Conhecimento** com 15+ padrões musicais
- **Embedding TF-IDF** para busca semântica
- **Retrieval Top-3** baseado no prompt
- **Context Injection**: escala, BPM, instrumentos, dinâmica recuperados do conhecimento

## Como Usar

### 1. TREINAR MoE
```
Actions → "Treinar MoE (5000 épocas)" → Run workflow
```
⚠️ Treino leva 2-5 horas!

### 2. GERAR com RAG
```
Actions → "Gerar Música (RAG+MoE)" → Run workflow
```

## Local (Termux)
```bash
pip install numpy scipy soundfile
python train.py --epochs 5000 --num-experts 8
python music_generator.py
```

## Estrutura
```
├── .github/workflows/
│   ├── train_moe.yml
│   └── generate_music.yml
├── models/ (modelos MoE)
│   ├── best_moe_model.npz
│   ├── final_moe_model.npz
│   └── training_log.json
├── song_output/ (músicas geradas)
│   ├── 1.wav + 1.json
│   └── ...
├── music_input/ (músicas para treino)
├── knowledge_base.json (RAG)
├── train.py (MoE training)
└── music_generator.py (RAG + geração)
```

## Estilos
epic, bossfight, dark, rock, ambient, electronic, jazz, classical, breakcore
