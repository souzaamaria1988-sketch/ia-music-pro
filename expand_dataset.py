#!/usr/bin/env python3
"""
📊 EXPAND DATASET - Organiza e expande o dataset de treinamento
- Cria subpastas por estilo em music_input/
- Gera amostras sintéticas adicionais
- Validação do dataset
"""
import os
import sys
import numpy as np
from pathlib import Path


STYLES = [
    "cinematic", "epic", "ambient", "electronic", "jazz",
    "rock", "classical", "folk", "latin", "breakcore",
    "pop", "dark", "bossfight"
]


def organize_music_input():
    """Cria subpastas em music_input/ por estilo"""
    music_dir = Path("music_input")
    music_dir.mkdir(exist_ok=True)
    
    print("📁 Organizando music_input/ em subpastas por estilo...")
    for style in STYLES:
        style_dir = music_dir / style
        style_dir.mkdir(exist_ok=True)
        existing = list(style_dir.glob("*"))
        print(f"   ✓ music_input/{style}/ ({len(existing)} arquivos)")
    
    # Contar arquivos soltos (não em subpastas)
    loose_files = [f for f in music_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in [".mp3", ".wav", ".flac", ".ogg"]]
    if loose_files:
        print(f"\n⚠️  {len(loose_files)} arquivos soltos em music_input/")
        print("   Considere movê-los para subpastas apropriadas")
    
    return True


def generate_more_synthetic(n_samples=3000):
    """Gera mais dados sintéticos variados"""
    print(f"\n🎵 Gerando {n_samples} amostras sintéticas...")
    
    from train import generate_synthetic_data
    
    synthetic = generate_synthetic_data(n_samples, n_features=256)
    
    out_path = Path("music_input/synthetic_expanded.npy")
    np.save(out_path, synthetic)
    print(f"✅ Salvo em {out_path}")
    print(f"   Shape: {synthetic.shape}")
    
    return synthetic


def validate_dataset():
    """Valida o dataset atual"""
    print("\n🔍 Validando dataset...")
    
    music_dir = Path("music_input")
    total_files = 0
    style_counts = {}
    
    for style in STYLES:
        style_dir = music_dir / style
        if style_dir.exists():
            count = len(list(style_dir.glob("*.mp3"))) + len(list(style_dir.glob("*.wav"))) + \
                    len(list(style_dir.glob("*.flac"))) + len(list(style_dir.glob("*.ogg")))
            style_counts[style] = count
            total_files += count
    
    print(f"\n📊 Arquivos por estilo:")
    for style, count in sorted(style_counts.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * min(count // 2, 30)
        print(f"   {style:15s}: {count:3d} {bar}")
    
    print(f"\n   Total: {total_files} arquivos reais")
    
    # Verificar arquivo sintético
    synth_path = music_dir / "synthetic_expanded.npy"
    if synth_path.exists():
        synth = np.load(synth_path)
        print(f"   Sintético: {synth.shape[0]} amostras")
        total_files += synth.shape[0]
    
    # Recomendações
    print(f"\n💡 Recomendações:")
    if total_files < 500:
        print("   ⚠️  Dataset muito pequeno! Adicione mais músicas.")
        print("      Recomendado: 2000+ amostras para autoencoder robusto.")
    elif total_files < 2000:
        print("   🟡 Dataset razoável. Considere adicionar mais 1000+ músicas.")
    else:
        print("   ✅ Dataset adequado para treinamento!")
    
    # Estilos faltando
    missing_styles = [s for s, c in style_counts.items() if c == 0]
    if missing_styles:
        print(f"\n   ⚠️  Estilos sem músicas: {', '.join(missing_styles)}")
        print(f"      Adicione pelo menos 10 músicas em cada.")
    
    return total_files


def main():
    print("=" * 60)
    print("📊 EXPAND DATASET")
    print("=" * 60)
    
    # Passo 1: Organizar
    organize_music_input()
    
    # Passo 2: Gerar sintético
    response = input("\nGerar 3000 amostras sintéticas? (s/n): ").strip().lower()
    if response == "s":
        generate_more_synthetic(3000)
    
    # Passo 3: Validar
    total = validate_dataset()
    
    print("\n" + "=" * 60)
    print("✅ PRÓXIMOS PASSOS:")
    print("=" * 60)
    print("1. Adicione músicas REAIS em music_input/<estilo>/")
    print("   Exemplo: music_input/cinematic/minha_musica.mp3")
    print("\n2. Retreine o autoencoder:")
    print("   python train.py --epochs 800 --batch-size 32")
    print("\n3. O treino vai parar automaticamente (early stopping)")
    print("   quando validation loss estagnar (~200-400 epochs)")
    print("\n4. Teste geração:")
    print("   python music_generator.py --style cinematic --duration 45")
    print("=" * 60)


if __name__ == "__main__":
    main()
