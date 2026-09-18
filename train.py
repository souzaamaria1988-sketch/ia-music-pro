#!/usr/bin/env python3
"""
🧠 TREINAMENTO EXTREMO - 50 Camadas, 5000 Épocas
Rede neural profunda com skip connections (ResNet-like)
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

try:
    from scipy.signal import lfilter, stft
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

MODEL_DIR = "models"
MUSIC_DIR = "music_input"

# ============================================================
# REDE NEURAL PROFUNDA COM SKIP CONNECTIONS
# ============================================================

class ResidualBlock:
    """Bloco residual: permite treinar 50+ camadas sem vanishing gradient"""
    
    def __init__(self, size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # He initialization
        self.w1 = np.random.randn(size, size) * np.sqrt(2.0 / size)
        self.b1 = np.zeros(size)
        self.w2 = np.random.randn(size, size) * np.sqrt(2.0 / size)
        self.b2 = np.zeros(size)
        
        # Adam optimizer state
        self.m_w1 = np.zeros_like(self.w1)
        self.v_w1 = np.zeros_like(self.w1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_w2 = np.zeros_like(self.w2)
        self.v_w2 = np.zeros_like(self.w2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
    
    def forward(self, x, training=True):
        self.input = x
        # Primeira camada
        self.z1 = x @ self.w1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        if training:
            # Dropout 10%
            self.mask = np.random.binomial(1, 0.9, size=self.a1.shape) / 0.9
            self.a1 = self.a1 * self.mask
        # Segunda camada
        self.z2 = self.a1 @ self.w2 + self.b2
        # Skip connection (residual)
        self.output = self.z2 + x
        return self.output
    
    def backward(self, grad_output, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        batch_size = grad_output.shape[0]
        
        # Gradiente através do skip connection
        grad_z2 = grad_output
        grad_w2 = self.a1.T @ grad_z2 / batch_size
        grad_b2 = np.mean(grad_z2, axis=0)
        grad_a1 = grad_z2 @ self.w2.T
        
        if hasattr(self, 'mask'):
            grad_a1 = grad_a1 * self.mask
        
        # ReLU derivative
        grad_z1 = grad_a1 * (self.z1 > 0).astype(float)
        grad_w1 = self.input.T @ grad_z1 / batch_size
        grad_b1 = np.mean(grad_z1, axis=0)
        
        # Gradient clipping
        grad_w1 = np.clip(grad_w1, -1.0, 1.0)
        grad_w2 = np.clip(grad_w2, -1.0, 1.0)
        
        # Adam update
        self.m_w1 = beta1 * self.m_w1 + (1 - beta1) * grad_w1
        self.v_w1 = beta2 * self.v_w1 + (1 - beta2) * (grad_w1 ** 2)
        m_hat = self.m_w1 / (1 - beta1 ** t)
        v_hat = self.v_w1 / (1 - beta2 ** t)
        self.w1 -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        self.m_b1 = beta1 * self.m_b1 + (1 - beta1) * grad_b1
        self.v_b1 = beta2 * self.v_b1 + (1 - beta2) * (grad_b1 ** 2)
        self.b1 -= lr * (self.m_b1 / (1 - beta1 ** t)) / (np.sqrt(self.v_b1 / (1 - beta2 ** t)) + eps)
        
        self.m_w2 = beta1 * self.m_w2 + (1 - beta1) * grad_w2
        self.v_w2 = beta2 * self.v_w2 + (1 - beta2) * (grad_w2 ** 2)
        m_hat = self.m_w2 / (1 - beta1 ** t)
        v_hat = self.v_w2 / (1 - beta2 ** t)
        self.w2 -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        self.m_b2 = beta1 * self.m_b2 + (1 - beta1) * grad_b2
        self.v_b2 = beta2 * self.v_b2 + (1 - beta2) * (grad_b2 ** 2)
        self.b2 -= lr * (self.m_b2 / (1 - beta1 ** t)) / (np.sqrt(self.v_b2 / (1 - beta2 ** t)) + eps)
        
        # Gradiente para próxima camada (input gradient)
        grad_input = grad_z1 @ self.w1.T + grad_output  # skip connection gradient
        return grad_input


class DeepMusicNet:
    """Rede neural profunda para música com 50 camadas"""
    
    def __init__(self, input_size, hidden_size, num_layers=50, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        np.random.seed(seed)
        
        # Camada de entrada: input_size -> hidden_size
        self.input_w = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.input_b = np.zeros(hidden_size)
        
        # 50 blocos residuais (cada bloco = 2 camadas, então 50 blocos = 100 camadas efetivas)
        # Para ter exatamente 50 camadas, usamos 25 blocos residuais
        self.num_blocks = num_layers // 2
        self.blocks = []
        for i in range(self.num_blocks):
            self.blocks.append(ResidualBlock(hidden_size, seed=seed + i))
        
        # Camada de saída: hidden_size -> input_size
        self.output_w = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / hidden_size)
        self.output_b = np.zeros(input_size)
        
        # Adam state para input/output
        self.m_iw = np.zeros_like(self.input_w)
        self.v_iw = np.zeros_like(self.input_w)
        self.m_ib = np.zeros_like(self.input_b)
        self.v_ib = np.zeros_like(self.input_b)
        self.m_ow = np.zeros_like(self.output_w)
        self.v_ow = np.zeros_like(self.output_w)
        self.m_ob = np.zeros_like(self.output_b)
        self.v_ob = np.zeros_like(self.output_b)
        
        self.t = 0
        
        total_params = self.count_params()
        print(f"🧠 DeepMusicNet inicializada:")
        print(f"   Input: {input_size}")
        print(f"   Hidden: {hidden_size}")
        print(f"   Blocos residuais: {self.num_blocks} (= {self.num_blocks * 2} camadas)")
        print(f"   Total camadas efetivas: {self.num_blocks * 2 + 2}")
        print(f"   Total parâmetros: {total_params:,}")
    
    def count_params(self):
        count = self.input_w.size + self.input_b.size
        count += self.output_w.size + self.output_b.size
        for block in self.blocks:
            count += block.w1.size + block.b1.size + block.w2.size + block.b2.size
        return count
    
    def forward(self, x, training=True):
        self.x = x
        # Input layer
        self.h = x @ self.input_w + self.input_b
        self.h = np.maximum(0, self.h)  # ReLU
        
        # Residual blocks
        for block in self.blocks:
            self.h = block.forward(self.h, training)
        
        # Output layer
        self.out = self.h @ self.output_w + self.output_b
        return self.out
    
    def backward(self, y, lr):
        self.t += 1
        batch_size = y.shape[0]
        
        # Output gradient
        grad_out = (self.out - y) * (2.0 / batch_size)
        
        grad_ow = self.h.T @ grad_out / batch_size
        grad_ob = np.mean(grad_out, axis=0)
        grad_h = grad_out @ self.output_w.T
        
        grad_ow = np.clip(grad_ow, -1.0, 1.0)
        
        # Adam para output
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.m_ow = beta1 * self.m_ow + (1-beta1) * grad_ow
        self.v_ow = beta2 * self.v_ow + (1-beta2) * (grad_ow**2)
        self.output_w -= lr * (self.m_ow/(1-beta1**self.t)) / (np.sqrt(self.v_ow/(1-beta2**self.t)) + eps)
        self.m_ob = beta1 * self.m_ob + (1-beta1) * grad_ob
        self.v_ob = beta2 * self.v_ob + (1-beta2) * (grad_ob**2)
        self.output_b -= lr * (self.m_ob/(1-beta1**self.t)) / (np.sqrt(self.v_ob/(1-beta2**self.t)) + eps)
        
        # Backward through residual blocks (reversed)
        for block in reversed(self.blocks):
            grad_h = block.backward(grad_h, lr, self.t, beta1, beta2, eps)
        
        # Input layer gradient
        grad_h_relu = grad_h * (self.h > 0).astype(float) if hasattr(self, 'h') else grad_h
        # Note: h já passou por ReLU, precisamos do pré-ativação
        grad_iw = self.x.T @ grad_h_relu / batch_size if grad_h_relu.shape == self.h.shape else self.x.T @ grad_h / batch_size
        grad_ib = np.mean(grad_h_relu, axis=0) if grad_h_relu.shape == self.h.shape else np.mean(grad_h, axis=0)
        
        grad_iw = np.clip(grad_iw, -1.0, 1.0)
        
        self.m_iw = beta1 * self.m_iw + (1-beta1) * grad_iw
        self.v_iw = beta2 * self.v_iw + (1-beta2) * (grad_iw**2)
        self.input_w -= lr * (self.m_iw/(1-beta1**self.t)) / (np.sqrt(self.v_iw/(1-beta2**self.t)) + eps)
        self.m_ib = beta1 * self.m_ib + (1-beta1) * grad_ib
        self.v_ib = beta2 * self.v_ib + (1-beta2) * (grad_ib**2)
        self.input_b -= lr * (self.m_ib/(1-beta1**self.t)) / (np.sqrt(self.v_ib/(1-beta2**self.t)) + eps)
    
    def train_step(self, x, y, lr):
        out = self.forward(x, training=True)
        loss = np.mean((out - y) ** 2)
        self.backward(y, lr)
        return loss
    
    def save(self, filepath):
        save_dict = {
            'input_w': self.input_w,
            'input_b': self.input_b,
            'output_w': self.output_w,
            'output_b': self.output_b,
            'input_size': np.array([self.input_size]),
            'hidden_size': np.array([self.hidden_size]),
            'num_layers': np.array([self.num_layers]),
        }
        for i, block in enumerate(self.blocks):
            save_dict[f'block_{i}_w1'] = block.w1
            save_dict[f'block_{i}_b1'] = block.b1
            save_dict[f'block_{i}_w2'] = block.w2
            save_dict[f'block_{i}_b2'] = block.b2
        np.savez(filepath, **save_dict)
        print(f"💾 Modelo salvo: {filepath}")


# ============================================================
# EXTRAÇÃO DE FEATURES DE ÁUDIO
# ============================================================

def extract_features_from_audio(filepath, sr=22050, n_features=128):
    """Extrai features espectrais de um arquivo de áudio"""
    try:
        import soundfile as sf
        audio, file_sr = sf.read(filepath, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        # Resample se necessário
        if file_sr != sr:
            indices = np.round(np.arange(0, len(audio), file_sr / sr)).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]
    except Exception as e:
        print(f"  ⚠️ Erro ao ler {filepath}: {e}")
        return None
    
    if len(audio) < sr:  # mínimo 1 segundo
        return None
    
    # Extrair múltiplos segmentos
    features = []
    segment_length = sr  # 1 segundo por segmento
    n_segments = min(len(audio) // segment_length, 20)  # máximo 20 segmentos
    
    for i in range(n_segments):
        start = i * segment_length
        segment = audio[start:start + segment_length]
        
        # FFT features
        fft = np.abs(np.fft.rfft(segment))[:n_features]
        # Normalizar
        fft = fft / (np.max(fft) + 1e-10)
        
        # Completar se necessário
        if len(fft) < n_features:
            fft = np.pad(fft, (0, n_features - len(fft)))
        
        features.append(fft)
    
    return features


def generate_synthetic_data(n_samples=200, n_features=128):
    """Gera dados sintéticos se não houver músicas"""
    print("  Gerando dados sintéticos para treinamento...")
    X = []
    for i in range(n_samples):
        # Gerar padrões espectrais variados
        freq_base = np.random.uniform(100, 2000)
        t = np.linspace(0, 1, 44100)
        
        # Combinação de harmônicos
        signal = np.zeros_like(t)
        n_harmonics = np.random.randint(3, 12)
        for h in range(1, n_harmonics + 1):
            signal += np.sin(2 * np.pi * freq_base * h * t) / h
        
        # Adicionar ruído
        signal += np.random.randn(len(t)) * 0.1
        
        # FFT
        fft = np.abs(np.fft.rfft(signal))[:n_features]
        fft = fft / (np.max(fft) + 1e-10)
        
        if len(fft) < n_features:
            fft = np.pad(fft, (0, n_features - len(fft)))
        
        X.append(fft)
    
    return np.array(X)


def prepare_training_data(n_features=128):
    """Prepara dados de treinamento"""
    print("📊 Preparando dados de treinamento...")
    
    all_features = []
    
    # Tentar ler músicas de music_input/
    music_path = Path(MUSIC_DIR)
    if music_path.exists():
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.flac', '*.ogg']:
            audio_files.extend(list(music_path.glob(ext)))
        
        print(f"  Encontradas {len(audio_files)} músicas")
        
        for filepath in audio_files[:30]:  # máximo 30 arquivos
            print(f"  Processando: {filepath.name}")
            features = extract_features_from_audio(filepath, n_features=n_features)
            if features:
                all_features.extend(features)
    
    # Se não houver dados suficientes, gerar sintéticos
    if len(all_features) < 50:
        print(f"  Poucos dados ({len(all_features)}). Completando com sintéticos...")
        synthetic = generate_synthetic_data(n_samples=200, n_features=n_features)
        all_features.extend(synthetic.tolist())
    
    X = np.array(all_features)
    print(f"  Dataset final: {X.shape[0]} amostras x {X.shape[1]} features")
    
    # Normalizar
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_normalized = (X - X_mean) / X_std
    
    # Autoencoder: input = output (aprender representação)
    # Adicionar ruído para denoising autoencoder
    noise = np.random.randn(*X_normalized.shape) * 0.15
    X_noisy = X_normalized + noise
    
    return X_noisy, X_normalized, X_mean, X_std


# ============================================================
# TREINAMENTO PRINCIPAL
# ============================================================

def train(epochs=5000, num_layers=50, batch_size=32, n_features=128):
    """Treinamento principal"""
    print("=" * 60)
    print("🧠 TREINAMENTO EXTREMO")
    print(f"   Épocas: {epochs}")
    print(f"   Camadas: {num_layers}")
    print(f"   Batch size: {batch_size}")
    print("=" * 60)
    print()
    
    # Criar pasta de modelos
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Preparar dados
    X_noisy, X_clean, X_mean, X_std = prepare_training_data(n_features)
    
    # Criar modelo
    hidden_size = 256  # Tamanho das camadas ocultas
    model = DeepMusicNet(n_features, hidden_size, num_layers=num_layers)
    
    print()
    print("🚀 INICIANDO TREINAMENTO...")
    print("-" * 60)
    
    # Training loop
    history = []
    best_loss = float('inf')
    start_time = time.time()
    
    # Learning rate schedule
    initial_lr = 0.001
    
    for epoch in range(1, epochs + 1):
        # Learning rate decay
        lr = initial_lr * (0.999 ** epoch)
        lr = max(lr, 0.00001)  # mínimo
        
        # Shuffle
        indices = np.random.permutation(len(X_noisy))
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_noisy), batch_size):
            batch_idx = indices[i:i + batch_size]
            x_batch = X_noisy[batch_idx]
            y_batch = X_clean[batch_idx]
            
            loss = model.train_step(x_batch, y_batch, lr)
            epoch_loss += loss
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        history.append(float(avg_loss))
        
        # Log a cada 50 épocas
        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            eta = (elapsed / epoch) * (epochs - epoch)
            print(f"  Epoch {epoch:5d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr:.6f} | ETA: {eta/60:.1f}min")
        
        # Checkpoint a cada 500 épocas
        if epoch % 500 == 0:
            checkpoint_path = os.path.join(MODEL_DIR, f"checkpoint_epoch_{epoch}.npz")
            model.save(checkpoint_path)
            print(f"  💾 Checkpoint salvo: epoch {epoch}")
        
        # Salvar melhor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(MODEL_DIR, "best_model.npz")
            model.save(best_path)
    
    # Salvar modelo final
    final_path = os.path.join(MODEL_DIR, "final_model.npz")
    model.save(final_path)
    
    # Salvar metadata
    total_time = time.time() - start_time
    metadata = {
        "epochs_trained": epochs,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "final_loss": float(history[-1]),
        "best_loss": float(best_loss),
        "initial_loss": float(history[0]),
        "improvement": float(history[0] - history[-1]),
        "total_parameters": int(model.count_params()),
        "training_time_seconds": float(total_time),
        "training_time_minutes": float(total_time / 60),
        "dataset_size": int(len(X_noisy)),
        "n_features": n_features,
        "history_sample": history[::max(1, len(history)//100)]  # amostra da história
    }
    
    with open(os.path.join(MODEL_DIR, "training_log.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Salvar normalização (necessário para geração)
    np.savez(os.path.join(MODEL_DIR, "normalization.npz"), mean=X_mean, std=X_std)
    
    print()
    print("=" * 60)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print(f"   Loss inicial: {history[0]:.6f}")
    print(f"   Loss final:   {history[-1]:.6f}")
    print(f"   Melhor loss:  {best_loss:.6f}")
    print(f"   Tempo total:  {total_time/60:.1f} minutos")
    print(f"   Parâmetros:   {model.count_params():,}")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--features", type=int, default=128)
    args = parser.parse_args()
    
    # Garantir mínimo de 5000 épocas
    epochs = max(args.epochs, 5000)
    layers = max(args.layers, 50)
    
    train(epochs=epochs, num_layers=layers, batch_size=args.batch_size, n_features=args.features)
