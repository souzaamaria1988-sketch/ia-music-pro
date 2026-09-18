#!/usr/bin/env python3
"""
🧠 TREINAMENTO MoE (Mixture of Experts)
8 Experts especializados + Gate Network
5000 épocas com skip connections
"""
import os, sys, json, time
import numpy as np
from pathlib import Path

try:
    from scipy.signal import lfilter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

MODEL_DIR = "models"
MUSIC_DIR = "music_input"
EXPERT_NAMES = ["epic", "dark", "electronic", "jazz", "breakcore", "ambient", "rock", "classical"]

class ResidualBlock:
    def __init__(self, size, seed=42):
        np.random.seed(seed)
        self.w1 = np.random.randn(size, size) * np.sqrt(2.0 / size)
        self.b1 = np.zeros(size)
        self.w2 = np.random.randn(size, size) * np.sqrt(2.0 / size)
        self.b2 = np.zeros(size)
        self.m_w1 = np.zeros_like(self.w1); self.v_w1 = np.zeros_like(self.w1)
        self.m_b1 = np.zeros_like(self.b1); self.v_b1 = np.zeros_like(self.b1)
        self.m_w2 = np.zeros_like(self.w2); self.v_w2 = np.zeros_like(self.w2)
        self.m_b2 = np.zeros_like(self.b2); self.v_b2 = np.zeros_like(self.b2)
    
    def forward(self, x, training=True):
        self.input = x
        self.z1 = x @ self.w1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        if training:
            self.mask = np.random.binomial(1, 0.9, size=self.a1.shape) / 0.9
            self.a1 *= self.mask
        self.z2 = self.a1 @ self.w2 + self.b2
        return self.z2 + x  # skip connection
    
    def backward(self, grad, lr, t, b1=0.9, b2=0.999, eps=1e-8):
        bs = grad.shape[0]
        grad_z2 = grad
        gw2 = np.clip(self.a1.T @ grad_z2 / bs, -1, 1)
        gb2 = np.mean(grad_z2, axis=0)
        ga1 = grad_z2 @ self.w2.T
        if hasattr(self, 'mask'): ga1 *= self.mask
        gz1 = ga1 * (self.z1 > 0)
        gw1 = np.clip(self.input.T @ gz1 / bs, -1, 1)
        gb1 = np.mean(gz1, axis=0)
        for param, g, m_name, v_name in [
            (self.w1, gw1, 'm_w1', 'v_w1'), (self.b1, gb1, 'm_b1', 'v_b1'),
            (self.w2, gw2, 'm_w2', 'v_w2'), (self.b2, gb2, 'm_b2', 'v_b2')]:
            m = getattr(self, m_name); v = getattr(self, v_name)
            m[:] = b1 * m + (1-b1) * g
            v[:] = b2 * v + (1-b2) * (g**2)
            param -= lr * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        return gz1 @ self.w1.T + grad

class Expert:
    def __init__(self, input_size, hidden_size, num_blocks=6, seed=42, name="expert"):
        self.name = name
        self.input_w = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.input_b = np.zeros(hidden_size)
        self.blocks = [ResidualBlock(hidden_size, seed=seed+i) for i in range(num_blocks)]
        self.output_w = np.random.randn(hidden_size, input_size) * np.sqrt(2.0/hidden_size)
        self.output_b = np.zeros(input_size)
        self.m_iw = np.zeros_like(self.input_w); self.v_iw = np.zeros_like(self.input_w)
        self.m_ib = np.zeros_like(self.input_b); self.v_ib = np.zeros_like(self.input_b)
        self.m_ow = np.zeros_like(self.output_w); self.v_ow = np.zeros_like(self.output_w)
        self.m_ob = np.zeros_like(self.output_b); self.v_ob = np.zeros_like(self.output_b)
    
    def forward(self, x, training=True):
        self.x = x
        h = np.maximum(0, x @ self.input_w + self.input_b)
        for block in self.blocks:
            h = block.forward(h, training)
        self.h = h
        return h @ self.output_w + self.output_b
    
    def backward(self, grad, lr, t):
        bs = grad.shape[0]
        gow = np.clip(self.h.T @ grad / bs, -1, 1)
        gob = np.mean(grad, axis=0)
        gh = grad @ self.output_w.T
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.m_ow[:] = b1*self.m_ow + (1-b1)*gow; self.v_ow[:] = b2*self.v_ow + (1-b2)*(gow**2)
        self.output_w -= lr * (self.m_ow/(1-b1**t)) / (np.sqrt(self.v_ow/(1-b2**t)) + eps)
        self.m_ob[:] = b1*self.m_ob + (1-b1)*gob; self.v_ob[:] = b2*self.v_ob + (1-b2)*(gob**2)
        self.output_b -= lr * (self.m_ob/(1-b1**t)) / (np.sqrt(self.v_ob/(1-b2**t)) + eps)
        for block in reversed(self.blocks):
            gh = block.backward(gh, lr, t, b1, b2, eps)
        giw = np.clip(self.x.T @ gh / bs, -1, 1)
        gib = np.mean(gh, axis=0)
        self.m_iw[:] = b1*self.m_iw + (1-b1)*giw; self.v_iw[:] = b2*self.v_iw + (1-b2)*(giw**2)
        self.input_w -= lr * (self.m_iw/(1-b1**t)) / (np.sqrt(self.v_iw/(1-b2**t)) + eps)
        self.m_ib[:] = b1*self.m_ib + (1-b1)*gib; self.v_ib[:] = b2*self.v_ib + (1-b2)*(gib**2)
        self.input_b -= lr * (self.m_ib/(1-b1**t)) / (np.sqrt(self.v_ib/(1-b2**t)) + eps)

class GateNetwork:
    def __init__(self, input_size, num_experts, seed=42):
        np.random.seed(seed)
        self.w = np.random.randn(input_size, num_experts) * np.sqrt(2.0/input_size)
        self.b = np.zeros(num_experts)
        self.m_w = np.zeros_like(self.w); self.v_w = np.zeros_like(self.w)
        self.m_b = np.zeros_like(self.b); self.v_b = np.zeros_like(self.b)
    
    def forward(self, x, top_k=2):
        self.x = x
        logits = x @ self.w + self.b
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        self.probs = probs
        # Top-k selection
        top_indices = np.argsort(probs, axis=-1)[:, -top_k:]
        gates = np.zeros_like(probs)
        for i in range(len(x)):
            gates[i, top_indices[i]] = probs[i, top_indices[i]]
        # Renormalizar
        gate_sum = np.sum(gates, axis=-1, keepdims=True) + 1e-10
        gates = gates / gate_sum
        self.gates = gates
        return gates
    
    def backward(self, grad_gates, lr, t):
        bs = grad_gates.shape[0]
        # Simplified: update gate weights based on gradient
        gw = np.clip(self.x.T @ grad_gates / bs, -1, 1)
        gb = np.mean(grad_gates, axis=0)
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.m_w[:] = b1*self.m_w + (1-b1)*gw; self.v_w[:] = b2*self.v_w + (1-b2)*(gw**2)
        self.w -= lr * (self.m_w/(1-b1**t)) / (np.sqrt(self.v_w/(1-b2**t)) + eps)
        self.m_b[:] = b1*self.m_b + (1-b1)*gb; self.v_b[:] = b2*self.v_b + (1-b2)*(gb**2)
        self.b -= lr * (self.m_b/(1-b1**t)) / (np.sqrt(self.v_b/(1-b2**t)) + eps)

class MixtureOfExperts:
    def __init__(self, input_size, hidden_size=256, num_experts=8, blocks_per_expert=6, top_k=2):
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = GateNetwork(input_size, num_experts, seed=0)
        self.experts = []
        for i in range(num_experts):
            name = EXPERT_NAMES[i] if i < len(EXPERT_NAMES) else f"expert_{i}"
            self.experts.append(Expert(input_size, hidden_size, blocks_per_expert, seed=42+i*100, name=name))
        self.t = 0
        total = self.count_params()
        print(f"🧠 MoE Inicializado:")
        print(f"   Experts: {num_experts} ({', '.join(e.name for e in self.experts)})")
        print(f"   Blocos por expert: {blocks_per_expert} (= {blocks_per_expert*2} camadas)")
        print(f"   Total camadas: {num_experts * blocks_per_expert * 2 + 2}")
        print(f"   Top-k routing: {top_k}")
        print(f"   Total parâmetros: {total:,}")
    
    def count_params(self):
        count = self.gate.w.size + self.gate.b.size
        for e in self.experts:
            count += e.input_w.size + e.input_b.size + e.output_w.size + e.output_b.size
            for b in e.blocks:
                count += b.w1.size + b.b1.size + b.w2.size + b.b2.size
        return count
    
    def forward(self, x, training=True):
        gates = self.gate.forward(x, self.top_k)
        self.active_gates = gates
        # Combinar saídas dos experts
        output = np.zeros_like(x)
        self.expert_outputs = []
        for i, expert in enumerate(self.experts):
            gate_i = gates[:, i:i+1]
            if np.sum(gate_i) > 1e-6:  # Expert ativo
                expert_out = expert.forward(x, training)
                self.expert_outputs.append((i, expert_out))
                output += expert_out * gate_i
        return output
    
    def backward(self, y, lr):
        self.t += 1
        bs = y.shape[0]
        grad = (self.forward_cache - y) * (2.0 / bs) if hasattr(self, 'forward_cache') else np.zeros_like(y)
        
        # Gate gradient (simplificado)
        gate_grad = np.zeros_like(self.active_gates)
        for i, expert_out in self.expert_outputs:
            expert_grad = np.sum((expert_out - y) ** 2, axis=1, keepdims=True)
            gate_grad[:, i:i+1] = -expert_grad * self.active_gates[:, i:i+1]
        self.gate.backward(gate_grad, lr, self.t)
        
        # Expert gradients
        for i, expert_out in self.expert_outputs:
            gate_i = self.active_gates[:, i:i+1]
            expert_grad = grad * gate_i
            self.experts[i].backward(expert_grad, lr, self.t)
    
    def train_step(self, x, y, lr):
        out = self.forward(x, training=True)
        self.forward_cache = out
        loss = np.mean((out - y) ** 2)
        self.backward(y, lr)
        return loss
    
    def save(self, filepath):
        save_dict = {
            'gate_w': self.gate.w, 'gate_b': self.gate.b,
            'input_size': np.array([self.input_size]),
            'num_experts': np.array([self.num_experts]),
            'top_k': np.array([self.top_k]),
        }
        for i, expert in enumerate(self.experts):
            save_dict[f'expert_{i}_iw'] = expert.input_w
            save_dict[f'expert_{i}_ib'] = expert.input_b
            save_dict[f'expert_{i}_ow'] = expert.output_w
            save_dict[f'expert_{i}_ob'] = expert.output_b
            for j, block in enumerate(expert.blocks):
                save_dict[f'expert_{i}_block_{j}_w1'] = block.w1
                save_dict[f'expert_{i}_block_{j}_b1'] = block.b1
                save_dict[f'expert_{i}_block_{j}_w2'] = block.w2
                save_dict[f'expert_{i}_block_{j}_b2'] = block.b2
        np.savez(filepath, **save_dict)
        print(f"💾 MoE salvo: {filepath}")

def extract_features(filepath, sr=22050, n_features=128):
    try:
        import soundfile as sf
        audio, fsr = sf.read(filepath, dtype='float32')
        if len(audio.shape) > 1: audio = np.mean(audio, axis=1)
        if fsr != sr:
            idx = np.round(np.arange(0, len(audio), fsr/sr)).astype(int)
            audio = audio[idx[idx < len(audio)]]
    except: return None
    if len(audio) < sr: return None
    features = []
    seg_len = sr
    n_segs = min(len(audio) // seg_len, 20)
    for i in range(n_segs):
        seg = audio[i*seg_len:(i+1)*seg_len]
        fft = np.abs(np.fft.rfft(seg))[:n_features]
        fft = fft / (np.max(fft) + 1e-10)
        if len(fft) < n_features: fft = np.pad(fft, (0, n_features-len(fft)))
        features.append(fft)
    return features

def generate_synthetic_data(n_samples=300, n_features=128):
    print("  Gerando dados sintéticos...")
    X = []
    for i in range(n_samples):
        freq = np.random.uniform(100, 2000)
        t = np.linspace(0, 1, 44100)
        signal = np.zeros_like(t)
        for h in range(1, np.random.randint(3, 12)):
            signal += np.sin(2*np.pi*freq*h*t) / h
        signal += np.random.randn(len(t)) * 0.1
        fft = np.abs(np.fft.rfft(signal))[:n_features]
        fft = fft / (np.max(fft) + 1e-10)
        if len(fft) < n_features: fft = np.pad(fft, (0, n_features-len(fft)))
        X.append(fft)
    return np.array(X)

def prepare_data(n_features=128):
    print("📊 Preparando dados...")
    all_features = []
    music_path = Path(MUSIC_DIR)
    if music_path.exists():
        files = []
        for ext in ['*.mp3', '*.wav', '*.flac', '*.ogg']:
            files.extend(list(music_path.glob(ext)))
        print(f"  {len(files)} músicas encontradas")
        for f in files[:30]:
            feats = extract_features(f, n_features=n_features)
            if feats: all_features.extend(feats)
    if len(all_features) < 50:
        synthetic = generate_synthetic_data(300, n_features)
        all_features.extend(synthetic.tolist())
    X = np.array(all_features)
    print(f"  Dataset: {X.shape[0]} amostras x {X.shape[1]} features")
    X_mean = np.mean(X, axis=0); X_std = np.std(X, axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    noise = np.random.randn(*X_norm.shape) * 0.15
    return X_norm + noise, X_norm, X_mean, X_std

def train(epochs=5000, num_experts=8, batch_size=32, n_features=128):
    print("=" * 60)
    print("🧠 TREINAMENTO MoE (Mixture of Experts)")
    print(f"   Épocas: {epochs} | Experts: {num_experts}")
    print("=" * 60)
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_noisy, X_clean, X_mean, X_std = prepare_data(n_features)
    model = MixtureOfExperts(n_features, hidden_size=256, num_experts=num_experts, blocks_per_expert=6, top_k=2)
    print("\n🚀 TREINANDO...")
    print("-" * 60)
    history = []; best_loss = float('inf'); start = time.time()
    lr_init = 0.001
    for epoch in range(1, epochs + 1):
        lr = max(lr_init * (0.999 ** epoch), 0.00001)
        indices = np.random.permutation(len(X_noisy))
        epoch_loss = 0; n_batches = 0
        for i in range(0, len(X_noisy), batch_size):
            idx = indices[i:i+batch_size]
            loss = model.train_step(X_noisy[idx], X_clean[idx], lr)
            epoch_loss += loss; n_batches += 1
        avg_loss = epoch_loss / n_batches
        history.append(float(avg_loss))
        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - start
            eta = (elapsed / epoch) * (epochs - epoch)
            print(f"  Epoch {epoch:5d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr:.6f} | ETA: {eta/60:.1f}min")
        if epoch % 500 == 0:
            model.save(os.path.join(MODEL_DIR, f"checkpoint_{epoch}.npz"))
            print(f"  💾 Checkpoint: epoch {epoch}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save(os.path.join(MODEL_DIR, "best_moe_model.npz"))
    model.save(os.path.join(MODEL_DIR, "final_moe_model.npz"))
    np.savez(os.path.join(MODEL_DIR, "normalization.npz"), mean=X_mean, std=X_std)
    total_time = time.time() - start
    metadata = {
        "type": "MixtureOfExperts", "epochs": epochs, "num_experts": num_experts,
        "expert_names": [e.name for e in model.experts],
        "top_k": model.top_k, "final_loss": float(history[-1]),
        "best_loss": float(best_loss), "total_params": int(model.count_params()),
        "training_time_min": float(total_time/60), "dataset_size": int(len(X_noisy)),
        "history_sample": history[::max(1, len(history)//100)]
    }
    with open(os.path.join(MODEL_DIR, "training_log.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("\n" + "=" * 60)
    print("✅ TREINAMENTO MoE CONCLUÍDO!")
    print(f"   Loss: {history[0]:.6f} → {history[-1]:.6f}")
    print(f"   Tempo: {total_time/60:.1f} min")
    print(f"   Parâmetros: {model.count_params():,}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(epochs=max(args.epochs, 5000), num_experts=max(args.num_experts, 8), batch_size=args.batch_size)
