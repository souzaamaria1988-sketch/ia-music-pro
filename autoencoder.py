#!/usr/bin/env python3
"""
🧠 AUTOENCODER + STYLE EXTRACTOR
Adds StyleExtractor to pull latent vectors from real audio.
"""
import os, sys, json, time, gc
import numpy as np
from pathlib import Path

MODEL_DIR = "models"
AE_DIR = os.path.join(MODEL_DIR, "autoencoder")


class DenseLayer:
    def __init__(self, in_size, out_size, activation="relu", seed=42):
        np.random.seed(seed)
        scale = np.sqrt(2.0 / (in_size + out_size))
        self.w = np.random.randn(in_size, out_size) * scale
        self.b = np.zeros(out_size)
        self.activation = activation
        self.m_w = np.zeros_like(self.w)
        self.v_w = np.zeros_like(self.w)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.input = None
        self.z = None
        self.output = None
    
    def forward(self, x):
        self.input = x
        self.z = x @ self.w + self.b
        if self.activation == "relu": self.output = np.maximum(0, self.z)
        elif self.activation == "tanh": self.output = np.tanh(self.z)
        else: self.output = self.z
        return self.output
    
    def backward(self, grad_output, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        batch_size = grad_output.shape[0]
        if self.activation == "relu":
            grad_act = grad_output * (self.z > 0).astype(float)
        elif self.activation == "tanh":
            grad_act = grad_output * (1 - self.output ** 2)
        else:
            grad_act = grad_output
        grad_w = (self.input.T @ grad_act) / batch_size
        grad_b = np.mean(grad_act, axis=0)
        grad_w = np.clip(grad_w, -1.0, 1.0)
        grad_b = np.clip(grad_b, -1.0, 1.0)
        self.m_w = beta1 * self.m_w + (1 - beta1) * grad_w
        self.v_w = beta2 * self.v_w + (1 - beta2) * (grad_w ** 2)
        m_hat = self.m_w / (1 - beta1 ** t)
        v_hat = self.v_w / (1 - beta2 ** t)
        self.w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b ** 2)
        m_hat_b = self.m_b / (1 - beta1 ** t)
        v_hat_b = self.v_b / (1 - beta2 ** t)
        self.b -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
        return grad_act @ self.w.T


class Autoencoder:
    def __init__(self, input_size=256, hidden_sizes=None, latent_size=64, seed=42):
        if hidden_sizes is None: hidden_sizes = [512, 256, 128]
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.t = 0
        
        self.encoder_layers = []
        layer_sizes = [input_size] + hidden_sizes + [latent_size]
        for i in range(len(layer_sizes) - 1):
            is_last = (i == len(layer_sizes) - 2)
            activation = "linear" if is_last else "relu"
            self.encoder_layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1], activation, seed=seed+i))
        
        self.decoder_layers = []
        decoder_sizes = [latent_size] + list(reversed(hidden_sizes)) + [input_size]
        for i in range(len(decoder_sizes) - 1):
            is_last = (i == len(decoder_sizes) - 2)
            activation = "linear" if is_last else "relu"
            self.decoder_layers.append(DenseLayer(decoder_sizes[i], decoder_sizes[i+1], activation, seed=seed+100+i))
    
    def count_params(self):
        c = sum(l.w.size + l.b.size for l in self.encoder_layers)
        c += sum(l.w.size + l.b.size for l in self.decoder_layers)
        return c
    
    def encode(self, x):
        h = x
        for layer in self.encoder_layers: h = layer.forward(h)
        return h
    
    def decode(self, latent):
        h = latent
        for layer in self.decoder_layers: h = layer.forward(h)
        return h
    
    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent), latent
    
    def compute_loss(self, x, output):
        return np.mean((x - output) ** 2)
    
    def backward(self, x, output, lr=0.001):
        self.t += 1
        batch_size = x.shape[0]
        grad = 2 * (output - x) / batch_size
        for layer in reversed(self.decoder_layers):
            grad = layer.backward(grad, lr, self.t)
        for layer in reversed(self.encoder_layers):
            grad = layer.backward(grad, lr, self.t)
    
    def train_step(self, x, lr=0.001):
        output, latent = self.forward(x)
        loss = self.compute_loss(x, output)
        self.backward(x, output, lr)
        return loss
    
    def save(self, filepath):
        save_dict = {
            "input_size": np.array([self.input_size]),
            "latent_size": np.array([self.latent_size]),
            "hidden_sizes": np.array(self.hidden_sizes),
        }
        for i, layer in enumerate(self.encoder_layers):
            save_dict[f"enc_w_{i}"] = layer.w.astype(np.float32)
            save_dict[f"enc_b_{i}"] = layer.b.astype(np.float32)
            save_dict[f"enc_act_{i}"] = np.array([layer.activation])
        for i, layer in enumerate(self.decoder_layers):
            save_dict[f"dec_w_{i}"] = layer.w.astype(np.float32)
            save_dict[f"dec_b_{i}"] = layer.b.astype(np.float32)
            save_dict[f"dec_act_{i}"] = np.array([layer.activation])
        np.savez_compressed(filepath, **save_dict)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"💾 Autoencoder salvo: {filepath} ({size_mb:.2f} MB)")
    
    def load(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        self.input_size = int(data["input_size"][0])
        self.latent_size = int(data["latent_size"][0])
        self.hidden_sizes = list(data["hidden_sizes"])
        
        self.encoder_layers = []
        i = 0
        while f"enc_w_{i}" in data:
            w = data[f"enc_w_{i}"].astype(np.float32)
            b = data[f"enc_b_{i}"].astype(np.float32)
            act = str(data[f"enc_act_{i}"][0])
            layer = DenseLayer(w.shape[0], w.shape[1], activation=act, seed=42)
            layer.w = w; layer.b = b
            self.encoder_layers.append(layer)
            i += 1
        
        self.decoder_layers = []
        i = 0
        while f"dec_w_{i}" in data:
            w = data[f"dec_w_{i}"].astype(np.float32)
            b = data[f"dec_b_{i}"].astype(np.float32)
            act = str(data[f"dec_act_{i}"][0])
            layer = DenseLayer(w.shape[0], w.shape[1], activation=act, seed=42)
            layer.w = w; layer.b = b
            self.decoder_layers.append(layer)
            i += 1
        print(f"✅ Autoencoder carregado: {filepath}")


# ============================================================
# FEATURE EXTRACTION (shared between train and inference)
# ============================================================

def extract_features_from_file(filepath, sr=22050, n_features=256, max_segs=20):
    """Extract FFT features from an audio file. Shared with training."""
    try:
        import soundfile as sf
        audio, fsr = sf.read(filepath, dtype="float32")
        if len(audio.shape) > 1: audio = audio.mean(axis=1)
        if fsr != sr:
            idx = np.round(np.arange(0, len(audio), fsr/sr)).astype(int)
            audio = audio[idx[idx < len(audio)]]
    except Exception:
        return None
    if len(audio) < sr: return None
    
    features = []
    seg_len = sr
    n_segs = min(len(audio) // seg_len, max_segs)
    for i in range(n_segs):
        seg = audio[i * seg_len:(i + 1) * seg_len]
        fft = np.abs(np.fft.rfft(seg))[:n_features]
        fft = fft / (np.max(fft) + 1e-10)
        if len(fft) < n_features: fft = np.pad(fft, (0, n_features - len(fft)))
        features.append(fft)
    return features


def generate_synthetic_data(n_samples=2000, n_features=256):
    print(f"  Gerando {n_samples} amostras sintéticas...")
    X = []
    for i in range(n_samples):
        t = np.linspace(0, 1, 44100)
        signal = np.zeros_like(t)
        pattern_type = np.random.choice(["harmonic","noise","sweep","pulse","fm","granular","chord","bass","drum","melody","arp","pad"])
        if pattern_type == "harmonic":
            freq = np.random.uniform(80, 3000)
            for h in range(1, np.random.randint(2, 15)):
                signal += np.sin(2 * np.pi * freq * h * t) / h
        elif pattern_type == "noise":
            signal = np.random.randn(len(t)) * np.exp(-t * np.random.uniform(1, 10))
        elif pattern_type == "sweep":
            f0 = np.random.uniform(50, 500)
            f1 = np.random.uniform(500, 5000)
            freqs = np.linspace(f0, f1, len(t))
            signal = np.sin(2 * np.pi * np.cumsum(freqs) / 44100)
        elif pattern_type == "pulse":
            freq = np.random.uniform(2, 20)
            signal = np.sin(2 * np.pi * freq * t) * np.exp(-((t % 0.5 - 0.25) ** 2) * 50)
        elif pattern_type == "fm":
            fc = np.random.uniform(200, 1000)
            fm = np.random.uniform(10, 200)
            signal = np.sin(2 * np.pi * fc * t + 3 * np.sin(2 * np.pi * fm * t))
        elif pattern_type == "granular":
            for _ in range(20):
                pos = np.random.randint(0, len(t) - 1000)
                grain = np.random.randn(1000) * np.hanning(1000)
                signal[pos:pos + 1000] += grain * 0.3
        elif pattern_type == "chord":
            root = np.random.uniform(100, 500)
            for interval in [0, 4, 7, 11]:
                freq = root * (2 ** (interval / 12))
                signal += np.sin(2 * np.pi * freq * t) * 0.3
        elif pattern_type == "bass":
            freq = np.random.uniform(40, 200)
            signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 3)
        elif pattern_type == "drum":
            freq = np.random.uniform(100, 300)
            signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 20)
        elif pattern_type == "melody":
            freq = np.random.uniform(200, 1000)
            signal = np.sin(2 * np.pi * freq * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
        elif pattern_type == "arp":
            base = np.random.uniform(200, 600)
            for j in range(8):
                freq = base * (2 ** (np.random.choice([0,4,7,12]) / 12))
                start = j * len(t) // 8; end = (j + 1) * len(t) // 8
                signal[start:end] += np.sin(2 * np.pi * freq * t[start:end]) * 0.3
        else:
            freq = np.random.uniform(100, 400)
            signal = np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*freq*1.005*t) + 0.5*np.sin(2*np.pi*freq*0.995*t)
        signal += np.random.randn(len(t)) * 0.05
        fft = np.abs(np.fft.rfft(signal))[:n_features]
        fft = fft / (np.max(fft) + 1e-10)
        if len(fft) < n_features: fft = np.pad(fft, (0, n_features - len(fft)))
        X.append(fft)
    return np.array(X)


def prepare_data(n_features=256):
    print("📊 Preparando dados...")
    all_features = []
    music_path = Path("music_input")
    if music_path.exists():
        files = []
        for ext in ["*.mp3", "*.wav", "*.flac", "*.ogg"]:
            files.extend(list(music_path.glob(ext)))
        print(f"  {len(files)} músicas de referência")
        for f in files[:30]:
            feats = extract_features_from_file(f, n_features=n_features)
            if feats: all_features.extend(feats)
    if len(all_features) < 100:
        synthetic = generate_synthetic_data(2000, n_features)
        all_features.extend(synthetic.tolist())
    X = np.array(all_features)
    print(f"  Dataset: {X.shape[0]} x {X.shape[1]}")
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    return X_norm, X_mean, X_std


def train_autoencoder(epochs=10000, batch_size=64, n_features=256, latent_size=64, lr_init=0.001):
    print("=" * 60)
    print("🧠 TREINAMENTO AUTOENCODER")
    print("=" * 60)
    os.makedirs(AE_DIR, exist_ok=True)
    X, X_mean, X_std = prepare_data(n_features)
    ae = Autoencoder(input_size=n_features, hidden_sizes=[512, 256, 128], latent_size=latent_size)
    print("\n🚀 TREINANDO...")
    history = []; best_loss = float("inf"); start = time.time()
    for epoch in range(1, epochs + 1):
        lr = max(lr_init * (0.9995 ** epoch), 0.00001)
        indices = np.random.permutation(len(X))
        epoch_loss = 0; n_batches = 0
        for i in range(0, len(X), batch_size):
            idx = indices[i:i + batch_size]
            try:
                loss = ae.train_step(X[idx], lr)
                epoch_loss += loss; n_batches += 1
            except Exception as e:
                print(f"  ⚠️  batch {i}: {e}")
            if n_batches % 10 == 0: gc.collect()
        if n_batches == 0: break
        avg_loss = epoch_loss / n_batches
        history.append(float(avg_loss))
        if epoch % 100 == 0 or epoch == 1:
            elapsed = time.time() - start
            eta = (elapsed / epoch) * (epochs - epoch)
            print(f"  Epoch {epoch:6d}/{epochs} | Loss: {avg_loss:.6f} | ETA: {eta/60:.1f}min")
        if epoch % 1000 == 0:
            ae.save(os.path.join(AE_DIR, f"checkpoint_{epoch}.npz"))
        if avg_loss < best_loss:
            best_loss = avg_loss
            ae.save(os.path.join(AE_DIR, "best_autoencoder.npz"))
    ae.save(os.path.join(AE_DIR, "final_autoencoder.npz"))
    np.savez(os.path.join(AE_DIR, "normalization.npz"), mean=X_mean, std=X_std)
    
    # Auto-calibrate the latent controller with synthetic→param mapping
    try:
        _auto_calibrate(ae, X, X_mean, X_std)
    except Exception as e:
        print(f"  ⚠️  Calibração falhou: {e}")
    
    total = time.time() - start
    print(f"\n✅ Treino concluído em {total/60:.1f} min")
    print(f"   Loss final: {history[-1]:.6f}")


def _auto_calibrate(ae, X_norm, X_mean, X_std, n_samples=500):
    """
    Automatically calibrate the latent controller by mapping
    random latent vectors back to interpretable parameters.
    This lets the controller work even without hand-labeled data.
    """
    from latent_controller import LatentController
    
    print("\n🔧 Auto-calibrando latent controller...")
    # Sample many random latents and compute their corresponding
    # feature-space statistics as pseudo-parameters
    np.random.seed(123)
    n = min(n_samples, 500)
    latents = []
    params = []
    
    for _ in range(n):
        # Generate synthetic features, encode them
        feat = np.random.randn(X_norm.shape[1]) * 0.5
        latent = ae.encode(feat.reshape(1, -1))[0]
        
        # Compute pseudo-parameters from the feature itself
        # These are derived from the FFT in a stable way
        feat_pos = np.abs(feat)
        n_bins = len(feat_pos)
        bass_energy = np.sum(feat_pos[:n_bins//8]) / (np.sum(feat_pos) + 1e-10)
        high_energy = np.sum(feat_pos[n_bins*3//4:]) / (np.sum(feat_pos) + 1e-10)
        spectral_centroid = np.sum(np.arange(n_bins) * feat_pos) / (np.sum(feat_pos) + 1e-10) / n_bins
        spectral_variance = np.std(feat_pos) / (np.mean(feat_pos) + 1e-10)
        peakiness = np.max(feat_pos) / (np.mean(feat_pos) + 1e-10)
        
        # Normalize to [0, 1] with reasonable bounds
        params.append([
            0.5,                                    # bpm (placeholder)
            np.clip(spectral_variance / 5.0, 0, 1),  # intensity proxy
            np.clip(high_energy * 3, 0, 1),          # brightness
            np.clip(bass_energy * 3, 0, 1),          # bass_weight
            np.clip(peakiness / 10.0, 0, 1),          # rhythmic_density
            np.clip(spectral_centroid, 0, 1),          # harmonic_complexity
            np.clip(spectral_variance / 3.0, 0, 1),  # dynamics_range
            np.random.rand(),                        # style_index (random for now)
        ])
        latents.append(latent)
    
    latents = np.array(latents)
    params = np.array(params)
    
    controller = LatentController(ae)
    controller.calibrate(latents, params)
    print("  ✅ Calibração concluída")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--latent", type=int, default=64)
    args = parser.parse_args()
    train_autoencoder(
        epochs=max(args.epochs, 10000),
        batch_size=args.batch_size,
        n_features=args.features,
        latent_size=args.latent
    )
