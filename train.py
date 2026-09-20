#!/usr/bin/env python3
"""
🧠 TREINAMENTO COM EARLY STOPPING + VALIDATION SPLIT
CORREÇÕES:
- Early stopping (para quando validation loss estagnar)
- Validation split (80/20)
- Dropout (0.2) para prevenir overfitting
- Batch size reduzido (32)
- Data augmentation básica
"""
import os,sys,json,time,gc
import numpy as np
from pathlib import Path

MODEL_DIR="models"
AE_DIR=os.path.join(MODEL_DIR,"autoencoder")

class DenseLayer:
    def __init__(self, in_size, out_size, activation="relu", dropout=0.0, seed=42):
        np.random.seed(seed)
        scale = np.sqrt(2.0 / (in_size + out_size))
        self.w = np.random.randn(in_size, out_size) * scale
        self.b = np.zeros(out_size)
        self.activation = activation
        self.dropout = dropout
        self.m_w = np.zeros_like(self.w)
        self.v_w = np.zeros_like(self.w)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.input = None
        self.z = None
        self.output = None
        self.dropout_mask = None
    
    def forward(self, x, training=True):
        self.input = x
        self.z = x @ self.w + self.b
        if self.activation == "relu": 
            self.output = np.maximum(0, self.z)
        elif self.activation == "tanh": 
            self.output = np.tanh(self.z)
        else: 
            self.output = self.z
        
        # Dropout (apenas em training)
        if training and self.dropout > 0:
            self.dropout_mask = (np.random.rand(*self.output.shape) > self.dropout) / (1 - self.dropout)
            self.output = self.output * self.dropout_mask
        else:
            self.dropout_mask = None
        
        return self.output
    
    def backward(self, grad_output, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        batch_size = grad_output.shape[0]
        
        # Aplicar máscara de dropout ao gradiente
        if self.dropout_mask is not None:
            grad_output = grad_output * self.dropout_mask
        
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
    def __init__(self, input_size=256, hidden_sizes=None, latent_size=64, dropout=0.2, seed=42):
        if hidden_sizes is None: hidden_sizes = [512, 256, 128]
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.dropout = dropout
        self.t = 0
        
        self.encoder_layers = []
        layer_sizes = [input_size] + hidden_sizes + [latent_size]
        for i in range(len(layer_sizes) - 1):
            is_last = (i == len(layer_sizes) - 2)
            activation = "linear" if is_last else "relu"
            drop = 0.0 if is_last else dropout
            self.encoder_layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1], activation, drop, seed=seed+i))
        
        self.decoder_layers = []
        decoder_sizes = [latent_size] + list(reversed(hidden_sizes)) + [input_size]
        for i in range(len(decoder_sizes) - 1):
            is_last = (i == len(decoder_sizes) - 2)
            activation = "linear" if is_last else "relu"
            drop = 0.0 if is_last else dropout
            self.decoder_layers.append(DenseLayer(decoder_sizes[i], decoder_sizes[i+1], activation, drop, seed=seed+100+i))
    
    def count_params(self):
        c = sum(l.w.size + l.b.size for l in self.encoder_layers)
        c += sum(l.w.size + l.b.size for l in self.decoder_layers)
        return c
    
    def encode(self, x, training=False):
        h = x
        for layer in self.encoder_layers: h = layer.forward(h, training)
        return h
    
    def decode(self, latent, training=False):
        h = latent
        for layer in self.decoder_layers: h = layer.forward(h, training)
        return h
    
    def forward(self, x, training=True):
        latent = self.encode(x, training)
        return self.decode(latent, training), latent
    
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
        output, latent = self.forward(x, training=True)
        loss = self.compute_loss(x, output)
        self.backward(x, output, lr)
        return loss
    
    def eval_step(self, x):
        """Forward sem dropout para validação"""
        output, _ = self.forward(x, training=False)
        return self.compute_loss(x, output)
    
    def save(self, filepath):
        save_dict = {
            "input_size": np.array([self.input_size]),
            "latent_size": np.array([self.latent_size]),
            "hidden_sizes": np.array(self.hidden_sizes),
            "dropout": np.array([self.dropout]),
        }
        for i, layer in enumerate(self.encoder_layers):
            save_dict[f"enc_w_{i}"] = layer.w.astype(np.float32)
            save_dict[f"enc_b_{i}"] = layer.b.astype(np.float32)
            save_dict[f"enc_act_{i}"] = np.array([layer.activation])
            save_dict[f"enc_drop_{i}"] = np.array([layer.dropout])
        for i, layer in enumerate(self.decoder_layers):
            save_dict[f"dec_w_{i}"] = layer.w.astype(np.float32)
            save_dict[f"dec_b_{i}"] = layer.b.astype(np.float32)
            save_dict[f"dec_act_{i}"] = np.array([layer.activation])
            save_dict[f"dec_drop_{i}"] = np.array([layer.dropout])
        np.savez_compressed(filepath, **save_dict)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"💾 Autoencoder salvo: {filepath} ({size_mb:.2f} MB)")
    
    def load(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        self.input_size = int(data["input_size"][0])
        self.latent_size = int(data["latent_size"][0])
        self.hidden_sizes = list(data["hidden_sizes"])
        self.dropout = float(data.get("dropout", [0.2])[0])
        
        self.encoder_layers = []
        i = 0
        while f"enc_w_{i}" in data:
            w = data[f"enc_w_{i}"].astype(np.float32)
            b = data[f"enc_b_{i}"].astype(np.float32)
            act = str(data[f"enc_act_{i}"][0])
            drop = float(data.get(f"enc_drop_{i}", [0.0])[0])
            layer = DenseLayer(w.shape[0], w.shape[1], activation=act, dropout=drop, seed=42)
            layer.w = w; layer.b = b
            self.encoder_layers.append(layer)
            i += 1
        
        self.decoder_layers = []
        i = 0
        while f"dec_w_{i}" in data:
            w = data[f"dec_w_{i}"].astype(np.float32)
            b = data[f"dec_b_{i}"].astype(np.float32)
            act = str(data[f"dec_act_{i}"][0])
            drop = float(data.get(f"dec_drop_{i}", [0.0])[0])
            layer = DenseLayer(w.shape[0], w.shape[1], activation=act, dropout=drop, seed=42)
            layer.w = w; layer.b = b
            self.decoder_layers.append(layer)
            i += 1
        print(f"✅ Autoencoder carregado: {filepath}")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features_from_file(filepath, sr=22050, n_features=256, max_segs=20):
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
    patterns = ["harmonic","noise","sweep","pulse","fm","granular","chord","bass","drum","melody","arp","pad","fm_sweep","chord_progression"]
    for i in range(n_samples):
        t = np.linspace(0, 1, 44100)
        signal = np.zeros_like(t)
        pattern_type = np.random.choice(patterns)
        
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
                start = j * len(t) // 8
                end = (j + 1) * len(t) // 8
                signal[start:end] += np.sin(2 * np.pi * freq * t[start:end]) * 0.3
        elif pattern_type == "fm_sweep":
            fc = np.linspace(200, 800, len(t))
            fm = np.linspace(10, 100, len(t))
            signal = np.sin(2 * np.pi * np.cumsum(fc)/44100 + 3 * np.sin(2 * np.pi * np.cumsum(fm)/44100))
        elif pattern_type == "chord_progression":
            roots = [261.63, 220.0, 196.0, 246.94]
            chord_dur = len(t) // 4
            for c, root in enumerate(roots):
                start = c * chord_dur
                end = (c+1) * chord_dur
                for interval in [0, 4, 7]:
                    freq = root * (2 ** (interval/12))
                    signal[start:end] += np.sin(2 * np.pi * freq * t[start:end]) * 0.25
        else:
            freq = np.random.uniform(100, 400)
            signal = np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*freq*1.005*t) + 0.5*np.sin(2*np.pi*freq*0.995*t)
        
        signal += np.random.randn(len(t)) * 0.05
        fft = np.abs(np.fft.rfft(signal))[:n_features]
        fft = fft / (np.max(fft) + 1e-10)
        if len(fft) < n_features: fft = np.pad(fft, (0, n_features - len(fft)))
        X.append(fft)
    return np.array(X)


def prepare_data_with_validation(n_features=256, train_ratio=0.8):
    """Prepara dados com split treino/validação - CORREÇÃO CRÍTICA"""
    print("📊 Preparando dados COM VALIDAÇÃO...")
    all_features = []
    
    music_path = Path("music_input")
    if music_path.exists():
        files = []
        for ext in ["*.mp3", "*.wav", "*.flac", "*.ogg"]:
            files.extend(list(music_path.glob(ext)))
        print(f"  {len(files)} músicas de referência")
        for f in files:
            feats = extract_features_from_file(f, n_features=n_features)
            if feats:
                all_features.extend(feats)
    
    # Gerar dados sintéticos se necessário (expandir para 2000+)
    target_size = 2000
    if len(all_features) < target_size:
        needed = target_size - len(all_features)
        print(f"  Gerando {needed} amostras sintéticas para completar...")
        synthetic = generate_synthetic_data(needed, n_features)
        all_features.extend(synthetic.tolist())
    
    X = np.array(all_features)
    print(f"  Dataset total: {X.shape[0]} x {X.shape[1]}")
    
    # Normalizar
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    
    # Data augmentation: adicionar pequeno ruído
    noise = np.random.randn(*X_norm.shape) * 0.05
    X_norm = X_norm + noise
    
    # Split treino/validação
    np.random.shuffle(X_norm)
    split_idx = int(len(X_norm) * train_ratio)
    X_train = X_norm[:split_idx]
    X_val = X_norm[split_idx:]
    
    print(f"  ✅ Treino: {X_train.shape[0]} | Validação: {X_val.shape[0]}")
    
    return X_train, X_val, X_mean, X_std


def train_epoch(ae, X, batch_size, lr):
    """Treina uma época completa"""
    indices = np.random.permutation(len(X))
    epoch_loss = 0
    n_batches = 0
    
    for i in range(0, len(X), batch_size):
        idx = indices[i:i + batch_size]
        x_batch = X[idx]
        loss = ae.train_step(x_batch, lr)
        epoch_loss += loss
        n_batches += 1
    
    return epoch_loss / max(n_batches, 1)


def validate_epoch(ae, X, batch_size):
    """Valida o modelo (sem dropout)"""
    epoch_loss = 0
    n_batches = 0
    
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i + batch_size]
        loss = ae.eval_step(x_batch)
        epoch_loss += loss
        n_batches += 1
    
    return epoch_loss / max(n_batches, 1)


def train_autoencoder(epochs=800, batch_size=32, n_features=256, latent_size=64, 
                     lr_init=0.001, early_stopping_patience=50, dropout=0.2):
    """
    TREINAMENTO COM EARLY STOPPING - CORREÇÃO DO OVERFITTING
    - Validação separada
    - Para quando validation loss não melhora por patience épocas
    - Dropout para regularização
    - Batch size reduzido
    """
    print("=" * 60)
    print("🧠 TREINAMENTO AUTOENCODER (CORRIGIDO - EARLY STOPPING)")
    print(f"   Épocas máximas: {epochs}")
    print(f"   Early stopping patience: {early_stopping_patience}")
    print(f"   Batch size: {batch_size}")
    print(f"   Dropout: {dropout}")
    print("=" * 60)
    
    os.makedirs(AE_DIR, exist_ok=True)
    
    X_train, X_val, X_mean, X_std = prepare_data_with_validation(n_features, train_ratio=0.8)
    
    ae = Autoencoder(
        input_size=n_features,
        hidden_sizes=[512, 256, 128],
        latent_size=latent_size,
        dropout=dropout
    )
    
    print(f"\n📊 Parâmetros: {ae.count_params():,}")
    print(f"📊 Amostra/Parâmetro ratio: {len(X_train)/ae.count_params()*1000:.2f} por mil")
    
    print("\n🚀 TREINANDO COM EARLY STOPPING...")
    history = {"train": [], "val": []}
    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    start = time.time()
    
    for epoch in range(1, epochs + 1):
        lr = lr_init * (0.95 ** (epoch // 100))
        
        train_loss = train_epoch(ae, X_train, batch_size, lr)
        history["train"].append(float(train_loss))
        
        val_loss = validate_epoch(ae, X_val, batch_size)
        history["val"].append(float(val_loss))
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            ae.save(os.path.join(AE_DIR, "best_autoencoder.npz"))
            marker = "⭐"
        else:
            patience_counter += 1
            marker = ""
        
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start
            eta = (elapsed / epoch) * (epochs - epoch) if epoch < epochs else 0
            print(f"  Epoch {epoch:4d}/{epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Best: {best_loss:.6f} (ep {best_epoch}) | "
                  f"Patience: {patience_counter}/{early_stopping_patience} {marker}")
        
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹️  EARLY STOPPING em epoch {epoch}")
            print(f"   Melhor validação: {best_loss:.6f} em epoch {best_epoch}")
            break
        
        if epoch % 50 == 0:
            gc.collect()
    
    ae.save(os.path.join(AE_DIR, "final_autoencoder.npz"))
    np.savez(os.path.join(AE_DIR, "normalization.npz"), mean=X_mean, std=X_std)
    
    total = time.time() - start
    metadata = {
        "type": "Autoencoder_Corrected",
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "input_size": n_features,
        "latent_size": latent_size,
        "hidden_sizes": [512, 256, 128],
        "dropout": dropout,
        "early_stopping": True,
        "patience": early_stopping_patience,
        "final_train_loss": float(history["train"][-1]),
        "final_val_loss": float(history["val"][-1]),
        "best_val_loss": float(best_loss),
        "training_time_min": float(total / 60),
        "dataset_size": int(len(X_train) + len(X_val)),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "total_params": ae.count_params(),
        "history_sample_train": history["train"][::max(1,len(history["train"])//50)],
        "history_sample_val": history["val"][::max(1,len(history["val"])//50)],
    }
    with open(os.path.join(AE_DIR, "training_log.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print(f"   Épocas executadas: {epoch}/{epochs}")
    print(f"   Melhor validação: {best_loss:.6f} (epoch {best_epoch})")
    print(f"   Tempo total: {total/60:.1f} min")
    print(f"   Overfitting check: train={history['train'][-1]:.6f} vs val={history['val'][-1]:.6f}")
    if history["val"][-1] > history["train"][-1] * 2:
        print("   ⚠️  Possível overfitting ainda presente!")
    else:
        print("   ✅ Modelo generaliza bem (train ≈ val)")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()
    train_autoencoder(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_features=args.features,
        latent_size=args.latent,
        early_stopping_patience=args.patience,
        dropout=args.dropout
    )
