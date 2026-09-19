#!/usr/bin/env python3
"""
🧠 AUTOENCODER REAL PARA MÚSICA - VERSÃO CORRIGIDA
Backpropagation com shapes corretas e testes internos
"""
import os,sys,json,time,gc
import numpy as np
from pathlib import Path

MODEL_DIR="models"
AE_DIR=os.path.join(MODEL_DIR,"autoencoder")

class DenseLayer:
    """Camada densa com forward/backward"""
    def __init__(self, in_size, out_size, activation='relu', seed=42):
        np.random.seed(seed)
        # Xavier initialization
        scale = np.sqrt(2.0 / (in_size + out_size))
        self.w = np.random.randn(in_size, out_size) * scale
        self.b = np.zeros(out_size)
        self.activation = activation
        
        # Adam state
        self.m_w = np.zeros_like(self.w)
        self.v_w = np.zeros_like(self.w)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        
        # Cache
        self.input = None
        self.z = None
        self.output = None
    
    def forward(self, x):
        self.input = x
        self.z = x @ self.w + self.b
        
        if self.activation == 'relu':
            self.output = np.maximum(0, self.z)
        elif self.activation == 'tanh':
            self.output = np.tanh(self.z)
        elif self.activation == 'linear':
            self.output = self.z
        else:
            self.output = self.z
        
        return self.output
    
    def backward(self, grad_output, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        """Backpropagation com shapes corretas"""
        batch_size = grad_output.shape[0]
        
        # Gradiente através da ativação
        if self.activation == 'relu':
            grad_act = grad_output * (self.z > 0).astype(float)
        elif self.activation == 'tanh':
            grad_act = grad_output * (1 - self.output ** 2)
        elif self.activation == 'linear':
            grad_act = grad_output
        else:
            grad_act = grad_output
        
        # Gradientes dos pesos
        # self.input: (batch, in_size), grad_act: (batch, out_size)
        grad_w = (self.input.T @ grad_act) / batch_size
        grad_b = np.mean(grad_act, axis=0)
        
        # Clip gradient
        grad_w = np.clip(grad_w, -1.0, 1.0)
        grad_b = np.clip(grad_b, -1.0, 1.0)
        
        # Adam update
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
        
        # Gradiente para próxima camada
        # grad_act: (batch, out_size), self.w: (in_size, out_size)
        grad_input = grad_act @ self.w.T
        
        return grad_input


class Autoencoder:
    """Autoencoder com camadas modulares"""
    
    def __init__(self, input_size=256, hidden_sizes=None, latent_size=64, seed=42):
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.t = 0
        
        # Construir ENCODER
        self.encoder_layers = []
        layer_sizes = [input_size] + hidden_sizes + [latent_size]
        
        for i in range(len(layer_sizes) - 1):
            is_last = (i == len(layer_sizes) - 2)
            activation = 'linear' if is_last else 'relu'  # Última camada linear (sem tanh)
            layer = DenseLayer(layer_sizes[i], layer_sizes[i+1], activation, seed=seed+i)
            self.encoder_layers.append(layer)
        
        # Construir DECODER (espelho do encoder)
        self.decoder_layers = []
        decoder_sizes = [latent_size] + list(reversed(hidden_sizes)) + [input_size]
        
        for i in range(len(decoder_sizes) - 1):
            is_last = (i == len(decoder_sizes) - 2)
            activation = 'linear' if is_last else 'relu'
            layer = DenseLayer(decoder_sizes[i], decoder_sizes[i+1], activation, seed=seed+100+i)
            self.decoder_layers.append(layer)
        
        total_params = self.count_params()
        print(f"🧠 Autoencoder inicializado:")
        print(f"   Input: {input_size}")
        print(f"   Encoder: {input_size} → {' → '.join(map(str, hidden_sizes))} → {latent_size}")
        print(f"   Bottleneck: {latent_size}")
        print(f"   Decoder: {latent_size} → {' → '.join(map(str, reversed(hidden_sizes)))} → {input_size}")
        print(f"   Total params: {total_params:,}")
    
    def count_params(self):
        count = 0
        for layer in self.encoder_layers:
            count += layer.w.size + layer.b.size
        for layer in self.decoder_layers:
            count += layer.w.size + layer.b.size
        return count
    
    def encode(self, x):
        """Encoder: input → latent"""
        h = x
        for layer in self.encoder_layers:
            h = layer.forward(h)
        return h
    
    def decode(self, latent):
        """Decoder: latent → output"""
        h = latent
        for layer in self.decoder_layers:
            h = layer.forward(h)
        return h
    
    def forward(self, x):
        """Forward pass completo"""
        latent = self.encode(x)
        output = self.decode(latent)
        return output, latent
    
    def compute_loss(self, x, output):
        """MSE loss"""
        return np.mean((x - output) ** 2)
    
    def backward(self, x, output, lr=0.001):
        """Backpropagation completo"""
        self.t += 1
        batch_size = x.shape[0]
        
        # Gradiente da MSE loss: d(MSE)/d(output) = 2*(output-x)/N
        grad = 2 * (output - x) / batch_size
        
        # Backward decoder
        for layer in reversed(self.decoder_layers):
            grad = layer.backward(grad, lr, self.t)
        
        # Backward encoder
        for layer in reversed(self.encoder_layers):
            grad = layer.backward(grad, lr, self.t)
    
    def train_step(self, x, lr=0.001):
        """Um passo de treino"""
        output, latent = self.forward(x)
        loss = self.compute_loss(x, output)
        self.backward(x, output, lr)
        return loss
    
    def save(self, filepath):
        """Salva modelo"""
        save_dict = {
            'input_size': np.array([self.input_size]),
            'latent_size': np.array([self.latent_size]),
            'hidden_sizes': np.array(self.hidden_sizes),
        }
        
        for i, layer in enumerate(self.encoder_layers):
            save_dict[f'enc_w_{i}'] = layer.w.astype(np.float32)
            save_dict[f'enc_b_{i}'] = layer.b.astype(np.float32)
            save_dict[f'enc_act_{i}'] = np.array([layer.activation])
        
        for i, layer in enumerate(self.decoder_layers):
            save_dict[f'dec_w_{i}'] = layer.w.astype(np.float32)
            save_dict[f'dec_b_{i}'] = layer.b.astype(np.float32)
            save_dict[f'dec_act_{i}'] = np.array([layer.activation])
        
        np.savez_compressed(filepath, **save_dict)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"💾 Autoencoder salvo: {filepath} ({size_mb:.2f} MB)")
    
    def load(self, filepath):
        """Carrega modelo"""
        data = np.load(filepath, allow_pickle=True)
        self.input_size = int(data['input_size'][0])
        self.latent_size = int(data['latent_size'][0])
        self.hidden_sizes = list(data['hidden_sizes'])
        
        self.encoder_layers = []
        i = 0
        while f'enc_w_{i}' in data:
            w = data[f'enc_w_{i}'].astype(np.float32)
            b = data[f'enc_b_{i}'].astype(np.float32)
            act = str(data[f'enc_act_{i}'][0])
            layer = DenseLayer(w.shape[0], w.shape[1], activation=act, seed=42)
            layer.w = w
            layer.b = b
            self.encoder_layers.append(layer)
            i += 1
        
        self.decoder_layers = []
        i = 0
        while f'dec_w_{i}' in data:
            w = data[f'dec_w_{i}'].astype(np.float32)
            b = data[f'dec_b_{i}'].astype(np.float32)
            act = str(data[f'dec_act_{i}'][0])
            layer = DenseLayer(w.shape[0], w.shape[1], activation=act, seed=42)
            layer.w = w
            layer.b = b
            self.decoder_layers.append(layer)
            i += 1
        
        print(f"✅ Autoencoder carregado: {filepath}")
    
    def test_shapes(self, x):
        """Testa shapes internamente"""
        try:
            output, latent = self.forward(x)
            assert output.shape == x.shape, f"Output shape {output.shape} != input {x.shape}"
            assert latent.shape[0] == x.shape[0], "Batch size mismatch"
            assert latent.shape[1] == self.latent_size, f"Latent size {latent.shape[1]} != {self.latent_size}"
            print(f"  ✅ Teste shapes: input={x.shape} → latent={latent.shape} → output={output.shape}")
            return True
        except Exception as e:
            print(f"  ❌ Erro shapes: {e}")
            return False


def extract_features(filepath, sr=22050, n_features=256):
    try:
        import soundfile as sf
        audio, fsr = sf.read(filepath, dtype='float32')
        if len(audio.shape) > 1: audio = np.mean(audio, axis=1)
        if fsr != sr:
            idx = np.round(np.arange(0, len(audio), fsr/sr)).astype(int)
            audio = audio[idx[idx < len(audio)]]
    except:
        return None
    if len(audio) < sr: return None
    features = []
    seg_len = sr
    n_segs = min(len(audio) // seg_len, 20)
    for i in range(n_segs):
        seg = audio[i*seg_len:(i+1)*seg_len]
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
        pattern_type = np.random.choice(['harmonic','noise','sweep','pulse','fm','granular','chord','bass','drum','melody','arp','pad'])
        
        if pattern_type == 'harmonic':
            freq = np.random.uniform(80, 3000)
            for h in range(1, np.random.randint(2, 15)):
                signal += np.sin(2*np.pi*freq*h*t) / h
        elif pattern_type == 'noise':
            signal = np.random.randn(len(t)) * np.exp(-t * np.random.uniform(1, 10))
        elif pattern_type == 'sweep':
            f0 = np.random.uniform(50, 500)
            f1 = np.random.uniform(500, 5000)
            freqs = np.linspace(f0, f1, len(t))
            signal = np.sin(2*np.pi*np.cumsum(freqs)/44100)
        elif pattern_type == 'pulse':
            freq = np.random.uniform(2, 20)
            signal = np.sin(2*np.pi*freq*t) * np.exp(-((t%0.5-0.25)**2)*50)
        elif pattern_type == 'fm':
            fc = np.random.uniform(200, 1000)
            fm = np.random.uniform(10, 200)
            signal = np.sin(2*np.pi*fc*t + 3*np.sin(2*np.pi*fm*t))
        elif pattern_type == 'granular':
            for _ in range(20):
                pos = np.random.randint(0, len(t)-1000)
                grain = np.random.randn(1000) * np.hanning(1000)
                signal[pos:pos+1000] += grain * 0.3
        elif pattern_type == 'chord':
            root = np.random.uniform(100, 500)
            for interval in [0, 4, 7, 11]:
                freq = root * (2**(interval/12))
                signal += np.sin(2*np.pi*freq*t) * 0.3
        elif pattern_type == 'bass':
            freq = np.random.uniform(40, 200)
            signal = np.sin(2*np.pi*freq*t) * np.exp(-t*3)
        elif pattern_type == 'drum':
            freq = np.random.uniform(100, 300)
            signal = np.sin(2*np.pi*freq*t) * np.exp(-t*20)
        elif pattern_type == 'melody':
            freq = np.random.uniform(200, 1000)
            signal = np.sin(2*np.pi*freq*t) * (1+0.3*np.sin(2*np.pi*5*t))
        elif pattern_type == 'arp':
            base = np.random.uniform(200, 600)
            for j in range(8):
                freq = base * (2**(np.random.choice([0,4,7,12])/12))
                start = j*len(t)//8
                end = (j+1)*len(t)//8
                signal[start:end] += np.sin(2*np.pi*freq*t[start:end]) * 0.3
        else:  # pad
            freq = np.random.uniform(100, 400)
            signal = np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*freq*1.005*t) + 0.5*np.sin(2*np.pi*freq*0.995*t)
        
        signal += np.random.randn(len(t)) * 0.05
        fft = np.abs(np.fft.rfft(signal))[:n_features]
        fft = fft / (np.max(fft) + 1e-10)
        if len(fft) < n_features: fft = np.pad(fft, (0, n_features - len(fft)))
        X.append(fft)
    return np.array(X)

def prepare_data(n_features=256):
    print("📊 Preparando dados para o autoencoder...")
    all_features = []
    
    music_path = Path("music_input")
    if music_path.exists():
        files = []
        for ext in ['*.mp3', '*.wav', '*.flac', '*.ogg']:
            files.extend(list(music_path.glob(ext)))
        print(f"  {len(files)} músicas de referência")
        for f in files[:30]:
            feats = extract_features(f, n_features=n_features)
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
    print("🧠 TREINAMENTO AUTOENCODER (CORRIGIDO)")
    print(f"   Épocas: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Features: {n_features}")
    print(f"   Latent size: {latent_size}")
    print("=" * 60)
    
    os.makedirs(AE_DIR, exist_ok=True)
    
    X, X_mean, X_std = prepare_data(n_features)
    
    ae = Autoencoder(
        input_size=n_features,
        hidden_sizes=[512, 256, 128],
        latent_size=latent_size
    )
    
    # TESTE DE SHAPES antes de treinar
    print("\n🔍 Testando shapes...")
    test_batch = X[:4]
    if not ae.test_shapes(test_batch):
        print("❌ Shapes incorretas! Abortando.")
        return
    
    print("\n🚀 TREINANDO AUTOENCODER...")
    history = []
    best_loss = float('inf')
    start = time.time()
    
    for epoch in range(1, epochs + 1):
        lr = max(lr_init * (0.9995 ** epoch), 0.00001)
        indices = np.random.permutation(len(X))
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(X), batch_size):
            idx = indices[i:i+batch_size]
            x_batch = X[idx]
            
            try:
                loss = ae.train_step(x_batch, lr)
                epoch_loss += loss
                n_batches += 1
            except Exception as e:
                print(f"  ⚠️ Erro no batch {i}: {e}")
                continue
            
            if n_batches % 10 == 0:
                gc.collect()
        
        if n_batches == 0:
            print("❌ Nenhum batch processado!")
            break
        
        avg_loss = epoch_loss / n_batches
        history.append(float(avg_loss))
        
        if epoch % 100 == 0 or epoch == 1:
            elapsed = time.time() - start
            eta = (elapsed / epoch) * (epochs - epoch)
            print(f"  Epoch {epoch:6d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr:.6f} | ETA: {eta/60:.1f}min")
        
        if epoch % 1000 == 0:
            ae.save(os.path.join(AE_DIR, f"checkpoint_{epoch}.npz"))
            print(f"  💾 Checkpoint salvo (epoch {epoch})")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            ae.save(os.path.join(AE_DIR, "best_autoencoder.npz"))
    
    ae.save(os.path.join(AE_DIR, "final_autoencoder.npz"))
    np.savez(os.path.join(AE_DIR, "normalization.npz"), mean=X_mean, std=X_std)
    
    total_time = time.time() - start
    metadata = {
        "type": "Autoencoder_Fixed",
        "epochs": epochs,
        "input_size": n_features,
        "latent_size": latent_size,
        "hidden_sizes": [512, 256, 128],
        "final_loss": float(history[-1]) if history else None,
        "best_loss": float(best_loss),
        "training_time_min": float(total_time / 60),
        "dataset_size": int(len(X)),
        "total_params": ae.count_params(),
    }
    with open(os.path.join(AE_DIR, "training_log.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ AUTOENCODER TREINADO!")
    if history:
        print(f"   Loss: {history[0]:.6f} → {history[-1]:.6f}")
    print(f"   Tempo: {total_time/60:.1f} min")
    print(f"   Parâmetros: {ae.count_params():,}")
    print("=" * 60)

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
