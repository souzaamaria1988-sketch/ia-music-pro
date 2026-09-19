#!/usr/bin/env python3
"""
🧠 AUTOENCODER REAL PARA MÚSICA
Aprende a comprimir e reconstruir features musicais

Arquitetura:
  Input: 256 features (FFT)
  Encoder: 256 → 512 → 256 → 128 → 64
  Bottleneck: 64 (espaço latente)
  Decoder: 64 → 128 → 256 → 512 → 256
  Output: 256 features (reconstrução)
  
Loss: MSE(input, output)
Otimizador: Adam
"""
import os,sys,json,time,gc
import numpy as np
from pathlib import Path

MODEL_DIR="models"
AE_DIR=os.path.join(MODEL_DIR,"autoencoder")

class Autoencoder:
    """Autoencoder completo com encoder e decoder"""
    
    def __init__(self,input_size=256,hidden_sizes=[512,256,128],latent_size=64,seed=42):
        self.input_size=input_size
        self.hidden_sizes=hidden_sizes
        self.latent_size=latent_size
        np.random.seed(seed)
        
        # ENCODER: input → hidden1 → hidden2 → ... → latent
        self.encoder_weights=[]
        self.encoder_biases=[]
        self.encoder_m=[]  # Adam momentum
        self.encoder_v=[]  # Adam velocity
        
        # Primeira camada do encoder
        prev_size=input_size
        for hidden_size in hidden_sizes:
            w=np.random.randn(prev_size,hidden_size)*np.sqrt(2.0/prev_size)
            b=np.zeros(hidden_size)
            self.encoder_weights.append(w)
            self.encoder_biases.append(b)
            self.encoder_m.append(np.zeros_like(w))
            self.encoder_v.append(np.zeros_like(w))
            prev_size=hidden_size
        
        # Camada final do encoder (para o bottleneck)
        w=np.random.randn(prev_size,latent_size)*np.sqrt(2.0/prev_size)
        b=np.zeros(latent_size)
        self.encoder_weights.append(w)
        self.encoder_biases.append(b)
        self.encoder_m.append(np.zeros_like(w))
        self.encoder_v.append(np.zeros_like(w))
        
        # DECODER: latent → ... → hidden2 → hidden1 → input
        self.decoder_weights=[]
        self.decoder_biases=[]
        self.decoder_m=[]
        self.decoder_v=[]
        
        # Camadas do decoder (inverso do encoder)
        decoder_sizes=[latent_size]+list(reversed(hidden_sizes))+[input_size]
        prev_size=latent_size
        for i,output_size in enumerate(decoder_sizes[1:]):
            w=np.random.randn(prev_size,output_size)*np.sqrt(2.0/prev_size)
            b=np.zeros(output_size)
            self.decoder_weights.append(w)
            self.decoder_biases.append(b)
            self.decoder_m.append(np.zeros_like(w))
            self.decoder_v.append(np.zeros_like(w))
            prev_size=output_size
        
        # Adam parameters
        self.beta1=0.9
        self.beta2=0.999
        self.epsilon=1e-8
        self.t=0
        
        total_params=self.count_params()
        print(f"🧠 Autoencoder inicializado:")
        print(f"   Input: {input_size}")
        print(f"   Encoder: {input_size} → {' → '.join(map(str,hidden_sizes))} → {latent_size}")
        print(f"   Bottleneck: {latent_size} (espaço latente)")
        print(f"   Decoder: {latent_size} → {' → '.join(map(str,reversed(hidden_sizes)))} → {input_size}")
        print(f"   Total params: {total_params:,}")
    
    def count_params(self):
        count=0
        for w,b in zip(self.encoder_weights,self.encoder_biases):
            count+=w.size+b.size
        for w,b in zip(self.decoder_weights,self.decoder_biases):
            count+=w.size+b.size
        return count
    
    def relu(self,x):
        return np.maximum(0,x)
    
    def relu_derivative(self,x):
        return (x>0).astype(float)
    
    def encode(self,x):
        """Encoder: input → latent space"""
        self.encoder_activations=[x]
        current=x
        
        for i,(w,b) in enumerate(zip(self.encoder_weights,self.encoder_biases)):
            z=current@w+b
            if i<len(self.encoder_weights)-1:
                # Camadas ocultas: ReLU
                current=self.relu(z)
            else:
                # Última camada (bottleneck): linear ou tanh
                current=np.tanh(z)
            self.encoder_activations.append(current)
        
        return current  # Latent representation
    
    def decode(self,latent):
        """Decoder: latent → reconstructed output"""
        self.decoder_activations=[latent]
        current=latent
        
        for i,(w,b) in enumerate(zip(self.decoder_weights,self.decoder_biases)):
            z=current@w+b
            if i<len(self.decoder_weights)-1:
                # Camadas ocultas: ReLU
                current=self.relu(z)
            else:
                # Última camada: linear (reconstrução)
                current=z
            self.decoder_activations.append(current)
        
        return current  # Reconstructed output
    
    def forward(self,x):
        """Forward pass completo: input → latent → output"""
        latent=self.encode(x)
        output=self.decode(latent)
        return output,latent
    
    def compute_loss(self,x,output):
        """Loss: MSE entre input e output"""
        return np.mean((x-output)**2)
    
    def backward(self,x,output,latent,lr=0.001):
        """Backpropagation com Adam optimizer"""
        self.t+=1
        batch_size=x.shape[0]
        
        # Gradiente da loss (MSE)
        grad_output=2*(output-x)/batch_size
        
        # BACKWARD DECODER
        grad_latent=grad_output
        for i in range(len(self.decoder_weights)-1,-1,-1):
            w=self.decoder_weights[i]
            b=self.decoder_biases[i]
            activation=self.decoder_activations[i]
            
            # Gradiente dos pesos
            grad_w=activation.T@grad_output
            grad_b=np.mean(grad_output,axis=0)
            
            # Clip gradient
            grad_w=np.clip(grad_w,-1.0,1.0)
            
            # Adam update
            self.decoder_m[i]=self.beta1*self.decoder_m[i]+(1-self.beta1)*grad_w
            self.decoder_v[i]=self.beta2*self.decoder_v[i]+(1-self.beta2)*(grad_w**2)
            m_hat=self.decoder_m[i]/(1-self.beta1**self.t)
            v_hat=self.decoder_v[i]/(1-self.beta2**self.t)
            self.decoder_weights[i]-=lr*m_hat/(np.sqrt(v_hat)+self.epsilon)
            
            self.decoder_biases[i]-=lr*grad_b
            
            # Gradiente para próxima camada
            if i>0:
                grad_latent=grad_output@w.T
                # ReLU derivative
                if i<len(self.decoder_weights)-1:
                    grad_latent=grad_latent*self.relu_derivative(self.decoder_activations[i])
                grad_output=grad_latent
        
        # BACKWARD ENCODER
        # Gradiente do bottleneck (vem do decoder)
        grad_encoder=grad_latent
        # Tanh derivative na última camada do encoder
        grad_encoder=grad_encoder*(1-latent**2)
        
        for i in range(len(self.encoder_weights)-1,-1,-1):
            w=self.encoder_weights[i]
            b=self.encoder_biases[i]
            activation=self.encoder_activations[i]
            
            # Gradiente dos pesos
            grad_w=activation.T@grad_encoder
            grad_b=np.mean(grad_encoder,axis=0)
            
            # Clip gradient
            grad_w=np.clip(grad_w,-1.0,1.0)
            
            # Adam update
            self.encoder_m[i]=self.beta1*self.encoder_m[i]+(1-self.beta1)*grad_w
            self.encoder_v[i]=self.beta2*self.encoder_v[i]+(1-self.beta2)*(grad_w**2)
            m_hat=self.encoder_m[i]/(1-self.beta1**self.t)
            v_hat=self.encoder_v[i]/(1-self.beta2**self.t)
            self.encoder_weights[i]-=lr*m_hat/(np.sqrt(v_hat)+self.epsilon)
            
            self.encoder_biases[i]-=lr*grad_b
            
            # Gradiente para próxima camada
            if i>0:
                grad_encoder=grad_encoder@w.T
                grad_encoder=grad_encoder*self.relu_derivative(self.encoder_activations[i])
    
    def train_step(self,x,lr=0.001):
        """Um passo de treinamento"""
        output,latent=self.forward(x)
        loss=self.compute_loss(x,output)
        self.backward(x,output,latent,lr)
        return loss
    
    def save(self,filepath):
        """Salva o autoencoder"""
        save_dict={
            'input_size':np.array([self.input_size]),
            'latent_size':np.array([self.latent_size]),
            'hidden_sizes':np.array(self.hidden_sizes),
        }
        for i,(w,b) in enumerate(zip(self.encoder_weights,self.encoder_biases)):
            save_dict[f'encoder_w_{i}']=w.astype(np.float16)
            save_dict[f'encoder_b_{i}']=b.astype(np.float16)
        for i,(w,b) in enumerate(zip(self.decoder_weights,self.decoder_biases)):
            save_dict[f'decoder_w_{i}']=w.astype(np.float16)
            save_dict[f'decoder_b_{i}']=b.astype(np.float16)
        np.savez_compressed(filepath,**save_dict)
        size_mb=os.path.getsize(filepath)/1024/1024
        print(f"💾 Autoencoder salvo: {filepath} ({size_mb:.2f} MB)")
    
    def load(self,filepath):
        """Carrega o autoencoder"""
        data=np.load(filepath)
        self.input_size=int(data['input_size'][0])
        self.latent_size=int(data['latent_size'][0])
        self.hidden_sizes=list(data['hidden_sizes'])
        
        self.encoder_weights=[]
        self.encoder_biases=[]
        i=0
        while f'encoder_w_{i}' in data:
            self.encoder_weights.append(data[f'encoder_w_{i}'].astype(np.float32))
            self.encoder_biases.append(data[f'encoder_b_{i}'].astype(np.float32))
            i+=1
        
        self.decoder_weights=[]
        self.decoder_biases=[]
        i=0
        while f'decoder_w_{i}' in data:
            self.decoder_weights.append(data[f'decoder_w_{i}'].astype(np.float32))
            self.decoder_biases.append(data[f'decoder_b_{i}'].astype(np.float32))
            i+=1
        
        print(f"✅ Autoencoder carregado: {filepath}")
    
    def generate_from_latent(self,latent):
        """Gera features a partir do espaço latente"""
        return self.decode(latent)
    
    def interpolate(self,latent1,latent2,steps=10):
        """Interpola entre dois pontos do espaço latente"""
        results=[]
        for t in np.linspace(0,1,steps):
            latent=latent1*(1-t)+latent2*t
            output=self.decode(latent)
            results.append(output)
        return results

def extract_features(filepath,sr=22050,n_features=256):
    """Extrai features FFT de um arquivo de áudio"""
    try:
        import soundfile as sf
        audio,fsr=sf.read(filepath,dtype='float32')
        if len(audio.shape)>1:audio=np.mean(audio,axis=1)
        if fsr!=sr:
            idx=np.round(np.arange(0,len(audio),fsr/sr)).astype(int)
            audio=audio[idx[idx<len(audio)]]
    except:
        return None
    if len(audio)<sr:return None
    
    features=[]
    seg_len=sr
    n_segs=min(len(audio)//seg_len,20)
    for i in range(n_segs):
        seg=audio[i*seg_len:(i+1)*seg_len]
        fft=np.abs(np.fft.rfft(seg))[:n_features]
        fft=fft/(np.max(fft)+1e-10)
        if len(fft)<n_features:fft=np.pad(fft,(0,n_features-len(fft)))
        features.append(fft)
    return features

def generate_synthetic_data(n_samples=2000,n_features=256):
    """Gera dados sintéticos diversos para treinar"""
    print(f"  Gerando {n_samples} amostras sintéticas...")
    X=[]
    for i in range(n_samples):
        t=np.linspace(0,1,44100)
        signal=np.zeros_like(t)
        pattern_type=np.random.choice(['harmonic','noise','sweep','pulse','fm','granular','chord','bass','drum','melody','arp','pad'])
        
        if pattern_type=='harmonic':
            freq=np.random.uniform(80,3000)
            for h in range(1,np.random.randint(2,15)):signal+=np.sin(2*np.pi*freq*h*t)/h
        elif pattern_type=='noise':
            signal=np.random.randn(len(t))*np.exp(-t*np.random.uniform(1,10))
        elif pattern_type=='sweep':
            f0=np.random.uniform(50,500);f1=np.random.uniform(500,5000)
            freqs=np.linspace(f0,f1,len(t))
            signal=np.sin(2*np.pi*np.cumsum(freqs)/44100)
        elif pattern_type=='pulse':
            freq=np.random.uniform(2,20)
            signal=np.sin(2*np.pi*freq*t)*np.exp(-((t%0.5-0.25)**2)*50)
        elif pattern_type=='fm':
            fc=np.random.uniform(200,1000);fm=np.random.uniform(10,200)
            signal=np.sin(2*np.pi*fc*t+3*np.sin(2*np.pi*fm*t))
        elif pattern_type=='granular':
            for _ in range(20):
                pos=np.random.randint(0,len(t)-1000)
                grain=np.random.randn(1000)*np.hanning(1000)
                signal[pos:pos+1000]+=grain*0.3
        elif pattern_type=='chord':
            root=np.random.uniform(100,500)
            for interval in [0,4,7,11]:
                freq=root*(2**(interval/12))
                signal+=np.sin(2*np.pi*freq*t)*0.3
        elif pattern_type=='bass':
            freq=np.random.uniform(40,200)
            signal=np.sin(2*np.pi*freq*t)*np.exp(-t*3)
        elif pattern_type=='drum':
            freq=np.random.uniform(100,300)
            signal=np.sin(2*np.pi*freq*t)*np.exp(-t*20)
        elif pattern_type=='melody':
            freq=np.random.uniform(200,1000)
            signal=np.sin(2*np.pi*freq*t)*(1+0.3*np.sin(2*np.pi*5*t))
        elif pattern_type=='arp':
            base=np.random.uniform(200,600)
            for j in range(8):
                freq=base*(2**(np.random.choice([0,4,7,12])/12))
                start=j*len(t)//8
                end=(j+1)*len(t)//8
                signal[start:end]+=np.sin(2*np.pi*freq*t[start:end])*0.3
        else:  # pad
            freq=np.random.uniform(100,400)
            signal=np.sin(2*np.pi*freq*t)+0.5*np.sin(2*np.pi*freq*1.005*t)+0.5*np.sin(2*np.pi*freq*0.995*t)
        
        signal+=np.random.randn(len(t))*0.05
        fft=np.abs(np.fft.rfft(signal))[:n_features]
        fft=fft/(np.max(fft)+1e-10)
        if len(fft)<n_features:fft=np.pad(fft,(0,n_features-len(fft)))
        X.append(fft)
    return np.array(X)

def prepare_data(n_features=256):
    """Prepara dados para o autoencoder"""
    print("📊 Preparando dados para o autoencoder...")
    all_features=[]
    
    music_path=Path("music_input")
    if music_path.exists():
        files=[]
        for ext in ['*.mp3','*.wav','*.flac','*.ogg']:
            files.extend(list(music_path.glob(ext)))
        print(f"  {len(files)} músicas de referência")
        for f in files[:30]:
            feats=extract_features(f,n_features=n_features)
            if feats:all_features.extend(feats)
    
    if len(all_features)<100:
        synthetic=generate_synthetic_data(2000,n_features)
        all_features.extend(synthetic.tolist())
    
    X=np.array(all_features)
    print(f"  Dataset: {X.shape[0]} x {X.shape[1]}")
    
    # Normalizar
    X_mean=np.mean(X,axis=0)
    X_std=np.std(X,axis=0)+1e-8
    X_norm=(X-X_mean)/X_std
    
    return X_norm,X_mean,X_std

def train_autoencoder(epochs=10000,batch_size=64,n_features=256,latent_size=64,lr_init=0.001):
    """Treina o autoencoder"""
    print("="*60)
    print("🧠 TREINAMENTO AUTOENCODER")
    print(f"   Épocas: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Features: {n_features}")
    print(f"   Latent size: {latent_size}")
    print(f"   Loss: MSE (reconstrução)")
    print("="*60)
    
    os.makedirs(AE_DIR,exist_ok=True)
    
    # Preparar dados
    X,X_mean,X_std=prepare_data(n_features)
    
    # Criar autoencoder
    ae=Autoencoder(input_size=n_features,hidden_sizes=[512,256,128],latent_size=latent_size)
    
    print("\n🚀 TREINANDO AUTOENCODER...")
    history=[];best_loss=float('inf');start=time.time()
    
    for epoch in range(1,epochs+1):
        lr=max(lr_init*(0.9995**epoch),0.00001)
        indices=np.random.permutation(len(X))
        epoch_loss=0;n_batches=0
        
        for i in range(0,len(X),batch_size):
            idx=indices[i:i+batch_size]
            x_batch=X[idx]
            loss=ae.train_step(x_batch,lr)
            epoch_loss+=loss
            n_batches+=1
        
        avg_loss=epoch_loss/n_batches
        history.append(float(avg_loss))
        
        if epoch%100==0 or epoch==1:
            elapsed=time.time()-start
            eta=(elapsed/epoch)*(epochs-epoch)
            print(f"  Epoch {epoch:6d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr:.6f} | ETA: {eta/60:.1f}min")
        
        # Checkpoint a cada 1000 épocas
        if epoch%1000==0:
            ae.save(os.path.join(AE_DIR,f"checkpoint_{epoch}.npz"))
            print(f"  💾 Checkpoint salvo (epoch {epoch})")
        
        if avg_loss<best_loss:
            best_loss=avg_loss
            ae.save(os.path.join(AE_DIR,"best_autoencoder.npz"))
    
    # Salvar modelo final
    ae.save(os.path.join(AE_DIR,"final_autoencoder.npz"))
    
    # Salvar normalização
    np.savez(os.path.join(AE_DIR,"normalization.npz"),mean=X_mean,std=X_std)
    
    # Salvar metadata
    total_time=time.time()-start
    metadata={
        "type":"Autoencoder",
        "epochs":epochs,
        "input_size":n_features,
        "latent_size":latent_size,
        "hidden_sizes":[512,256,128],
        "final_loss":float(history[-1]),
        "best_loss":float(best_loss),
        "training_time_min":float(total_time/60),
        "dataset_size":int(len(X)),
        "total_params":ae.count_params(),
        "history_sample":history[::max(1,len(history)//100)]
    }
    with open(os.path.join(AE_DIR,"training_log.json"),'w') as f:
        json.dump(metadata,f,indent=2)
    
    print("\n"+"="*60)
    print("✅ AUTOENCODER TREINADO!")
    print(f"   Loss: {history[0]:.6f} → {history[-1]:.6f}")
    print(f"   Tempo: {total_time/60:.1f} min")
    print(f"   Parâmetros: {ae.count_params():,}")
    print(f"   Latent size: {latent_size}")
    print("="*60)
    
    return ae

def load_trained_autoencoder():
    """Carrega autoencoder treinado"""
    ae_path=os.path.join(AE_DIR,"best_autoencoder.npz")
    if not os.path.exists(ae_path):
        print("⚠️ Autoencoder não treinado ainda")
        return None
    
    ae=Autoencoder()
    ae.load(ae_path)
    return ae

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=10000)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--features",type=int,default=256)
    parser.add_argument("--latent",type=int,default=64)
    args=parser.parse_args()
    
    train_autoencoder(
        epochs=max(args.epochs,10000),
        batch_size=args.batch_size,
        n_features=args.features,
        latent_size=args.latent
    )
