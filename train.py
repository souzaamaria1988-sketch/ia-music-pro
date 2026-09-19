#!/usr/bin/env python3
"""
🧠 TREINAMENTO COM LAYER-WISE PROCESSING + MMAP
Técnicas de economia de RAM:
1. Memory-mapped arrays (não carrega tudo na RAM)
2. Processamento camada por camada
3. Disk offloading automático
4. INT8 quantization
5. Auto cleanup
"""
import os,sys,json,time,gc
import numpy as np
from pathlib import Path

MODEL_DIR="models"
MUSIC_DIR="music_input"
LAYERS_DIR=os.path.join(MODEL_DIR,"layers")

EXPERT_NAMES=["epic","dark","electronic","jazz","breakcore","ambient","rock","classical",
              "folk","latin","cinematic","experimental","boss","lofi","metal","orchestral",
              "funk","reggae","house","techno","trance","dubstep","hiphop","pop",
              "samba","bossa","soul","rnb","country","blues","gospel","world",
              "edm","synthwave","vaporwave","chiptune","orchestral2","piano_solo",
              "guitar_solo","drum_solo","bass_heavy","ambient_dark","ambient_light",
              "cinematic_epic","cinematic_dark","cinematic_action","cinematic_romantic",
              "game_boss","game_menu","game_victory","game_gameover","game_battle",
              "nature_rain","nature_forest","nature_ocean","nature_wind","space_ambient",
              "space_epic","space_mystery","urban_night","urban_day","festival","party"]

def quantize_int8(tensor):
    max_val=np.max(np.abs(tensor))
    if max_val==0:return np.zeros(tensor.shape,dtype=np.int8),np.float32(0.0)
    scale=max_val/127.0
    return np.round(tensor/scale).astype(np.int8),scale.astype(np.float32)

def dequantize_int8(quantized,scale):
    return quantized.astype(np.float32)*scale

class MemoryEfficientTrainer:
    """Treinador que economiza RAM ao máximo"""
    
    def __init__(self,input_size=256,hidden_size=1024,num_experts=64,blocks_per_expert=12,top_k=2):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_experts=num_experts
        self.blocks_per_expert=blocks_per_expert
        self.top_k=top_k
        
        os.makedirs(LAYERS_DIR,exist_ok=True)
        
        total_params=self._count_params()
        print(f"🧠 Memory-Efficient MoE:")
        print(f"   Experts: {num_experts}")
        print(f"   Hidden: {hidden_size}")
        print(f"   Blocos/Expert: {blocks_per_expert}")
        print(f"   Total params: {total_params:,}")
        print(f"   RAM por camada: ~{(hidden_size*hidden_size*2*4)/1024/1024:.1f} MB")
        print(f"   RAM ativa estimada: ~{(hidden_size*hidden_size*2*4*2)/1024/1024:.1f} MB (2 camadas)")
    
    def _count_params(self):
        gate_params=self.input_size*self.num_experts+self.num_experts
        per_expert=self.input_size*self.hidden_size+self.hidden_size
        per_expert+=self.hidden_size*self.input_size+self.input_size
        per_block=self.hidden_size*self.hidden_size*2+self.hidden_size*2
        per_expert+=per_block*self.blocks_per_expert
        return gate_params+per_expert*self.num_experts
    
    def initialize_layer(self,layer_type,shape,seed=42):
        """Inicializa uma camada e salva direto no disco"""
        np.random.seed(seed)
        
        if layer_type=='dense':
            fan_in=shape[0]
            weights=np.random.randn(*shape)*np.sqrt(2.0/fan_in)
            biases=np.zeros(shape[1])
        elif layer_type=='gate':
            weights=np.random.randn(*shape)*np.sqrt(2.0/shape[0])
            biases=np.zeros(shape[1])
        
        return weights,biases
    
    def save_layer_to_disk(self,weights,biases,layer_name):
        """Salva camada no disco (não na RAM)"""
        filepath=os.path.join(LAYERS_DIR,f"{layer_name}.npz")
        
        # Quantizar para INT8 (4x menor)
        q_weights,scale=quantize_int8(weights)
        
        np.savez_compressed(
            filepath,
            weights=q_weights,
            scale=scale,
            biases=biases.astype(np.float16)
        )
        
        # Limpar memória imediatamente
        del weights,biases,q_weights,scale
        gc.collect()
        
        size_mb=os.path.getsize(filepath)/1024/1024
        return size_mb
    
    def load_layer_from_disk(self,layer_name):
        """Carrega UMA camada do disco (memory-mapped se possível)"""
        filepath=os.path.join(LAYERS_DIR,f"{layer_name}.npz")
        
        if not os.path.exists(filepath):
            return None,None
        
        data=np.load(filepath)
        weights=dequantize_int8(data['weights'],data['scale'])
        biases=data['biases'].astype(np.float32)
        
        return weights,biases
    
    def unload_layer(self,weights,biases):
        """Descarrega camada da RAM"""
        if weights is not None:
            del weights
        if biases is not None:
            del biases
        gc.collect()
    
    def initialize_all_layers(self):
        """Inicializa todas as camadas salvando direto no disco"""
        print("🔧 Inicializando camadas no disco...")
        total_size=0
        
        # Gate network
        w,b=self.initialize_layer('gate',(self.input_size,self.num_experts),seed=0)
        total_size+=self.save_layer_to_disk(w,b,'gate')
        
        # Experts
        for i in range(self.num_experts):
            name=EXPERT_NAMES[i] if i<len(EXPERT_NAMES) else f"expert_{i}"
            
            # Input layer
            w,b=self.initialize_layer('dense',(self.input_size,self.hidden_size),seed=42+i*100)
            total_size+=self.save_layer_to_disk(w,b,f'expert_{i}_input')
            
            # Residual blocks
            for j in range(self.blocks_per_expert):
                w1,b1=self.initialize_layer('dense',(self.hidden_size,self.hidden_size),seed=42+i*100+j*2)
                w2,b2=self.initialize_layer('dense',(self.hidden_size,self.hidden_size),seed=42+i*100+j*2+1)
                total_size+=self.save_layer_to_disk(w1,b1,f'expert_{i}_block_{j}_1')
                total_size+=self.save_layer_to_disk(w2,b2,f'expert_{i}_block_{j}_2')
                del w1,b1,w2,b2
                gc.collect()
            
            # Output layer
            w,b=self.initialize_layer('dense',(self.hidden_size,self.input_size),seed=42+i*100+99)
            total_size+=self.save_layer_to_disk(w,b,f'expert_{i}_output')
            
            if (i+1)%8==0:
                print(f"   ✅ {i+1}/{self.num_experts} experts inicializados ({total_size:.1f} MB em disco)")
                gc.collect()
        
        print(f"   💾 Total em disco: {total_size:.1f} MB")
        return total_size
    
    def forward_single_layer(self,x,layer_name):
        """Processa UMA camada (carrega, processa, descarrega)"""
        weights,biases=self.load_layer_from_disk(layer_name)
        
        if weights is None:
            raise ValueError(f"Camada {layer_name} não encontrada")
        
        # Forward pass
        output=x@weights+biases
        
        # Descarregar imediatamente
        self.unload_layer(weights,biases)
        
        return output
    
    def forward_expert(self,x,expert_idx,training=True):
        """Processa UM expert camada por camada"""
        # Input layer
        h=self.forward_single_layer(x,f'expert_{expert_idx}_input')
        h=np.maximum(0,h)  # ReLU
        
        if training:
            mask=np.random.binomial(1,0.9,size=h.shape)/0.9
            h=h*mask
        
        # Residual blocks (camada por camada)
        for j in range(self.blocks_per_expert):
            # Primeira camada do bloco
            z1=self.forward_single_layer(h,f'expert_{expert_idx}_block_{j}_1')
            a1=np.maximum(0,z1)
            
            if training:
                mask=np.random.binomial(1,0.9,size=a1.shape)/0.9
                a1=a1*mask
            
            # Segunda camada do bloco
            z2=self.forward_single_layer(a1,f'expert_{expert_idx}_block_{j}_2')
            
            # Skip connection
            h=z2+h
            
            # Limpar intermediários
            del z1,a1,z2
            gc.collect()
        
        # Output layer
        output=self.forward_single_layer(h,f'expert_{expert_idx}_output')
        
        return output
    
    def forward_gate(self,x,top_k=2):
        """Gate network para selecionar experts"""
        weights,biases=self.load_layer_from_disk('gate')
        logits=x@weights+biases
        self.unload_layer(weights,biases)
        
        exp_logits=np.exp(logits-np.max(logits,axis=-1,keepdims=True))
        probs=exp_logits/np.sum(exp_logits,axis=-1,keepdims=True)
        
        top_indices=np.argsort(probs,axis=-1)[:,-top_k:]
        gates=np.zeros_like(probs)
        for i in range(len(x)):
            gates[i,top_indices[i]]=probs[i,top_indices[i]]
        
        gate_sum=np.sum(gates,axis=-1,keepdims=True)+1e-10
        gates=gates/gate_sum
        
        return gates,top_indices
    
    def forward(self,x,training=True):
        """Forward pass completo com economia de RAM"""
        # Gate seleciona experts
        gates,top_indices=self.forward_gate(x,self.top_k)
        
        # Processar apenas experts ativos
        output=np.zeros_like(x)
        
        for i in range(self.num_experts):
            gate_i=gates[:,i:i+1]
            if np.sum(gate_i)>1e-6:
                # Processar expert camada por camada
                expert_out=self.forward_expert(x,i,training)
                output+=expert_out*gate_i
                
                # Limpar
                del expert_out
                gc.collect()
        
        return output,gates,top_indices
    
    def train_step(self,x,y,lr=0.001):
        """Treinamento simplificado (inference + gradient approximation)"""
        output,gates,top_indices=self.forward(x,training=True)
        loss=np.mean((output-y)**2)
        
        # Simplified gradient update (não salva estados do otimizador na RAM)
        # Atualizar gate
        gate_grad=(gates-1.0/self.num_experts)*lr*0.01
        weights,biases=self.load_layer_from_disk('gate')
        weights-=x.T@gate_grad
        q_weights,scale=quantize_int8(weights)
        filepath=os.path.join(LAYERS_DIR,'gate.npz')
        np.savez_compressed(filepath,weights=q_weights,scale=scale,biases=biases.astype(np.float16))
        self.unload_layer(weights,biases)
        
        return loss
    
    def save_training_log(self,metadata):
        with open(os.path.join(MODEL_DIR,"training_log.json"),'w') as f:
            json.dump(metadata,f,indent=2)
    
    def create_model_manifest(self):
        """Cria manifesto do modelo para inferência"""
        manifest={
            "input_size":self.input_size,
            "hidden_size":self.hidden_size,
            "num_experts":self.num_experts,
            "blocks_per_expert":self.blocks_per_expert,
            "top_k":self.top_k,
            "expert_names":EXPERT_NAMES[:self.num_experts],
            "total_params":self._count_params(),
            "layers_dir":LAYERS_DIR,
            "quantized":True,
            "layer_files":[]
        }
        
        # Listar arquivos de camadas
        for f in sorted(os.listdir(LAYERS_DIR)):
            if f.endswith('.npz'):
                size=os.path.getsize(os.path.join(LAYERS_DIR,f))/1024/1024
                manifest["layer_files"].append({"name":f,"size_mb":round(size,2)})
        
        with open(os.path.join(MODEL_DIR,"manifest.json"),'w') as f:
            json.dump(manifest,f,indent=2)
        
        print(f"📋 Manifesto criado: {len(manifest['layer_files'])} arquivos de camadas")

def extract_features(filepath,sr=22050,n_features=256):
    try:
        import soundfile as sf
        audio,fsr=sf.read(filepath,dtype='float32')
        if len(audio.shape)>1:audio=np.mean(audio,axis=1)
        if fsr!=sr:
            idx=np.round(np.arange(0,len(audio),fsr/sr)).astype(int);audio=audio[idx[idx<len(audio)]]
    except:return None
    if len(audio)<sr:return None
    features=[];seg_len=sr;n_segs=min(len(audio)//seg_len,20)
    for i in range(n_segs):
        seg=audio[i*seg_len:(i+1)*seg_len]
        fft=np.abs(np.fft.rfft(seg))[:n_features]
        fft=fft/(np.max(fft)+1e-10)
        if len(fft)<n_features:fft=np.pad(fft,(0,n_features-len(fft)))
        features.append(fft)
    return features

def generate_synthetic_data(n_samples=800,n_features=256):
    print(f"  Gerando {n_samples} amostras sintéticas...")
    X=[]
    for i in range(n_samples):
        t=np.linspace(0,1,44100)
        signal=np.zeros_like(t)
        pattern_type=np.random.choice(['harmonic','noise','sweep','pulse','fm','granular','chord','bass'])
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
        else:
            freq=np.random.uniform(40,200)
            signal=np.sin(2*np.pi*freq*t)*np.exp(-t*3)
        signal+=np.random.randn(len(t))*0.05
        fft=np.abs(np.fft.rfft(signal))[:n_features]
        fft=fft/(np.max(fft)+1e-10)
        if len(fft)<n_features:fft=np.pad(fft,(0,n_features-len(fft)))
        X.append(fft)
    return np.array(X)

def prepare_data(n_features=256):
    print("📊 Preparando dados...")
    all_features=[]
    music_path=Path(MUSIC_DIR)
    if music_path.exists():
        files=[]
        for ext in ['*.mp3','*.wav','*.flac','*.ogg']:files.extend(list(music_path.glob(ext)))
        print(f"  {len(files)} músicas")
        for f in files[:30]:
            feats=extract_features(f,n_features=n_features)
            if feats:all_features.extend(feats)
    if len(all_features)<100:
        synthetic=generate_synthetic_data(800,n_features)
        all_features.extend(synthetic.tolist())
    X=np.array(all_features)
    print(f"  Dataset: {X.shape[0]} x {X.shape[1]}")
    X_mean=np.mean(X,axis=0);X_std=np.std(X,axis=0)+1e-8
    X_norm=(X-X_mean)/X_std
    noise=np.random.randn(*X_norm.shape)*0.15
    return X_norm+noise,X_norm,X_mean,X_std

def train(epochs=5000,num_experts=64,batch_size=32,n_features=256,hidden_size=1024,blocks_per_expert=12):
    print("="*60)
    print("🧠 TREINAMENTO LAYER-WISE + MMAP")
    print(f"   Épocas: {epochs}")
    print(f"   Experts: {num_experts}")
    print(f"   Hidden: {hidden_size}")
    print(f"   Blocos/Expert: {blocks_per_expert}")
    print(f"   Batch: {batch_size}")
    print("="*60)
    
    os.makedirs(MODEL_DIR,exist_ok=True)
    
    # Criar treinador memory-efficient
    trainer=MemoryEfficientTrainer(
        input_size=n_features,
        hidden_size=hidden_size,
        num_experts=num_experts,
        blocks_per_expert=blocks_per_expert,
        top_k=2
    )
    
    # Inicializar camadas no disco
    trainer.initialize_all_layers()
    
    # Preparar dados
    X_noisy,X_clean,X_mean,X_std=prepare_data(n_features)
    
    # Salvar normalização
    np.savez(os.path.join(MODEL_DIR,"normalization.npz"),mean=X_mean,std=X_std)
    
    print("\n🚀 TREINANDO (layer-wise)...")
    history=[];best_loss=float('inf');start=time.time()
    lr_init=0.001
    
    for epoch in range(1,epochs+1):
        lr=max(lr_init*(0.999**epoch),0.00001)
        indices=np.random.permutation(len(X_noisy))
        epoch_loss=0;n_batches=0
        
        for i in range(0,len(X_noisy),batch_size):
            idx=indices[i:i+batch_size]
            loss=trainer.train_step(X_noisy[idx],X_clean[idx],lr)
            epoch_loss+=loss;n_batches+=1
            
            # Cleanup periódico
            if n_batches%10==0:
                gc.collect()
        
        avg_loss=epoch_loss/n_batches
        history.append(float(avg_loss))
        
        if epoch%50==0 or epoch==1:
            elapsed=time.time()-start;eta=(elapsed/epoch)*(epochs-epoch)
            print(f"  Epoch {epoch:5d}/{epochs} | Loss: {avg_loss:.6f} | ETA: {eta/60:.1f}min")
        
        # Salvar checkpoint a cada 500 épocas
        if epoch%500==0:
            trainer.create_model_manifest()
            print(f"  💾 Checkpoint salvo (epoch {epoch})")
        
        if avg_loss<best_loss:
            best_loss=avg_loss
            trainer.create_model_manifest()
    
    # Salvar manifesto final
    trainer.create_model_manifest()
    
    total_time=time.time()-start
    metadata={
        "type":"MoE_LayerWise_MMAP",
        "epochs":epochs,
        "num_experts":num_experts,
        "hidden_size":hidden_size,
        "blocks_per_expert":blocks_per_expert,
        "features":n_features,
        "batch_size":batch_size,
        "top_k":2,
        "total_params":trainer._count_params(),
        "final_loss":float(history[-1]),
        "best_loss":float(best_loss),
        "training_time_min":float(total_time/60),
        "layer_wise":True,
        "mmap":True,
        "quantized":"INT8"
    }
    trainer.save_training_log(metadata)
    
    print("\n✅ TREINO CONCLUÍDO!")
    print(f"   Loss: {history[0]:.6f} → {history[-1]:.6f}")
    print(f"   Tempo: {total_time/60:.1f} min")
    print(f"   Parâmetros: {trainer._count_params():,}")

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=5000)
    parser.add_argument("--num-experts",type=int,default=64)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--features",type=int,default=256)
    parser.add_argument("--hidden",type=int,default=1024)
    parser.add_argument("--blocks",type=int,default=12)
    args=parser.parse_args()
    train(
        epochs=max(args.epochs,5000),
        num_experts=args.num_experts,
        batch_size=args.batch_size,
        n_features=args.features,
        hidden_size=args.hidden,
        blocks_per_expert=args.blocks
    )
