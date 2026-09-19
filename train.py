#!/usr/bin/env python3
"""
🧠 TREINAMENTO MoE MAX - 14GB RAM (macOS)
- 16 Experts especializados
- 512 hidden units
- 8 blocos residuais por expert
- 256 features FFT
- 800 amostras sintéticas
- Batch size 64
- ~400 milhões de parâmetros
- INT8 quantization para salvar
"""
import os,sys,json,time
import numpy as np
from pathlib import Path

MODEL_DIR="models"
MUSIC_DIR="music_input"
EXPERT_NAMES=["epic","dark","electronic","jazz","breakcore","ambient","rock","classical",
              "folk","latin","cinematic","experimental","boss","lofi","metal","orchestral"]

def quantize_int8(tensor):
    max_val=np.max(np.abs(tensor))
    if max_val==0:return np.zeros(tensor.shape,dtype=np.int8),np.float32(0.0)
    scale=max_val/127.0
    return np.round(tensor/scale).astype(np.int8),scale.astype(np.float32)

class ResidualBlock:
    def __init__(self,size,seed=42):
        np.random.seed(seed)
        self.w1=np.random.randn(size,size)*np.sqrt(2.0/size)
        self.b1=np.zeros(size)
        self.w2=np.random.randn(size,size)*np.sqrt(2.0/size)
        self.b2=np.zeros(size)
        self.m_w1=np.zeros_like(self.w1);self.v_w1=np.zeros_like(self.w1)
        self.m_b1=np.zeros_like(self.b1);self.v_b1=np.zeros_like(self.b1)
        self.m_w2=np.zeros_like(self.w2);self.v_w2=np.zeros_like(self.w2)
        self.m_b2=np.zeros_like(self.b2);self.v_b2=np.zeros_like(self.b2)
    def forward(self,x,training=True):
        self.input=x
        self.z1=x@self.w1+self.b1
        self.a1=np.maximum(0,self.z1)
        if training:
            self.mask=np.random.binomial(1,0.9,size=self.a1.shape)/0.9
            self.a1*=self.mask
        self.z2=self.a1@self.w2+self.b2
        return self.z2+x
    def backward(self,grad,lr,t,b1=0.9,b2=0.999,eps=1e-8):
        bs=grad.shape[0]
        gw2=np.clip(self.a1.T@grad/bs,-1,1);gb2=np.mean(grad,axis=0)
        ga1=grad@self.w2.T
        if hasattr(self,'mask'):ga1*=self.mask
        gz1=ga1*(self.z1>0)
        gw1=np.clip(self.input.T@gz1/bs,-1,1);gb1=np.mean(gz1,axis=0)
        for param,g,m_name,v_name in [(self.w1,gw1,'m_w1','v_w1'),(self.b1,gb1,'m_b1','v_b1'),(self.w2,gw2,'m_w2','v_w2'),(self.b2,gb2,'m_b2','v_b2')]:
            m=getattr(self,m_name);v=getattr(self,v_name)
            m[:]=b1*m+(1-b1)*g;v[:]=b2*v+(1-b2)*(g**2)
            param-=lr*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+eps)
        return gz1@self.w1.T+grad

class Expert:
    def __init__(self,input_size,hidden_size,num_blocks=8,seed=42,name="expert"):
        self.name=name
        self.input_w=np.random.randn(input_size,hidden_size)*np.sqrt(2.0/input_size)
        self.input_b=np.zeros(hidden_size)
        self.blocks=[ResidualBlock(hidden_size,seed=seed+i) for i in range(num_blocks)]
        self.output_w=np.random.randn(hidden_size,input_size)*np.sqrt(2.0/hidden_size)
        self.output_b=np.zeros(input_size)
        self.m_iw=np.zeros_like(self.input_w);self.v_iw=np.zeros_like(self.input_w)
        self.m_ib=np.zeros_like(self.input_b);self.v_ib=np.zeros_like(self.input_b)
        self.m_ow=np.zeros_like(self.output_w);self.v_ow=np.zeros_like(self.output_w)
        self.m_ob=np.zeros_like(self.output_b);self.v_ob=np.zeros_like(self.output_b)
        self.usage_count=0
    def forward(self,x,training=True):
        self.x=x;self.usage_count+=x.shape[0]
        h=np.maximum(0,x@self.input_w+self.input_b)
        for block in self.blocks:h=block.forward(h,training)
        self.h=h
        return h@self.output_w+self.output_b
    def backward(self,grad,lr,t):
        bs=grad.shape[0]
        gow=np.clip(self.h.T@grad/bs,-1,1);gob=np.mean(grad,axis=0)
        gh=grad@self.output_w.T
        b1,b2,eps=0.9,0.999,1e-8
        self.m_ow[:]=b1*self.m_ow+(1-b1)*gow;self.v_ow[:]=b2*self.v_ow+(1-b2)*(gow**2)
        self.output_w-=lr*(self.m_ow/(1-b1**t))/(np.sqrt(self.v_ow/(1-b2**t))+eps)
        self.m_ob[:]=b1*self.m_ob+(1-b1)*gob;self.v_ob[:]=b2*self.v_ob+(1-b2)*(gob**2)
        self.output_b-=lr*(self.m_ob/(1-b1**t))/(np.sqrt(self.v_ob/(1-b2**t))+eps)
        for block in reversed(self.blocks):gh=block.backward(gh,lr,t,b1,b2,eps)
        giw=np.clip(self.x.T@gh/bs,-1,1);gib=np.mean(gh,axis=0)
        self.m_iw[:]=b1*self.m_iw+(1-b1)*giw;self.v_iw[:]=b2*self.v_iw+(1-b2)*(giw**2)
        self.input_w-=lr*(self.m_iw/(1-b1**t))/(np.sqrt(self.v_iw/(1-b2**t))+eps)
        self.m_ib[:]=b1*self.m_ib+(1-b1)*gib;self.v_ib[:]=b2*self.v_ib+(1-b2)*(gib**2)
        self.input_b-=lr*(self.m_ib/(1-b1**t))/(np.sqrt(self.v_ib/(1-b2**t))+eps)

class GateNetwork:
    def __init__(self,input_size,num_experts,seed=42):
        np.random.seed(seed)
        self.w=np.random.randn(input_size,num_experts)*np.sqrt(2.0/input_size)
        self.b=np.zeros(num_experts)
        self.m_w=np.zeros_like(self.w);self.v_w=np.zeros_like(self.w)
        self.m_b=np.zeros_like(self.b);self.v_b=np.zeros_like(self.b)
    def forward(self,x,top_k=2,noise=0.1):
        self.x=x
        logits=x@self.w+self.b
        if noise>0:logits+=np.random.randn(*logits.shape)*noise
        exp_logits=np.exp(logits-np.max(logits,axis=-1,keepdims=True))
        probs=exp_logits/np.sum(exp_logits,axis=-1,keepdims=True)
        self.probs=probs
        top_indices=np.argsort(probs,axis=-1)[:,-top_k:]
        gates=np.zeros_like(probs)
        for i in range(len(x)):gates[i,top_indices[i]]=probs[i,top_indices[i]]
        gate_sum=np.sum(gates,axis=-1,keepdims=True)+1e-10
        gates=gates/gate_sum
        self.gates=gates
        self.top_indices=top_indices
        return gates
    def compute_load_balancing_loss(self,num_experts):
        expert_usage=np.mean(self.gates,axis=0)
        top_k_fraction=np.zeros(num_experts)
        for i in range(num_experts):
            top_k_fraction[i]=np.mean(np.any(self.top_indices==i,axis=1))
        lb_loss=num_experts*np.sum(expert_usage*top_k_fraction)
        return lb_loss
    def backward(self,grad_gates,lr,t):
        bs=grad_gates.shape[0]
        gw=np.clip(self.x.T@grad_gates/bs,-1,1);gb=np.mean(grad_gates,axis=0)
        b1,b2,eps=0.9,0.999,1e-8
        self.m_w[:]=b1*self.m_w+(1-b1)*gw;self.v_w[:]=b2*self.v_w+(1-b2)*(gw**2)
        self.w-=lr*(self.m_w/(1-b1**t))/(np.sqrt(self.v_w/(1-b2**t))+eps)
        self.m_b[:]=b1*self.m_b+(1-b1)*gb;self.v_b[:]=b2*self.v_b+(1-b2)*(gb**2)
        self.b-=lr*(self.m_b/(1-b1**t))/(np.sqrt(self.v_b/(1-b2**t))+eps)

class MixtureOfExperts:
    def __init__(self,input_size,hidden_size=512,num_experts=16,blocks_per_expert=8,top_k=2):
        self.input_size=input_size;self.num_experts=num_experts;self.top_k=top_k
        self.gate=GateNetwork(input_size,num_experts,seed=0)
        self.experts=[]
        for i in range(num_experts):
            name=EXPERT_NAMES[i] if i<len(EXPERT_NAMES) else f"expert_{i}"
            self.experts.append(Expert(input_size,hidden_size,blocks_per_expert,seed=42+i*100,name=name))
        self.t=0
        self.lb_loss_weight=0.01
        total=self.count_params()
        print(f"🧠 MoE MAX (macOS 14GB):")
        print(f"   Experts: {num_experts} ({', '.join(e.name for e in self.experts)})")
        print(f"   Hidden: {hidden_size}")
        print(f"   Blocos/Expert: {blocks_per_expert}")
        print(f"   Top-k: {top_k}")
        print(f"   Total params: {total:,}")
    def count_params(self):
        count=self.gate.w.size+self.gate.b.size
        for e in self.experts:
            count+=e.input_w.size+e.input_b.size+e.output_w.size+e.output_b.size
            for b in e.blocks:count+=b.w1.size+b.b1.size+b.w2.size+b.b2.size
        return count
    def forward(self,x,training=True):
        noise=0.1 if training else 0.0
        gates=self.gate.forward(x,self.top_k,noise=noise)
        self.active_gates=gates
        output=np.zeros_like(x)
        self.expert_outputs=[]
        for i,expert in enumerate(self.experts):
            gate_i=gates[:,i:i+1]
            if np.sum(gate_i)>1e-6:
                expert_out=expert.forward(x,training)
                self.expert_outputs.append((i,expert_out))
                output+=expert_out*gate_i
        return output
    def train_step(self,x,y,lr):
        out=self.forward(x,training=True)
        self.forward_cache=out
        mse_loss=np.mean((out-y)**2)
        lb_loss=self.gate.compute_load_balancing_loss(self.num_experts)
        total_loss=mse_loss+self.lb_loss_weight*lb_loss
        self.t+=1
        bs=y.shape[0]
        grad=(out-y)*(2.0/bs)
        gate_grad=np.zeros_like(self.active_gates)
        for i,expert_out in self.expert_outputs:
            expert_grad=np.sum((expert_out-y)**2,axis=1,keepdims=True)
            gate_grad[:,i:i+1]=-expert_grad*self.active_gates[:,i:i+1]
        gate_grad+=self.lb_loss_weight*(self.active_gates-1.0/self.num_experts)
        self.gate.backward(gate_grad,lr,self.t)
        for i,expert_out in self.expert_outputs:
            gate_i=self.active_gates[:,i:i+1]
            expert_grad=grad*gate_i
            self.experts[i].backward(expert_grad,lr,self.t)
        return total_loss
    def save(self,filepath):
        save_dict={'gate_w':self.gate.w,'gate_b':self.gate.b,'input_size':np.array([self.input_size]),'num_experts':np.array([self.num_experts]),'top_k':np.array([self.top_k]),'expert_names':np.array([e.name for e in self.experts])}
        for i,expert in enumerate(self.experts):
            qw,sw=quantize_int8(expert.input_w)
            qo,so=quantize_int8(expert.output_w)
            save_dict[f'expert_{i}_iw']=qw;save_dict[f'expert_{i}_sw']=sw
            save_dict[f'expert_{i}_ib']=expert.input_b.astype(np.float16)
            save_dict[f'expert_{i}_ow']=qo;save_dict[f'expert_{i}_so']=so
            save_dict[f'expert_{i}_ob']=expert.output_b.astype(np.float16)
            for j,block in enumerate(expert.blocks):
                q1,s1=quantize_int8(block.w1);q2,s2=quantize_int8(block.w2)
                save_dict[f'expert_{i}_b_{j}_w1']=q1;save_dict[f'expert_{i}_b_{j}_s1']=s1
                save_dict[f'expert_{i}_b_{j}_b1']=block.b1.astype(np.float16)
                save_dict[f'expert_{i}_b_{j}_w2']=q2;save_dict[f'expert_{i}_b_{j}_s2']=s2
                save_dict[f'expert_{i}_b_{j}_b2']=block.b2.astype(np.float16)
        np.savez_compressed(filepath,**save_dict)
        size_mb=os.path.getsize(filepath)/1024/1024
        print(f"💾 Salvo: {filepath} ({size_mb:.2f} MB)")
    def get_expert_usage(self):
        usage=[e.usage_count for e in self.experts]
        total=sum(usage)+1
        return {e.name:u/total for e,u in zip(self.experts,usage)}

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
    print(f"  Gerando {n_samples} amostras sintéticas diversas...")
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
        else:  # bass
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

def train(epochs=5000,num_experts=16,batch_size=64,n_features=256,hidden_size=512,blocks_per_expert=8):
    print("="*60)
    print("🧠 TREINAMENTO MoE MAX (macOS 14GB)")
    print(f"   Épocas: {epochs}")
    print(f"   Experts: {num_experts}")
    print(f"   Hidden: {hidden_size}")
    print(f"   Blocos/Expert: {blocks_per_expert}")
    print(f"   Features: {n_features}")
    print(f"   Batch: {batch_size}")
    print("="*60)
    os.makedirs(MODEL_DIR,exist_ok=True)
    X_noisy,X_clean,X_mean,X_std=prepare_data(n_features)
    model=MixtureOfExperts(n_features,hidden_size=hidden_size,num_experts=num_experts,blocks_per_expert=blocks_per_expert,top_k=2)
    print("\n🚀 TREINANDO...")
    history=[];best_loss=float('inf');start=time.time()
    lr_init=0.001
    for epoch in range(1,epochs+1):
        lr=max(lr_init*(0.999**epoch),0.00001)
        indices=np.random.permutation(len(X_noisy))
        epoch_loss=0;n_batches=0
        for i in range(0,len(X_noisy),batch_size):
            idx=indices[i:i+batch_size]
            loss=model.train_step(X_noisy[idx],X_clean[idx],lr)
            epoch_loss+=loss;n_batches+=1
        avg_loss=epoch_loss/n_batches
        history.append(float(avg_loss))
        if epoch%50==0 or epoch==1:
            elapsed=time.time()-start;eta=(elapsed/epoch)*(epochs-epoch)
            usage=model.get_expert_usage()
            top_experts=sorted(usage.items(),key=lambda x:x[1],reverse=True)[:3]
            top_str=", ".join([f"{k}:{v:.1%}" for k,v in top_experts])
            print(f"  Epoch {epoch:5d}/{epochs} | Loss: {avg_loss:.6f} | Top: {top_str} | ETA: {eta/60:.1f}min")
        if epoch%500==0:
            model.save(os.path.join(MODEL_DIR,f"checkpoint_{epoch}.npz"))
        if avg_loss<best_loss:
            best_loss=avg_loss
            model.save(os.path.join(MODEL_DIR,"best_moe_model.npz"))
    model.save(os.path.join(MODEL_DIR,"final_moe_model.npz"))
    np.savez(os.path.join(MODEL_DIR,"normalization.npz"),mean=X_mean,std=X_std)
    total_time=time.time()-start
    metadata={"type":"MoE_MAX_macOS","epochs":epochs,"num_experts":num_experts,"hidden_size":hidden_size,"blocks_per_expert":blocks_per_expert,"features":n_features,"batch_size":batch_size,"expert_names":[e.name for e in model.experts],"top_k":model.top_k,"load_balancing":True,"final_loss":float(history[-1]),"best_loss":float(best_loss),"total_params":int(model.count_params()),"training_time_min":float(total_time/60),"expert_usage":model.get_expert_usage(),"ram_gb":14}
    with open(os.path.join(MODEL_DIR,"training_log.json"),'w') as f:json.dump(metadata,f,indent=2)
    print("\n✅ TREINO CONCLUÍDO!")
    print(f"   Loss: {history[0]:.6f} → {history[-1]:.6f}")
    print(f"   Tempo: {total_time/60:.1f} min")
    print(f"   Parâmetros: {model.count_params():,}")

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=5000)
    parser.add_argument("--num-experts",type=int,default=16)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--features",type=int,default=256)
    parser.add_argument("--hidden",type=int,default=512)
    parser.add_argument("--blocks",type=int,default=8)
    args=parser.parse_args()
    train(epochs=max(args.epochs,5000),num_experts=max(args.num_experts,16),batch_size=args.batch_size,n_features=args.features,hidden_size=args.hidden,blocks_per_expert=args.blocks)
