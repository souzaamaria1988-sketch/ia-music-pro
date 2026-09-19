#!/usr/bin/env python3
"""
🎵 IA MUSIC GENERATOR PRO - VERSÃO COMPLETA
Novas funções:
- FM Synthesis
- Granular Synthesis
- Physical Modeling Drums
- Chorus/Flanger/Sidechain
- Modulação de tom
- Polirritmia
- 30+ progressões
"""
import os,sys,json,time
import numpy as np
from pathlib import Path

try:
    from scipy.signal import lfilter
    HAS_SCIPY=True
except ImportError:
    HAS_SCIPY=False

OUTPUT_DIR="song_output"

def get_dynamic_seed():
    return int(time.time()*1000)%(2**32)^np.random.randint(0,2**31)

def get_next_song_number():
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    existing=[f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
    if not existing:return 1
    numbers=[]
    for f in existing:
        try:numbers.append(int(f.replace('.wav','')))
        except:pass
    return max(numbers,default=0)+1

def save_song(audio,sr,metadata=None):
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    number=get_next_song_number()
    filepath=os.path.join(OUTPUT_DIR,f"{number}.wav")
    try:
        import soundfile as sf;sf.write(filepath,audio,sr)
    except ImportError:
        import wave;audio_int16=(audio*32767).astype(np.int16)
        with wave.open(filepath,'w') as wf:
            wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())
    if metadata:
        meta_path=os.path.join(OUTPUT_DIR,f"{number}.json")
        metadata['song_number']=number
        with open(meta_path,'w',encoding='utf-8') as f:json.dump(metadata,f,indent=2,ensure_ascii=False)
    print(f"💾 Salvo: {filepath}")
    return filepath,number

# ============================================================
# NOVAS FUNÇÕES DE SÍNTESE
# ============================================================

def fm_synthesis(freq,duration,sr=44100,mod_ratio=2.0,mod_index=3.0):
    """FM Synthesis - sons metálicos, sinos, pianos elétricos"""
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    mod_freq=freq*mod_ratio
    carrier=np.sin(2*np.pi*freq*t+mod_index*np.sin(2*np.pi*mod_freq*t))
    envelope=np.exp(-t*3)
    return carrier*envelope/(np.max(np.abs(carrier*envelope))+1e-10)

def granular_synthesis(duration,sr=44100,grain_size=0.05,density=10,base_freq=440):
    """Granular Synthesis - texturas, pads, atmosferas"""
    total_samples=int(duration*sr)
    output=np.zeros(total_samples)
    n_grains=int(duration*density)
    for _ in range(n_grains):
        pos=np.random.randint(0,max(1,total_samples-int(grain_size*sr)))
        grain_len=int(grain_size*sr)
        t=np.arange(grain_len)/sr
        freq=base_freq*np.random.uniform(0.5,2.0)
        grain=np.sin(2*np.pi*freq*t)*np.hanning(grain_len)
        grain+=np.random.randn(grain_len)*0.1*np.hanning(grain_len)
        end=min(pos+grain_len,total_samples)
        output[pos:end]+=grain[:end-pos]*np.random.uniform(0.2,0.5)
    return output/(np.max(np.abs(output))+1e-10)

def physical_membrane(freq,duration,sr=44100,damping=10):
    """Physical modeling de membrana (congas, bongos, toms)"""
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    # Modos de vibração de membrana circular
    modes=[1.0,1.59,2.14,2.30,3.60]
    signal=np.zeros_like(t)
    for i,mode in enumerate(modes):
        mode_freq=freq*mode
        amp=1.0/(i+1)
        signal+=amp*np.sin(2*np.pi*mode_freq*t)*np.exp(-t*damping*(i+1))
    # Ataque percussivo
    attack=np.exp(-t*50)*np.random.randn(len(t))*0.3
    signal+=attack
    return signal/(np.max(np.abs(signal))+1e-10)

def karplus_strong(freq,duration,sr=44100,damping=0.996,brightness=0.5):
    N=max(2,int(sr/freq));n_samples=int(duration*sr)
    x=np.zeros(n_samples);x[:min(N,n_samples)]=np.random.uniform(-1,1,min(N,n_samples))
    if HAS_SCIPY:
        a=np.zeros(N+2);a[0]=1.0;a[N]=-damping*brightness;a[N+1]=-damping*(1.0-brightness)
        y=lfilter([1.0],a,x)
    else:
        y=np.zeros(n_samples);delay=np.random.uniform(-1,1,N)
        for i in range(n_samples):
            y[i]=delay[i%N];delay[i%N]=damping*0.5*(delay[i%N]+delay[(i+1)%N])
    return y/(np.max(np.abs(y))+1e-10)

def piano_note(freq,duration,sr=44100,velocity=1.0):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    signal=np.zeros_like(t)
    for h,amp,dec in zip([1,2,3,4,5,6,7,8],[1,.6,.35,.25,.18,.14,.11,.09],[2.5,2.2,2,1.8,1.6,1.4,1.3,1.2]):
        inharmonic=1.0+0.00008*(h**2)
        signal+=amp*np.sin(2*np.pi*freq*h*inharmonic*t)*np.exp(-t*dec)
    atk=int(0.003*sr)
    if 0<atk<len(signal):signal[:atk]*=np.linspace(0,1,atk)
    signal+=np.random.randn(len(signal))*0.03*np.exp(-t*30)
    return signal*velocity/(np.max(np.abs(signal))+1e-10)

def violin_note(freq,duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    vibrato=0.015*np.sin(2*np.pi*5.5*t)
    phase=2*np.pi*np.cumsum(freq*(1.0+vibrato))/sr
    signal=np.zeros_like(t)
    for h in range(1,12):signal+=np.sin(h*phase)/(h*1.2)
    envelope=np.ones_like(t)
    atk=int(min(0.08,duration*0.2)*sr);rel=int(min(0.05,duration*0.1)*sr)
    if 0<atk<len(t):envelope[:atk]=np.linspace(0,1,atk)**0.5
    if 0<rel<len(t):envelope[-rel:]=np.linspace(1,0,rel)
    bow_noise=np.random.randn(len(t))*0.01*envelope
    return (signal*envelope*0.3+bow_noise)/(np.max(np.abs(signal))+1e-10)

def synth_pad(freq,duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    saw1=2*(t*freq%1)-1;saw2=2*(t*freq*1.003%1)-1;saw3=2*(t*freq*0.997%1)-1
    signal=(saw1+saw2+saw3)/3
    signal=np.convolve(signal,np.ones(15)/15,mode='same')
    envelope=np.ones_like(t)
    atk=min(int(0.1*sr),len(t)//3);rel=min(int(0.2*sr),len(t)//3)
    if atk>0:envelope[:atk]=np.linspace(0,1,atk)
    if rel>0:envelope[-rel:]=np.linspace(1,0,rel)
    return signal*envelope*0.3/(np.max(np.abs(signal))+1e-10)

def flute_note(freq,duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    signal=np.sin(2*np.pi*freq*t)+0.3*np.sin(2*np.pi*freq*2*t)+0.1*np.sin(2*np.pi*freq*3*t)
    breath=np.random.randn(len(t))*0.05
    if len(breath)>10:breath=np.convolve(breath,np.ones(10)/10,mode='same')
    envelope=np.ones_like(t)
    atk=int(0.04*sr);rel=int(0.06*sr)
    if 0<atk<len(t):envelope[:atk]=np.linspace(0,1,atk)
    if 0<rel<len(t):envelope[-rel:]=np.linspace(1,0,rel)
    return (signal*envelope*0.3+breath*envelope)/(np.max(np.abs(signal))+1e-10)

def brass_note(freq,duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    phase=2*np.pi*freq*t;signal=np.zeros_like(t)
    for h in range(1,16):signal+=np.sin(h*phase)/h
    envelope=np.ones_like(t);atk=int(0.02*sr)
    if 0<atk<len(t):
        env=np.linspace(0,1.3,atk);split=int(atk*0.7)
        if split<atk:env[split:]=np.linspace(1.3,1.0,atk-split)
        envelope[:atk]=env
    rel=int(0.05*sr)
    if 0<rel<len(t):envelope[-rel:]=np.linspace(1,0,rel)
    return signal*envelope*0.25*1.5/(np.max(np.abs(signal))+1e-10)

def kick_drum(sr=44100):
    t=np.linspace(0,0.35,int(0.35*sr),endpoint=False)
    freq_curve=160*np.exp(-t*25)+45;phase=2*np.pi*np.cumsum(freq_curve)/sr
    signal=np.sin(phase)*np.exp(-t*12)
    return signal/(np.max(np.abs(signal))+1e-10)

def snare_drum(sr=44100):
    t=np.linspace(0,0.22,int(0.22*sr),endpoint=False)
    tone=np.sin(2*np.pi*195*t)*np.exp(-t*35)+0.5*np.sin(2*np.pi*330*t)*np.exp(-t*40)
    noise=np.random.randn(len(t))*np.exp(-t*22)
    return (0.4*tone+0.6*noise)/(np.max(np.abs(0.4*tone+0.6*noise))+1e-10)

def hihat(sr=44100):
    t=np.linspace(0,0.06,int(0.06*sr),endpoint=False)
    noise=np.random.randn(len(t));filtered=np.diff(noise,prepend=noise[0])
    return filtered*np.exp(-t*45)/(np.max(np.abs(filtered))+1e-10)

# ============================================================
# NOVOS EFEITOS
# ============================================================

def chorus_effect(audio,sr=44100,rate=1.5,depth=0.003,mix=0.3):
    """Chorus: engrossa o som"""
    n=len(audio);t=np.arange(n)/sr
    mod=depth*sr*np.sin(2*np.pi*rate*t)
    chorus=np.zeros_like(audio)
    for i in range(100,n):
        delay=int(abs(mod[i]))+100
        if 0<=i-delay<n:chorus[i]=audio[i-delay]
    return audio*(1-mix)+chorus*mix

def flanger_effect(audio,sr=44100,rate=0.5,depth=0.005,feedback=0.5,mix=0.4):
    """Flanger: efeito de avião"""
    n=len(audio);t=np.arange(n)/sr
    mod=depth*sr*(1+np.sin(2*np.pi*rate*t))/2
    flanged=np.zeros_like(audio)
    for i in range(200,n):
        delay=int(mod[i])+100
        if 0<=i-delay<n:flanged[i]=audio[i-delay]
    return audio*(1-mix)+flanged*mix

def sidechain_compress(audio,kick_pattern,sr=44100,threshold=0.3,ratio=4.0,attack=0.005,release=0.1):
    """Sidechain: ducking no kick (efeito EDM)"""
    compressed=audio.copy()
    envelope=np.ones_like(audio)
    kick_times=np.where(kick_pattern>0.5)[0]
    for kt in kick_times:
        start=int(kt)
        attack_samples=int(attack*sr)
        release_samples=int(release*sr)
        end=min(start+attack_samples+release_samples,len(audio))
        if start<len(audio):
            # Duck
            duck_len=min(attack_samples,len(audio)-start)
            if duck_len>0:
                envelope[start:start+duck_len]*=threshold
            # Release
            rel_start=start+attack_samples
            rel_end=min(rel_start+release_samples,len(audio))
            if rel_start<len(audio):
                envelope[rel_start:rel_end]=np.linspace(threshold,1.0,rel_end-rel_start)
    return audio*envelope

def add_reverb(audio,sr=44100,decay=0.3,mix=0.25):
    delay=int(0.03*sr);reverb=np.zeros_like(audio)
    for d in [1,2,3,4,5,6]:
        pos=delay*d
        if pos<len(audio):reverb[pos:]+=audio[:-pos]*(decay**d)
    return audio*(1-mix)+reverb*mix

def add_distortion(audio,gain=3.0,mix=0.7):
    return audio*(1-mix)+np.tanh(audio*gain)*mix

def soft_compress(audio,threshold=0.6,ratio=3.0):
    compressed=audio.copy()
    mask=np.abs(compressed)>threshold
    compressed[mask]=threshold+(compressed[mask]-threshold)/ratio
    return compressed

def bitcrush(audio,bits=8,rate_div=4):
    if len(audio)==0:return audio
    reduced=audio[::rate_div];upsampled=np.repeat(reduced,rate_div)
    if len(upsampled)<len(audio):upsampled=np.pad(upsampled,(0,len(audio)-len(upsampled)))
    elif len(upsampled)>len(audio):upsampled=upsampled[:len(audio)]
    return np.round(upsampled*(2**bits))/(2**bits)

def stutter(audio,sr=44100,size=0.03,repeats=4):
    chunk_size=max(1,int(size*sr))
    if len(audio)==0:return audio
    output=[]
    for i in range(0,len(audio),chunk_size):
        chunk=audio[i:i+chunk_size]
        for _ in range(repeats):output.append(chunk)
    if not output:return audio
    result=np.concatenate(output)
    return result[:len(audio)] if len(result)>=len(audio) else np.pad(result,(0,len(audio)-len(result)))

def reese_bass(freq,duration,sr=44100,detune=0.03):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    saw1=2*(t*freq*(1-detune)%1)-1;saw2=2*(t*freq*(1+detune)%1)-1
    signal=(saw1+saw2)*0.5;signal=np.convolve(signal,np.ones(8)/8,mode='same')
    envelope=np.ones_like(t);rel=min(int(0.05*sr),len(t))
    if rel>0:envelope[-rel:]=np.linspace(1,0,rel)
    return signal*envelope*0.6

def sub_808(freq,duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    freq_curve=freq*2*np.exp(-t*15)+freq;phase=2*np.pi*np.cumsum(freq_curve)/sr
    return np.tanh(np.sin(phase)*1.5)*np.exp(-t*4)

def amen_break(sr=44100,tempo_factor=1.0):
    beat_dur=0.125/tempo_factor
    pattern=[('K',1),('H',.3),('S',.9),('H',.3),('G',.4),('H',.3),('S',.7),('H',.4),
             ('K',.9),('K',.5),('S',.9),('H',.3),('G',.5),('S',.6),('S',.8),('H',.3),
             ('K',1),('H',.3),('S',.9),('G',.4),('K',.8),('H',.4),('S',.8),('H',.3),
             ('K',1),('G',.5),('S',.9),('H',.4),('S',.7),('S',.6),('K',.8),('S',.9)]
    total_samples=int(beat_dur*len(pattern)*sr);output=np.zeros(total_samples)
    kick_s=kick_drum(sr)[:int(0.12*sr)];snare_s=snare_drum(sr)[:int(0.1*sr)]
    ghost_s=snare_drum(sr)[:int(0.06*sr)]*0.4;hat_s=hihat(sr)[:int(0.04*sr)]
    for i,(hit,vel) in enumerate(pattern):
        pos=int(i*beat_dur*sr)
        if hit=='K':sample=kick_s
        elif hit=='S':sample=snare_s
        elif hit=='G':sample=ghost_s
        else:sample=hat_s
        jitter=np.random.randint(-int(0.002*sr),int(0.002*sr)+1)
        pos=max(0,pos+jitter);vel*=np.random.uniform(0.85,1.1)
        end=min(pos+len(sample),total_samples)
        if pos<total_samples:output[pos:end]+=sample[:end-pos]*vel
    return output/(np.max(np.abs(output))+1e-10)

def note_to_freq(semitone,base_freq=261.63):
    return base_freq*(2**(semitone/12.0))

# ============================================================
# 30+ PROGRESSÕES E 15 ESCALAS
# ============================================================

ALL_PROGRESSIONS=[
    [[0,2,4],[5,0,2],[3,5,0],[4,6,1]],
    [[0,2,4],[3,5,0],[4,6,1],[5,0,2]],
    [[5,0,2],[3,5,0],[0,2,4],[4,6,1]],
    [[0,2,4],[0,2,4],[5,0,2],[4,6,1]],
    [[1,3,5],[4,6,1],[0,2,4],[5,0,2]],
    [[0,2,4,6],[4,6,1,3],[5,0,2,4],[0,2,4,6]],
    [[0,3,5],[5,0,2],[4,6,1],[0,2,4]],
    [[0,2,4],[4,6,1],[0,2,4],[5,0,2]],
    [[0,4,6],[3,5,0],[2,4,6],[5,0,2]],
    [[0,2,4],[5,0,2],[4,6,1],[0,2,4]],
    [[0,3,5],[3,5,0],[5,0,2],[4,6,1]],
    [[0,2,4],[2,4,6],[4,6,1],[5,0,2]],
    [[5,0,2],[4,6,1],[3,5,0],[0,2,4]],
    [[0,2,4],[6,1,3],[5,0,2],[4,6,1]],
    [[0,2,4],[5,0,2],[5,0,2],[0,2,4]],
    [[0,2,4],[4,6,1],[3,5,0],[5,0,2]],
    [[0,2,4,7],[5,0,2,6],[3,5,0,4],[4,6,1,5]],
    [[0,3],[5,0],[3,5],[4,6]],
    [[0,4],[5,2],[3,0],[4,1]],
    [[0,2,4,7,9],[5,0,2,6,9],[3,5,0,4,7],[4,6,1,5,9]],
    [[0,2,4],[3,5,0],[0,2,4],[4,6,1]],  # pedal
    [[0,2,4],[5,0,2],[3,5,0],[0,2,4]],  # volta
    [[0,2,4],[1,3,5],[4,6,1],[0,2,4]],  # cromática
    [[0,2,4],[6,1,3],[2,4,6],[5,0,2]],  # empréstimo modal
    [[0,2,4],[3,5,0],[6,1,3],[4,6,1]],  # napolitano
    [[0,2,4],[4,6,1],[2,4,6],[5,0,2]],  # secundária
    [[0,2,4],[5,0,2],[1,3,5],[4,6,1]],  # variação
    [[0,2,4],[3,5,0],[4,6,1],[0,2,4]],  # simples
    [[0,2,4],[0,2,4],[3,5,0],[4,6,1]],  # repetição
    [[0,2,4],[5,0,2],[0,2,4],[4,6,1]],  # alternância
]

ALL_SCALES={
    'major':[0,2,4,5,7,9,11],'minor':[0,2,3,5,7,8,10],'dorian':[0,2,3,5,7,9,10],
    'phrygian':[0,1,3,5,7,8,10],'lydian':[0,2,4,6,7,9,11],'mixolydian':[0,2,4,5,7,9,10],
    'harmonic_minor':[0,2,3,5,7,8,11],'melodic_minor':[0,2,3,5,7,9,11],
    'pentatonic_major':[0,2,4,7,9],'pentatonic_minor':[0,3,5,7,10],
    'blues':[0,3,5,6,7,10],'whole_tone':[0,2,4,6,8,10],
    'hungarian_minor':[0,2,3,6,7,8,11],'japanese':[0,1,5,7,8],
    'bebop':[0,2,4,5,7,9,10,11],
}
SCALE_NAMES=list(ALL_SCALES.keys())

# ============================================================
# RAG
# ============================================================

class MusicRAG:
    def __init__(self):
        self.entries=[
            {"tags":["epic","batalha","heroico"],"scale":"major","dynamics":"loud"},
            {"tags":["dark","sombrio","terror"],"scale":"harmonic_minor","dynamics":"quiet"},
            {"tags":["jazz","swing","blues"],"scale":"dorian","dynamics":"medium"},
            {"tags":["breakcore","glitch","amen"],"scale":"minor","dynamics":"extreme"},
            {"tags":["ambient","calmo"],"scale":"lydian","dynamics":"very_quiet"},
            {"tags":["rock","metal","pesado"],"scale":"phrygian","dynamics":"loud"},
            {"tags":["classical","clássica"],"scale":"major","dynamics":"varied"},
            {"tags":["triste","melancólico"],"scale":"harmonic_minor","dynamics":"quiet"},
            {"tags":["feliz","alegre","pop"],"scale":"major","dynamics":"medium"},
            {"tags":["medieval","fantasia"],"scale":"dorian","dynamics":"medium"},
            {"tags":["eletrônica","techno"],"scale":"minor","dynamics":"loud"},
            {"tags":["boss","intenso"],"scale":"phrygian","dynamics":"extreme"},
            {"tags":["romântico","amor"],"scale":"melodic_minor","dynamics":"quiet"},
            {"tags":["tensão","suspense"],"scale":"whole_tone","dynamics":"building"},
            {"tags":["vitória","triunfo"],"scale":"lydian","dynamics":"loud"},
            {"tags":["folk","acústico","violão"],"scale":"mixolydian","dynamics":"medium"},
            {"tags":["latin","samba","bossa"],"scale":"major","dynamics":"medium"},
            {"tags":["cinematic","filme"],"scale":"harmonic_minor","dynamics":"building"},
            {"tags":["experimental","avant"],"scale":"whole_tone","dynamics":"varied"},
        ]
    def get_context(self,query):
        if not query:return self._random_context()
        query_lower=query.lower()
        scores=[]
        for entry in self.entries:
            score=sum(1 for tag in entry['tags'] if tag in query_lower)
            if score>0:scores.append((score,entry))
        if scores:
            scores.sort(key=lambda x:x[0],reverse=True)
            best=scores[0][1]
        else:
            best=np.random.choice(self.entries)
        progression=ALL_PROGRESSIONS[np.random.randint(0,len(ALL_PROGRESSIONS))]
        if np.random.random()<0.3:
            scale_name=np.random.choice(SCALE_NAMES)
        else:
            scale_name=best.get('scale','major')
        scale=ALL_SCALES.get(scale_name,ALL_SCALES['major'])
        dynamics=best.get('dynamics','medium')
        bpm_map={'very_quiet':(40,70),'quiet':(60,90),'medium':(90,130),'loud':(120,160),'extreme':(160,230),'building':(100,140),'varied':(70,140)}
        bpm_range=bpm_map.get(dynamics,(90,130))
        bpm=np.random.randint(bpm_range[0],bpm_range[1]+1)
        intensity_map={'very_quiet':0.35,'quiet':0.55,'medium':0.7,'loud':0.85,'extreme':0.95,'building':0.75,'varied':0.65}
        intensity=intensity_map.get(dynamics,0.7)+np.random.uniform(-0.1,0.1)
        intensity=np.clip(intensity,0.2,0.98)
        print(f"  🎲 RAG: escala={scale_name}, BPM={bpm}, prog={ALL_PROGRESSIONS.index(progression)}")
        return {'scale':scale,'scale_name':scale_name,'progression':progression,'bpm':bpm,'intensity':intensity,'dynamics':dynamics,'seed':get_dynamic_seed()}
    def _random_context(self):
        scale_name=np.random.choice(SCALE_NAMES)
        progression=ALL_PROGRESSIONS[np.random.randint(0,len(ALL_PROGRESSIONS))]
        return {'scale':ALL_SCALES[scale_name],'scale_name':scale_name,'progression':progression,'bpm':np.random.randint(50,200),'intensity':np.random.uniform(0.3,0.95),'seed':get_dynamic_seed()}

# ============================================================
# GERAÇÃO COM MODULAÇÃO E POLIRRITMIA
# ============================================================

def generate_breakcore(duration,sr=44100,intensity=1.0):
    seed=get_dynamic_seed()
    np.random.seed(seed)
    total_samples=int(duration*sr);bpm=np.random.randint(180,230)
    print(f"  💥 BREAKCORE (seed={seed}): BPM={bpm}")
    break_source=amen_break(sr,tempo_factor=bpm/180)
    drums_track=np.zeros(total_samples);pos=0
    while pos<total_samples:
        max_start=max(1,len(break_source)-int(0.15*sr))
        start=np.random.randint(0,max_start)
        length=np.random.choice([int(0.03*sr),int(0.06*sr),int(0.12*sr),int(0.25*sr)])
        chunk=break_source[start:start+length].copy()
        if len(chunk)==0:pos+=int(0.05*sr);continue
        effect=np.random.choice(['none','stutter','reverse','crush','pitch_up','pitch_down'])
        if effect=='stutter':chunk=stutter(chunk,sr,0.015,np.random.randint(2,8))
        elif effect=='reverse':chunk=chunk[::-1]
        elif effect=='crush':chunk=bitcrush(chunk,np.random.randint(4,10),np.random.randint(2,8))
        elif effect=='pitch_up':
            idx=np.round(np.arange(0,len(chunk),np.random.uniform(1.3,2.0))).astype(int)
            chunk=chunk[idx[idx<len(chunk)]]
        elif effect=='pitch_down':
            idx=np.round(np.arange(0,len(chunk),np.random.uniform(0.5,0.8))).astype(int)
            chunk=chunk[idx[idx<len(chunk)]]
        if np.random.random()<0.4:chunk=np.tanh(chunk*np.random.uniform(2,6))
        end=min(pos+len(chunk),total_samples)
        if pos<total_samples and len(chunk)>0:
            drums_track[pos:end]+=chunk[:end-pos]*np.random.uniform(0.5,1.0)*intensity*0.8
        pos+=len(chunk)
        if np.random.random()<0.24:pos+=np.random.randint(int(0.01*sr),int(0.08*sr))
    bass_track=np.zeros(total_samples);beat_dur=60.0/bpm
    for i in range(int(duration/beat_dur)):
        pos=int(i*beat_dur*sr)
        if np.random.random()<0.6:
            freq=np.random.choice([55.0,58.27,65.41,73.42,82.41])
            note_dur=beat_dur*np.random.choice([0.5,1.0,1.5])
            bass_note=reese_bass(freq,note_dur,sr) if np.random.random()<0.5 else sub_808(freq,note_dur,sr)
            end=min(pos+len(bass_note),total_samples)
            if pos<total_samples:bass_track[pos:end]+=bass_note[:end-pos]*0.7
    glitch_track=np.zeros(total_samples)
    for _ in range(int(duration*3)):
        if total_samples<=int(0.1*sr):break
        pos=np.random.randint(0,max(1,total_samples-int(0.1*sr)))
        length=np.random.randint(int(0.005*sr),int(0.08*sr))
        t=np.arange(length)/sr
        gt=np.random.choice(['noise','tone','sweep'])
        if gt=='noise':glitch=np.random.randn(length)*np.exp(-np.arange(length)/(length/3))
        elif gt=='tone':glitch=np.sin(2*np.pi*np.random.uniform(500,6000)*t)*np.exp(-t*25)
        else:
            freqs=np.linspace(np.random.uniform(200,3000),np.random.uniform(200,3000),length)
            glitch=np.sin(2*np.pi*np.cumsum(freqs)/sr)
        glitch=bitcrush(glitch,np.random.randint(3,8),np.random.randint(2,10))
        end=min(pos+len(glitch),total_samples)
        if len(glitch)>0:glitch_track[pos:end]+=glitch[:end-pos]*0.15
    mix=drums_track*0.85+bass_track*0.75+glitch_track*0.6
    mix=np.tanh(mix*2.0);mix=soft_compress(mix,0.35,5.0)
    mix=mix/(np.max(np.abs(mix))+1e-10)*0.95
    fi,fo=int(0.1*sr),int(0.3*sr)
    if fi<len(mix):mix[:fi]*=np.linspace(0,1,fi)
    if fo<len(mix):mix[-fo:]*=np.linspace(1,0,fo)
    return mix,sr

def generate_with_rag(duration,rag_context,sr=44100):
    np.random.seed(rag_context['seed'])
    total_samples=int(duration*sr)
    bpm=rag_context['bpm']
    beat_duration=60.0/bpm
    scale=rag_context['scale']
    progression=rag_context['progression']
    intensity=rag_context['intensity']
    base_freq_options=[220.0,246.94,261.63,293.66,329.63,349.23]
    base_freq=np.random.choice(base_freq_options)
    
    # MODULAÇÃO: chance de mudar de tom no meio
    use_modulation=np.random.random()<0.3
    modulation_point=len(progression)//2 if use_modulation else len(progression)+1
    modulation_shift=np.random.choice([2,3,5,7]) if use_modulation else 0
    
    print(f"🎵 Gerando (seed={rag_context['seed']}): BPM={bpm}, tom={base_freq:.1f}Hz")
    if use_modulation:print(f"   🔄 Modulação: +{modulation_shift} semitons na metade")
    
    drums=np.zeros(total_samples)
    bass_track=np.zeros(total_samples)
    chords_track=np.zeros(total_samples)
    melody_track=np.zeros(total_samples)
    
    # POLIRRITMIA: chance de ter camadas rítmicas diferentes
    use_polyrhythm=np.random.random()<0.25
    poly_ratio=np.random.choice([3,4,5]) if use_polyrhythm else 0
    
    print("  🥁 Bateria...")
    kick,snare,hh=kick_drum(sr),snare_drum(sr),hihat(sr)
    n_beats=int(duration/beat_duration)
    
    # Kick pattern para sidechain
    kick_pattern=np.zeros(n_beats)
    
    drum_patterns=['standard','four_floor','half_time','syncopated','no_kick']
    drum_pattern=np.random.choice(drum_patterns)
    
    for beat in range(n_beats):
        pos=int(beat*beat_duration*sr)
        if drum_pattern=='standard':
            if beat%4 in [0,2] and pos+len(kick)<=total_samples:
                drums[pos:pos+len(kick)]+=kick*0.8
                kick_pattern[beat]=1
            if beat%4 in [1,3] and pos+len(snare)<=total_samples:drums[pos:pos+len(snare)]+=snare*0.7
        elif drum_pattern=='four_floor':
            if pos+len(kick)<=total_samples:
                drums[pos:pos+len(kick)]+=kick*0.7
                kick_pattern[beat]=1
            if beat%4 in [1,3] and pos+len(snare)<=total_samples:drums[pos:pos+len(snare)]+=snare*0.6
        elif drum_pattern=='half_time':
            if beat%4==0 and pos+len(kick)<=total_samples:
                drums[pos:pos+len(kick)]+=kick*0.9
                kick_pattern[beat]=1
            if beat%4==2 and pos+len(snare)<=total_samples:drums[pos:pos+len(snare)]+=snare*0.8
        elif drum_pattern=='syncopated':
            if beat%4 in [0,3] and pos+len(kick)<=total_samples:
                drums[pos:pos+len(kick)]+=kick*0.7
                kick_pattern[beat]=1
            if beat%4 in [1,2] and pos+len(snare)<=total_samples:drums[pos:pos+len(snare)]+=snare*0.6
        else:
            if beat%2==0 and pos+len(snare)<=total_samples:drums[pos:pos+len(snare)]+=snare*0.7
        
        hat_subdivision=np.random.choice([2,4])
        for sub in range(hat_subdivision):
            sp=int((beat+sub/hat_subdivision)*beat_duration*sr)
            if sp+len(hh)<=total_samples:
                vel=0.4-(0.15*sub/hat_subdivision)
                drums[sp:sp+len(hh)]+=hh*vel
    
    # POLIRRITMIA: camada extra
    if use_polyrhythm:
        print(f"  🌀 Polirritmia: {poly_ratio} contra 4")
        poly_beat_dur=beat_duration*4/poly_ratio
        for i in range(int(duration/poly_beat_dur)):
            pos=int(i*poly_beat_dur*sr)
            if pos+len(hh)<=total_samples:
                drums[pos:pos+len(hh)]+=hh*0.2
    
    # Baixo
    print("  🎸 Baixo...")
    bass_patterns=['root_only','walking','arpeggio','octaves']
    bass_pattern=np.random.choice(bass_patterns)
    
    for beat in range(n_beats):
        chord_idx=(beat//4)%len(progression)
        root=progression[chord_idx][0]
        
        # Aplicar modulação
        if chord_idx>=modulation_point:
            root=(root+modulation_shift)%len(scale)
        
        root_freq=note_to_freq(scale[root%len(scale)],base_freq)/2
        
        if bass_pattern=='root_only' and beat%2==0:
            note=karplus_strong(root_freq,beat_duration*1.5,sr,damping=0.998,brightness=0.3)
            pos=int(beat*beat_duration*sr)
            if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*0.4
        elif bass_pattern=='walking':
            walk_steps=[0,2,4,2]
            step=walk_steps[beat%4]
            freq=note_to_freq(scale[(root+step)%len(scale)],base_freq)/2
            note=karplus_strong(freq,beat_duration*0.9,sr,damping=0.997)
            pos=int(beat*beat_duration*sr)
            if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*0.35
        elif bass_pattern=='arpeggio':
            chord=progression[chord_idx]
            arp_note=chord[beat%len(chord)]
            if chord_idx>=modulation_point:
                arp_note=(arp_note+modulation_shift)%len(scale)
            freq=note_to_freq(scale[arp_note%len(scale)],base_freq)/2
            note=karplus_strong(freq,beat_duration*0.8,sr)
            pos=int(beat*beat_duration*sr)
            if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*0.3
        elif bass_pattern=='octaves':
            octave_choice=1 if beat%2==0 else 2
            freq=root_freq*octave_choice
            note=karplus_strong(freq,beat_duration*0.9,sr)
            pos=int(beat*beat_duration*sr)
            if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*0.35
    
    # Acordes
    print("  🎹 Acordes...")
    chord_duration=beat_duration*4
    chord_instruments=['piano','strings','synth','guitar','fm','mixed']
    chord_inst=np.random.choice(chord_instruments)
    
    for i in range(int(duration/chord_duration)):
        chord=progression[i%len(progression)]
        pos=int(i*chord_duration*sr)
        
        # Aplicar modulação
        if i%len(progression)>=modulation_point:
            chord=[(c+modulation_shift)%len(scale) for c in chord]
        
        voicing_style=np.random.choice(['close','spread','drop2','root_pos'])
        
        for j,nd in enumerate(chord):
            if voicing_style=='close':octave=0
            elif voicing_style=='spread':octave=j
            elif voicing_style=='drop2':octave=0 if j!=1 else -1
            else:octave=0
            
            freq=note_to_freq(scale[nd%len(scale)],base_freq)*(2**octave)
            
            if chord_inst=='piano' or chord_inst=='mixed':
                note=piano_note(freq,chord_duration*0.9,sr)
                if pos+len(note)<=total_samples:chords_track[pos:pos+len(note)]+=note*0.15
            if chord_inst=='strings' or chord_inst=='mixed':
                note=violin_note(freq,chord_duration*0.95,sr)
                if pos+len(note)<=total_samples:chords_track[pos:pos+len(note)]+=note*0.12
            if chord_inst=='synth':
                note=synth_pad(freq,chord_duration*0.95,sr)
                if pos+len(note)<=total_samples:chords_track[pos:pos+len(note)]+=note*0.18
            if chord_inst=='guitar':
                note=karplus_strong(freq,chord_duration*0.8,sr,damping=0.995)
                if pos+len(note)<=total_samples:chords_track[pos:pos+len(note)]+=note*0.2
            if chord_inst=='fm':
                note=fm_synthesis(freq,chord_duration*0.8,sr,mod_ratio=np.random.uniform(1.5,3.0))
                if pos+len(note)<=total_samples:chords_track[pos:pos+len(note)]+=note*0.15
    
    # Melodia
    print("  🎶 Melodia...")
    note_duration=beat_duration/2
    melody_styles=['stepwise','arpeggios','leaps','repeated','call_response']
    melody_style=np.random.choice(melody_styles)
    
    melody_notes=[]
    current_degree=0
    call_phrase=[]
    
    for i in range(int(duration/note_duration)):
        if melody_style=='stepwise':
            step=np.random.choice([-2,-1,0,1,2],p=[0.15,0.25,0.2,0.25,0.15])
        elif melody_style=='arpeggios':
            chord_idx=(i//8)%len(progression)
            chord=progression[chord_idx]
            if chord_idx>=modulation_point:
                chord=[(c+modulation_shift)%len(scale) for c in chord]
            current_degree=chord[i%len(chord)]
            step=0
        elif melody_style=='leaps':
            step=np.random.choice([-4,-3,3,4,5],p=[0.2,0.2,0.2,0.2,0.2])
        elif melody_style=='repeated':
            step=0 if np.random.random()<0.6 else np.random.choice([-1,1])
        else:
            if i%16<8:
                step=np.random.choice([1,2,3])
                call_phrase.append(current_degree)
            else:
                if len(call_phrase)>0:
                    current_degree=call_phrase.pop()
                    step=np.random.choice([-3,-2,-1])
                else:
                    step=np.random.choice([-1,0,1])
        
        current_degree=max(0,min(current_degree+step,len(scale)*2-1))
        octave=current_degree//len(scale)
        degree=current_degree%len(scale)
        
        # Aplicar modulação na melodia também
        if i//(8*len(progression))%2==1 and use_modulation:
            degree=(degree+modulation_shift)%len(scale)
        
        freq=note_to_freq(scale[degree],base_freq)*(2**octave)
        
        play_prob=0.85 if melody_style!='repeated' else 0.95
        if np.random.random()<play_prob:
            dur_mult=np.random.choice([0.5,1.0,1.5,2.0],p=[0.2,0.4,0.3,0.1])
            melody_notes.append((freq,i*note_duration,dur_mult))
    
    melody_instruments=['flute','piano','violin','synth','guitar','brass','fm']
    melody_inst=np.random.choice(melody_instruments)
    
    for freq,start_time,dur_mult in melody_notes:
        pos=int(start_time*sr)
        nlen=int(note_duration*sr*1.8*dur_mult)
        if melody_inst=='flute':note=flute_note(freq,nlen/sr,sr)
        elif melody_inst=='piano':note=piano_note(freq,nlen/sr,sr)
        elif melody_inst=='violin':note=violin_note(freq,nlen/sr,sr)
        elif melody_inst=='synth':note=synth_pad(freq,nlen/sr,sr)
        elif melody_inst=='guitar':note=karplus_strong(freq,nlen/sr,sr)
        elif melody_inst=='brass':note=brass_note(freq,nlen/sr,sr)
        elif melody_inst=='fm':note=fm_synthesis(freq,nlen/sr,sr,mod_ratio=np.random.uniform(1.5,4.0))
        else:note=flute_note(freq,nlen/sr,sr)
        if pos+len(note)<=total_samples:melody_track[pos:pos+len(note)]+=note*0.3*intensity
    
    # Chance de adicionar textura granular
    if np.random.random()<0.2:
        print("  ✨ Textura granular...")
        granular=granular_synthesis(duration,sr,grain_size=np.random.uniform(0.02,0.1),density=np.random.uniform(5,20),base_freq=base_freq)
        melody_track+=granular*0.1
    
    # Mixagem
    print("  🎛️ Mixagem...")
    mix=drums+bass_track+chords_track+melody_track
    
    # SIDECHAIN se tiver kick pattern e estilo eletrônico
    if np.sum(kick_pattern)>0 and np.random.random()<0.3:
        print("  🔊 Sidechain compression...")
        mix=sidechain_compress(mix,kick_pattern,sr,threshold=np.random.uniform(0.3,0.5))
    
    # Efeitos variados
    reverb_amount=np.random.uniform(0.15,0.35)
    mix=add_reverb(mix,sr,decay=np.random.uniform(0.25,0.45),mix=reverb_amount)
    
    # Chance de chorus
    if np.random.random()<0.3:
        mix=chorus_effect(mix,sr,mix=0.2)
    
    # Chance de flanger
    if np.random.random()<0.15:
        mix=flanger_effect(mix,sr,mix=0.2)
    
    threshold=np.random.uniform(0.4,0.7)
    mix=soft_compress(mix,threshold=threshold,ratio=np.random.uniform(2.5,4.0))
    
    mix=mix/(np.max(np.abs(mix))+1e-10)*0.9
    
    fade_in=int(np.random.uniform(0.1,1.0)*sr)
    fade_out=int(np.random.uniform(0.5,2.0)*sr)
    if fade_in<len(mix):mix[:fade_in]*=np.linspace(0,1,fade_in)
    if fade_out<len(mix):mix[-fade_out:]*=np.linspace(1,0,fade_out)
    
    return mix,sr

# ============================================================
# MAIN
# ============================================================

def generate_music(prompt=None,style=None,duration=45,use_rag=True):
    rag=MusicRAG() if use_rag else None
    query=prompt if prompt else style
    
    if query and any(kw in query.lower() for kw in ['breakcore','amen','glitch','jungle']):
        print("💥 Modo BREAKCORE")
        return generate_breakcore(duration)
    
    if use_rag and rag:
        print("📚 Usando RAG...")
        context=rag.get_context(query)
        return generate_with_rag(duration,context)
    else:
        context=rag._random_context() if rag else {
            'scale':[0,2,4,5,7,9,11],'scale_name':'major',
            'progression':ALL_PROGRESSIONS[0],'bpm':120,
            'intensity':0.7,'seed':get_dynamic_seed()
        }
        return generate_with_rag(duration,context)

def batch_generate():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--prompt",type=str,default="")
    parser.add_argument("--style",type=str,default="epic")
    parser.add_argument("--duration",type=int,default=45)
    parser.add_argument("--use-rag",type=str,default="true")
    parser.add_argument("--batch",action="store_true")
    args=parser.parse_args()
    
    use_rag=args.use_rag.lower()=="true"
    print("="*60)
    print("🎵 IA MUSIC GENERATOR PRO - MoE+RAG+Novas Funções")
    print("="*60)
    
    prompt=args.prompt if args.prompt else None
    style=args.style if not prompt else None
    audio,sr=generate_music(prompt=prompt,style=style,duration=args.duration,use_rag=use_rag)
    metadata={"prompt":args.prompt,"style":args.style,"duration":args.duration,"use_rag":use_rag}
    filepath,number=save_song(audio,sr,metadata)
    print(f"\n✅ Música #{number}: {filepath}")

def main():
    while True:
        print("="*50);print("🎵 IA MUSIC PRO");print("="*50)
        print("1. Prompt  2. Estilo  3. Breakcore  4. Ver  5. Sair")
        choice=input("> ").strip()
        if choice=='1':
            prompt=input("Prompt: ").strip()
            duration=input("Duração (30/45/60/90): ").strip()
            if duration not in ["30","45","60","90"]:duration="45"
            audio,sr=generate_music(prompt=prompt,duration=int(duration))
            fp,num=save_song(audio,sr,{"prompt":prompt,"duration":int(duration)})
            print(f"✅ #{num}: {fp}")
        elif choice=='2':
            print("1.Epico 2.Boss 3.Dark 4.Rock 5.Ambient 6.Eletronico 7.Jazz 8.Classico")
            s=input("Estilo: ").strip()
            styles={"1":"epic","2":"bossfight","3":"dark","4":"rock","5":"ambient","6":"electronic","7":"jazz","8":"classical"}
            duration=input("Duração: ").strip()
            if duration not in ["30","45","60","90"]:duration="45"
            audio,sr=generate_music(style=styles.get(s,"epic"),duration=int(duration))
            fp,num=save_song(audio,sr,{"style":styles.get(s,"epic")})
            print(f"✅ #{num}: {fp}")
        elif choice=='3':
            duration=input("Duração: ").strip()
            if duration not in ["30","45","60","90"]:duration="45"
            audio,sr=generate_breakcore(int(duration))
            fp,num=save_song(audio,sr,{"style":"breakcore"})
            print(f"✅ #{num}: {fp}")
        elif choice=='4':
            songs=sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')])
            for s in songs:print(f"  {s}")
        elif choice=='5':break
        input("ENTER...")

if __name__=="__main__":
    if "--batch" in sys.argv:batch_generate()
    else:main()
