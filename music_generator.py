#!/usr/bin/env python3
"""
🎵 IA MUSIC GENERATOR - COM FEEDBACK LOOP + AUTO-MIXING
Usa music_intelligence.py para aprender e melhorar
"""
import os,sys,json,time,gc
import numpy as np
from pathlib import Path

try:
    from scipy.signal import lfilter
    HAS_SCIPY=True
except ImportError:
    HAS_SCIPY=False

# Importar inteligência
from music_intelligence import FeedbackLoop, AutoMixer, MusicMemory

OUTPUT_DIR="song_output"
MODEL_DIR="models"
MUSIC_DIR="music_input"
SAMPLE_CACHE={}

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
# INSTRUMENTOS (COM NORMALIZAÇÃO AUTOMÁTICA)
# ============================================================

def make_kick(sr=44100,velocity=1.0):
    key=('kick',sr,velocity)
    if key in SAMPLE_CACHE:return SAMPLE_CACHE[key].copy()
    t=np.linspace(0,0.35,int(0.35*sr),endpoint=False)
    freq_curve=160*np.exp(-t*25)+45;phase=2*np.pi*np.cumsum(freq_curve)/sr
    signal=np.sin(phase)*np.exp(-t*12)*velocity
    result=signal/(np.max(np.abs(signal))+1e-10)
    SAMPLE_CACHE[key]=result
    return result.copy()

def make_snare(sr=44100,velocity=1.0):
    key=('snare',sr,velocity)
    if key in SAMPLE_CACHE:return SAMPLE_CACHE[key].copy()
    t=np.linspace(0,0.22,int(0.22*sr),endpoint=False)
    tone=np.sin(2*np.pi*195*t)*np.exp(-t*35)+0.5*np.sin(2*np.pi*330*t)*np.exp(-t*40)
    noise=np.random.randn(len(t))*np.exp(-t*22)
    signal=(0.4*tone+0.6*noise)*velocity
    result=signal/(np.max(np.abs(signal))+1e-10)
    SAMPLE_CACHE[key]=result
    return result.copy()

def make_hihat(sr=44100,velocity=1.0):
    key=('hihat',sr,velocity)
    if key in SAMPLE_CACHE:return SAMPLE_CACHE[key].copy()
    t=np.linspace(0,0.06,int(0.06*sr),endpoint=False)
    noise=np.random.randn(len(t));filtered=np.diff(noise,prepend=noise[0])
    result=filtered*np.exp(-t*45)*velocity/(np.max(np.abs(filtered))+1e-10)
    SAMPLE_CACHE[key]=result
    return result.copy()

def make_tom(freq=120,sr=44100,velocity=1.0):
    key=('tom',freq,sr,velocity)
    if key in SAMPLE_CACHE:return SAMPLE_CACHE[key].copy()
    t=np.linspace(0,0.3,int(0.3*sr),endpoint=False)
    freq_curve=freq*np.exp(-t*8)+freq*0.7;phase=2*np.pi*np.cumsum(freq_curve)/sr
    signal=np.sin(phase)*np.exp(-t*10)*velocity
    result=signal/(np.max(np.abs(signal))+1e-10)
    SAMPLE_CACHE[key]=result
    return result.copy()

def make_crash(sr=44100,velocity=1.0):
    key=('crash',sr,velocity)
    if key in SAMPLE_CACHE:return SAMPLE_CACHE[key].copy()
    t=np.linspace(0,1.5,int(1.5*sr),endpoint=False)
    noise=np.random.randn(len(t))
    metallic=0
    for f in [5000,6500,8000,9500,11000,13000]:
        metallic+=0.15*np.sin(2*np.pi*f*t+np.random.uniform(0,2*np.pi))
    signal=(noise*0.4+metallic*0.4)*np.exp(-t*3)*velocity
    result=signal/(np.max(np.abs(signal))+1e-10)
    SAMPLE_CACHE[key]=result
    return result.copy()

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
    return (signal*envelope*0.3)/(np.max(np.abs(signal))+1e-10)

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

# ============================================================
# ESTRUTURA MUSICAL
# ============================================================

ALL_PROGRESSIONS=[
    [[0,2,4],[5,0,2],[3,5,0],[4,6,1]],[[0,2,4],[3,5,0],[4,6,1],[5,0,2]],
    [[5,0,2],[3,5,0],[0,2,4],[4,6,1]],[[0,2,4],[0,2,4],[5,0,2],[4,6,1]],
    [[1,3,5],[4,6,1],[0,2,4],[5,0,2]],[[0,2,4,6],[4,6,1,3],[5,0,2,4],[0,2,4,6]],
]

ALL_SCALES={
    'major':[0,2,4,5,7,9,11],'minor':[0,2,3,5,7,8,10],
    'dorian':[0,2,3,5,7,9,10],'phrygian':[0,1,3,5,7,8,10],
    'lydian':[0,2,4,6,7,9,11],'mixolydian':[0,2,4,5,7,9,10],
    'harmonic_minor':[0,2,3,5,7,8,11],'pentatonic_major':[0,2,4,7,9],
}
SCALE_NAMES=list(ALL_SCALES.keys())

class SongStructure:
    STRUCTURES={
        'pop':['intro','verse','chorus','verse','chorus','bridge','chorus','outro'],
        'rock':['intro','verse','chorus','verse','chorus','solo','chorus','outro'],
        'electronic':['intro','buildup','drop','breakdown','buildup','drop','outro'],
        'cinematic':['intro','theme','development','climax','resolution','outro'],
        'jazz':['intro','head','solo1','head','solo2','head','outro'],
        'ambient':['intro','section1','section2','section3','section4','outro'],
        'classical':['exposition','development','recapitulation','coda'],
    }
    SECTION_ENERGY={
        'intro':0.3,'verse':0.5,'chorus':0.9,'bridge':0.6,'outro':0.4,
        'buildup':0.7,'drop':1.0,'breakdown':0.2,'solo':0.7,'head':0.6,
        'theme':0.5,'development':0.7,'climax':1.0,'resolution':0.5,
        'exposition':0.5,'recapitulation':0.7,'coda':0.4,
        'section1':0.4,'section2':0.5,'section3':0.6,'section4':0.5,
    }
    def __init__(self,style='pop',duration=45,bpm=120):
        self.style=style;self.duration=duration;self.bpm=bpm
        self.sections=self._generate_sections()
        self.section_times=self._calculate_times()
    def _generate_sections(self):
        base_structure=self.STRUCTURES.get(self.style,self.STRUCTURES['pop'])
        sections=[]
        for section in base_structure:
            if section=='intro':bars=np.random.choice([4,8])
            elif section in ['verse','head','theme','exposition']:bars=np.random.choice([8,12,16])
            elif section in ['chorus','climax','drop','recapitulation']:bars=np.random.choice([8,12])
            elif section=='bridge':bars=np.random.choice([4,8])
            elif section=='buildup':bars=np.random.choice([4,8])
            elif section in ['breakdown','coda']:bars=np.random.choice([4,8])
            elif section in ['solo','development']:bars=np.random.choice([8,12,16])
            elif section=='outro':bars=np.random.choice([4,8])
            else:bars=8
            sections.append({'name':section,'bars':bars,'energy':self.SECTION_ENERGY.get(section,0.5)})
        return sections
    def _calculate_times(self):
        beats_per_bar=4;beat_duration=60.0/self.bpm;times=[];current_time=0.0
        for section in self.sections:
            section_duration=section['bars']*beats_per_bar*beat_duration
            times.append({'name':section['name'],'start':current_time,'end':current_time+section_duration,'duration':section_duration,'energy':section['energy']})
            current_time+=section_duration
        total_time=current_time
        if total_time>0:
            scale=self.duration/total_time
            for t in times:t['start']*=scale;t['end']*=scale;t['duration']*=scale
        return times
    def get_section_at_time(self,time):
        for section in self.section_times:
            if section['start']<=time<section['end']:return section
        return self.section_times[-1]
    def get_energy_at_time(self,time):
        section=self.get_section_at_time(time)
        section_duration=section['duration']
        position=(time-section['start'])/section_duration if section_duration>0 else 0
        base_energy=section['energy']
        if section['name']=='buildup':return base_energy*(0.3+0.7*position)
        elif section['name']=='drop':return 1.0 if position<0.1 else base_energy*(1.0-0.2*(position-0.1)/0.9)
        elif section['name']=='outro':return base_energy*(1.0-position*0.7)
        else:return base_energy+0.1*np.sin(position*np.pi*2)

def note_to_freq(semitone,base_freq=261.63):
    return base_freq*(2**(semitone/12.0))

# ============================================================
# GERAÇÃO COM AUTO-MIXING
# ============================================================

def generate_with_automix(duration,bpm=120,style='pop',sr=44100):
    """Gera música com auto-mixing usando tracks separadas"""
    seed = get_dynamic_seed()
    np.random.seed(seed)
    
    base_freq=np.random.choice([220.0,246.94,261.63,293.66,329.63,349.23])
    scale=ALL_SCALES[np.random.choice(SCALE_NAMES)]
    progression=ALL_PROGRESSIONS[np.random.randint(0,len(ALL_PROGRESSIONS))]
    
    structure=SongStructure(style=style,duration=duration,bpm=bpm)
    
    print(f"   🎼 Estrutura: {[s['name'] for s in structure.sections]}")
    
    total_samples=int(duration*sr)
    beat_duration=60.0/bpm
    
    # TRACKS SEPARADAS (para mixagem automática)
    drums_track=np.zeros(total_samples)
    bass_track=np.zeros(total_samples)
    chords_track=np.zeros(total_samples)
    melody_track=np.zeros(total_samples)
    fx_track=np.zeros(total_samples)
    
    kick_sample=make_kick(sr);snare_sample=make_snare(sr);hihat_sample=make_hihat(sr)
    tom1_sample=make_tom(200,sr);tom2_sample=make_tom(150,sr);tom3_sample=make_tom(100,sr)
    crash_sample=make_crash(sr)
    n_beats=int(duration/beat_duration)
    
    # BATERIA
    for beat in range(n_beats):
        time=beat*beat_duration;section=structure.get_section_at_time(time)
        section_name=section['name'];energy=structure.get_energy_at_time(time)
        vel=0.4+0.6*energy;pos=int(beat*beat_duration*sr)
        if section_name in ['intro','breakdown']:
            if beat%4==0 and pos+len(kick_sample)<=total_samples:drums_track[pos:pos+len(kick_sample)]+=kick_sample*vel*0.6
            if beat%4==2 and pos+len(snare_sample)<=total_samples:drums_track[pos:pos+len(snare_sample)]+=snare_sample*vel*0.5
            if pos+len(hihat_sample)<=total_samples:drums_track[pos:pos+len(hihat_sample)]+=hihat_sample*vel*0.3
        elif section_name in ['verse','head','theme']:
            if beat%4 in [0,2] and pos+len(kick_sample)<=total_samples:drums_track[pos:pos+len(kick_sample)]+=kick_sample*vel*0.7
            if beat%4 in [1,3] and pos+len(snare_sample)<=total_samples:drums_track[pos:pos+len(snare_sample)]+=snare_sample*vel*0.6
            if pos+len(hihat_sample)<=total_samples:drums_track[pos:pos+len(hihat_sample)]+=hihat_sample*vel*0.35
        elif section_name in ['chorus','drop','climax']:
            if beat%2==0 and pos+len(kick_sample)<=total_samples:drums_track[pos:pos+len(kick_sample)]+=kick_sample*vel*0.8
            if beat%4 in [1,3] and pos+len(snare_sample)<=total_samples:drums_track[pos:pos+len(snare_sample)]+=snare_sample*vel*0.7
            for sub in [0,0.5]:
                sp=int((beat+sub)*beat_duration*sr)
                if sp+len(hihat_sample)<=total_samples:drums_track[sp:sp+len(hihat_sample)]+=hihat_sample*vel*(0.35 if sub==0 else 0.2)
        else:
            if beat%4 in [0,2] and pos+len(kick_sample)<=total_samples:drums_track[pos:pos+len(kick_sample)]+=kick_sample*vel*0.6
            if beat%4 in [1,3] and pos+len(snare_sample)<=total_samples:drums_track[pos:pos+len(snare_sample)]+=snare_sample*vel*0.5
            if pos+len(hihat_sample)<=total_samples:drums_track[pos:pos+len(hihat_sample)]+=hihat_sample*vel*0.3
    
    # BAIXO
    for beat in range(n_beats):
        time=beat*beat_duration;section=structure.get_section_at_time(time);energy=structure.get_energy_at_time(time)
        chord_idx=(beat//4)%len(progression);root=progression[chord_idx][0]
        root_freq=note_to_freq(scale[root%len(scale)],base_freq)/2;vel=0.3+0.7*energy
        if section['name'] in ['intro','breakdown']:
            if beat%4==0:
                note=karplus_strong(root_freq,beat_duration*3,sr,damping=0.998)
                pos=int(beat*beat_duration*sr)
                if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*vel*0.25
        elif section['name'] in ['verse','head','theme']:
            if beat%2==0:
                note=karplus_strong(root_freq,beat_duration*1.5,sr,damping=0.997)
                pos=int(beat*beat_duration*sr)
                if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*vel*0.35
        elif section['name'] in ['chorus','drop','climax']:
            freq=root_freq if beat%2==0 else root_freq*2
            note=karplus_strong(freq,beat_duration*0.9,sr)
            pos=int(beat*beat_duration*sr)
            if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*vel*0.4
        else:
            if beat%2==0:
                note=karplus_strong(root_freq,beat_duration*1.2,sr)
                pos=int(beat*beat_duration*sr)
                if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*vel*0.3
    
    # ACORDES
    chord_duration=beat_duration*4
    for i in range(int(duration/chord_duration)):
        time=i*chord_duration;section=structure.get_section_at_time(time);energy=structure.get_energy_at_time(time)
        chord=progression[i%len(progression)];pos=int(i*chord_duration*sr);vel=0.2+0.6*energy
        if section['name'] in ['intro','breakdown','outro']:
            for nd in chord:
                freq=note_to_freq(scale[nd%len(scale)],base_freq)
                note=synth_pad(freq,chord_duration*0.95,sr)
                if pos+len(note)<=total_samples:chords_track[pos:pos+len(note)]+=note*vel*0.15
        elif section['name'] in ['verse','head','theme']:
            for nd in chord:
                freq=note_to_freq(scale[nd%len(scale)],base_freq)
                note=piano_note(freq,chord_duration*0.9,sr)
                if pos+len(note)<=total_samples:chords_track[pos:pos+len(note)]+=note*vel*0.18
        else:
            for nd in chord:
                freq=note_to_freq(scale[nd%len(scale)],base_freq)
                note_p=piano_note(freq,chord_duration*0.9,sr)
                if pos+len(note_p)<=total_samples:chords_track[pos:pos+len(note_p)]+=note_p*vel*0.15
                note_s=violin_note(freq,chord_duration*0.95,sr)
                if pos+len(note_s)<=total_samples:chords_track[pos:pos+len(note_s)]+=note_s*vel*0.1
    
    # MELODIA
    note_duration=beat_duration/2;current_degree=0;prev_section=None
    for i in range(int(duration/note_duration)):
        time=i*note_duration;section=structure.get_section_at_time(time)
        section_name=section['name'];energy=structure.get_energy_at_time(time)
        if section!=prev_section:current_degree=np.random.randint(0,len(scale));prev_section=section
        if section_name in ['intro','breakdown']:
            if np.random.random()<0.3:
                step=np.random.choice([-1,0,1])
                current_degree=max(0,min(current_degree+step,len(scale)-1))
                freq=note_to_freq(scale[current_degree],base_freq)
                pos=int(time*sr);nlen=int(note_duration*sr*3)
                note=flute_note(freq,nlen/sr,sr)
                if pos+len(note)<=total_samples:melody_track[pos:pos+len(note)]+=note*energy*0.2
        elif section_name in ['verse','head','theme']:
            if np.random.random()<0.7:
                step=np.random.choice([-2,-1,0,1,2],p=[0.1,0.25,0.3,0.25,0.1])
                current_degree=max(0,min(current_degree+step,len(scale)*2-1))
                octave=current_degree//len(scale);degree=current_degree%len(scale)
                freq=note_to_freq(scale[degree],base_freq)*(2**octave)
                pos=int(time*sr);nlen=int(note_duration*sr*1.5)
                note=piano_note(freq,nlen/sr,sr)
                if pos+len(note)<=total_samples:melody_track[pos:pos+len(note)]+=note*energy*0.25
        elif section_name in ['chorus','drop','climax']:
            if np.random.random()<0.85:
                step=np.random.choice([-3,-2,-1,0,1,2,3],p=[0.05,0.15,0.2,0.2,0.2,0.15,0.05])
                current_degree=max(0,min(current_degree+step,len(scale)*3-1))
                octave=current_degree//len(scale);degree=current_degree%len(scale)
                freq=note_to_freq(scale[degree],base_freq)*(2**octave)
                pos=int(time*sr)
                dur_mult=np.random.choice([0.5,1.0,1.5,2.0],p=[0.3,0.4,0.2,0.1])
                nlen=int(note_duration*sr*1.8*dur_mult)
                inst_choice=np.random.choice(['piano','violin','brass'])
                if inst_choice=='piano':note=piano_note(freq,nlen/sr,sr)
                elif inst_choice=='violin':note=violin_note(freq,nlen/sr,sr)
                else:note=brass_note(freq,nlen/sr,sr)
                if pos+len(note)<=total_samples:melody_track[pos:pos+len(note)]+=note*energy*0.3
    
    # AUTO-MIXING com tracks separadas
    mixer = AutoMixer(sr)
    tracks = {
        'drums': drums_track,
        'bass': bass_track,
        'chords': chords_track,
        'melody': melody_track,
        'fx': fx_track
    }
    
    print("  🎛️ Auto-mixing (balanceamento automático)...")
    mix = mixer.mix_tracks(tracks)
    
    # Fade in/out
    fade_in=int(0.5*sr);fade_out=int(1.5*sr)
    if fade_in<len(mix):mix[:fade_in]*=np.linspace(0,1,fade_in)
    if fade_out<len(mix):mix[-fade_out:]*=np.linspace(1,0,fade_out)
    
    metadata = {
        "bpm": bpm,
        "style": style,
        "duration": duration,
        "seed": seed,
        "main_instrument": "mixed"
    }
    
    return mix, sr, metadata


# ============================================================
# MAIN COM FEEDBACK LOOP
# ============================================================

def generate_music(prompt=None,style=None,duration=45):
    """Gera música com feedback loop (aprende com erros)"""
    print("="*60)
    print("🧠 IA MUSIC - COM FEEDBACK LOOP")
    print("="*60)
    
    feedback = FeedbackLoop()
    
    # Mostrar relatório de aprendizado
    report = feedback.get_report()
    print(f"  📊 Histórico:")
    print(f"     Músicas geradas: {report['total_geradas']}")
    print(f"     Taxa de sucesso: {report['taxa_sucesso']:.1f}%")
    print(f"     Qualidade média: {report['qualidade_media']:.2f}")
    print(f"     Sucessos na memória: {report['sucessos']}")
    print(f"     Erros na memória: {report['erros']}")
    print()
    
    # Obter parâmetros aprendidos
    learned = feedback.memory.get_learned_params()
    bpm = np.random.randint(learned['preferred_bpm_range'][0], learned['preferred_bpm_range'][1]+1)
    final_style = style or 'pop'
    
    print(f"  🎯 Parâmetros aprendidos:")
    print(f"     BPM preferido: {learned['preferred_bpm_range']}")
    print(f"     Intensidade preferida: {learned['preferred_intensity']:.2f}")
    if learned['avoid_patterns']:
        print(f"     Evitando padrões: {learned['avoid_patterns'][:3]}")
    
    print()
    
    # Feedback loop: gera até atingir qualidade
    def generation_attempt():
        return generate_with_automix(
            duration=duration,
            bpm=bpm,
            style=final_style,
            sr=44100
        )
    
    audio, sr, metadata = feedback.generate_with_feedback(
        generate_func=generation_attempt,
        max_attempts=3,
        min_quality=0.55
    )
    
    # Relatório final
    print()
    print("  📊 Relatório de aprendizado atualizado:")
    report = feedback.get_report()
    print(f"     Sucessos: {report['sucessos']}")
    print(f"     Erros: {report['erros']}")
    print(f"     Taxa de sucesso: {report['taxa_sucesso']:.1f}%")
    
    return audio, sr, metadata


def batch_generate():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--prompt",type=str,default="")
    parser.add_argument("--style",type=str,default="pop")
    parser.add_argument("--duration",type=int,default=45)
    parser.add_argument("--batch",action="store_true")
    args=parser.parse_args()
    
    print("="*60)
    print("🎵 IA MUSIC - FEEDBACK LOOP (LINUX)")
    print("="*60)
    
    audio,sr,metadata=generate_music(
        prompt=args.prompt,
        style=args.style,
        duration=args.duration
    )
    
    filepath,number=save_song(audio,sr,metadata)
    print(f"\n✅ Música #{number}: {filepath}")


def main():
    while True:
        print("="*60)
        print("🎵 IA MUSIC PRO - FEEDBACK LOOP")
        print("="*60)
        print("1. 🎵 Gerar música (com aprendizado)")
        print("2. 📊 Ver relatório de aprendizado")
        print("3. 🧠 Ver memórias")
        print("4. 🗑️ Limpar memórias")
        print("5. ❌ Sair")
        choice=input("\nOpção: ").strip()
        
        if choice=='1':
            prompt=input("Prompt (opcional): ").strip()
            style=input("Estilo (pop/rock/electronic/cinematic/jazz/ambient): ").strip() or "pop"
            duration=input("Duração (30/45/60/90): ").strip()
            if duration not in ["30","45","60","90"]:duration="45"
            audio,sr,metadata=generate_music(prompt=prompt,style=style,duration=int(duration))
            fp,num=save_song(audio,sr,metadata)
            print(f"\n✅ Música #{num}: {fp}")
            input("ENTER...")
        elif choice=='2':
            feedback = FeedbackLoop()
            report = feedback.get_report()
            print(f"\n📊 RELATÓRIO DE APRENDIZADO:")
            print(f"   Músicas geradas: {report['total_geradas']}")
            print(f"   Taxa de sucesso: {report['taxa_sucesso']:.1f}%")
            print(f"   Qualidade média: {report['qualidade_media']:.3f}")
            print(f"   Melhor qualidade: {report['melhor_qualidade']:.3f}")
            print(f"   Sucessos na memória: {report['sucessos']}")
            print(f"   Erros na memória: {report['erros']}")
            input("\nENTER...")
        elif choice=='3':
            memory = MusicMemory()
            print(f"\n🧠 MEMÓRIAS:")
            print(f"   Sucessos: {len(memory.successes)}")
            if memory.successes:
                print("   Últimos 3 sucessos:")
                for s in memory.successes[-3:]:
                    print(f"     - BPM={s.get('bpm')}, style={s.get('style')}, quality={s.get('quality',0):.2f}")
            print(f"   Erros: {len(memory.errors)}")
            if memory.errors:
                print("   Últimos 3 erros:")
                for e in memory.errors[-3:]:
                    print(f"     - Motivo: {e.get('reason','?')}")
            input("\nENTER...")
        elif choice=='4':
            if input("Tem certeza? (s/n): ").strip().lower()=='s':
                memory = MusicMemory()
                memory.successes = []
                memory.errors = []
                memory.stats = {
                    "total_generated": 0, "total_accepted": 0,
                    "total_rejected": 0, "average_quality": 0.0,
                    "best_quality": 0.0, "worst_quality": 1.0
                }
                memory._save(memory.success_file, [])
                memory._save(memory.errors_file, [])
                memory._save(memory.stats_file, memory.stats)
                print("✅ Memórias limpas!")
            input("ENTER...")
        elif choice=='5':
            print("\n👋 Até a próxima! 🎵")
            break


if __name__=="__main__":
    if "--batch" in sys.argv:batch_generate()
    else:main()
