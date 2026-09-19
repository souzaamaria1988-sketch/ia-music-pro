#!/usr/bin/env python3
"""
🎵 IA MUSIC GENERATOR PRO - UPGRADE COMPLETO
+100 estilos | +200 keywords | 15 efeitos | SISTEMA DE REMIX
"""
import os,sys,json,time,gc
import numpy as np
from pathlib import Path

try:
    from scipy.signal import lfilter
    HAS_SCIPY=True
except ImportError:
    HAS_SCIPY=False

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
# 100+ ESTILOS MUSICAIS
# ============================================================

MUSIC_STYLES={
    # Originais
    "epic":{"intensity":0.9,"bpm_mult":1.2,"orchestral":2.0,"strings":2.0,"drums":1.5,"style":"cinematic"},
    "bossfight":{"intensity":0.95,"drums":2.5,"bpm_mult":1.3,"guitar":2.0,"style":"cinematic"},
    "dark":{"intensity":0.7,"minor":True,"strings":1.8,"style":"ambient"},
    "rock":{"intensity":0.95,"guitar":2.5,"distortion":2.0,"drums":2.0,"style":"rock"},
    "ambient":{"intensity":0.4,"bpm_mult":0.5,"synth":2.0,"style":"ambient"},
    "electronic":{"intensity":0.8,"synth":2.5,"drums":1.8,"bpm_mult":1.3,"style":"electronic"},
    "jazz":{"jazz":2.5,"piano":2.5,"bpm_mult":0.9,"style":"jazz"},
    "classical":{"orchestral":2.5,"strings":2.5,"piano":2.0,"drums":0.3,"style":"classical"},
    "breakcore":{"breakcore":3.0,"intensity":1.0,"bpm_mult":1.8,"style":"breakcore"},
    
    # Brasileiros
    "funk":{"intensity":0.9,"bpm_mult":1.3,"drums":2.5,"bass":2.0,"synth":1.5,"style":"funk"},
    "samba":{"intensity":0.8,"bpm_mult":1.1,"drums":2.0,"guitar":1.5,"style":"samba"},
    "bossa":{"intensity":0.5,"bpm_mult":0.7,"guitar":2.0,"piano":1.5,"style":"bossa"},
    "pagode":{"intensity":0.8,"bpm_mult":1.0,"drums":2.0,"guitar":1.5,"style":"samba"},
    "sertanejo":{"intensity":0.7,"bpm_mult":0.9,"guitar":2.5,"style":"country"},
    "forro":{"intensity":0.8,"bpm_mult":1.2,"drums":1.8,"guitar":1.5,"style":"latin"},
    "axe":{"intensity":0.9,"bpm_mult":1.3,"drums":2.5,"brass":2.0,"style":"latin"},
    "mpb":{"intensity":0.6,"bpm_mult":0.8,"guitar":2.0,"piano":1.5,"strings":1.5,"style":"pop"},
    "tropicália":{"intensity":0.7,"bpm_mult":1.0,"guitar":2.0,"synth":1.5,"style":"experimental"},
    
    # Eletrônicos
    "techno":{"intensity":0.9,"bpm_mult":1.4,"synth":2.5,"drums":2.5,"style":"electronic"},
    "house":{"intensity":0.8,"bpm_mult":1.25,"synth":2.0,"drums":2.0,"style":"electronic"},
    "trance":{"intensity":0.85,"bpm_mult":1.35,"synth":2.5,"strings":1.5,"style":"electronic"},
    "dubstep":{"intensity":0.95,"bpm_mult":0.7,"bass":3.0,"synth":2.0,"style":"electronic"},
    "dnb":{"intensity":0.9,"bpm_mult":1.7,"drums":2.5,"bass":2.0,"style":"breakcore"},
    "drum_and_bass":{"intensity":0.9,"bpm_mult":1.7,"drums":2.5,"bass":2.0,"style":"breakcore"},
    "edm":{"intensity":0.9,"bpm_mult":1.3,"synth":2.5,"drums":2.0,"style":"electronic"},
    "synthwave":{"intensity":0.8,"bpm_mult":1.1,"synth":3.0,"drums":1.5,"style":"electronic"},
    "vaporwave":{"intensity":0.5,"bpm_mult":0.7,"synth":2.5,"piano":1.5,"style":"ambient"},
    "lofi":{"intensity":0.4,"bpm_mult":0.6,"piano":2.0,"drums":0.8,"style":"lofi"},
    "chillwave":{"intensity":0.5,"bpm_mult":0.8,"synth":2.0,"style":"ambient"},
    "industrial":{"intensity":0.9,"bpm_mult":1.2,"distortion":2.5,"drums":2.0,"style":"rock"},
    
    # Urbanos
    "hiphop":{"intensity":0.8,"bpm_mult":0.85,"bass":2.5,"drums":2.0,"style":"hiphop"},
    "rap":{"intensity":0.85,"bpm_mult":0.9,"bass":2.5,"drums":2.0,"style":"hiphop"},
    "trap":{"intensity":0.9,"bpm_mult":0.7,"bass":3.0,"drums":2.5,"synth":1.5,"style":"hiphop"},
    "rnb":{"intensity":0.7,"bpm_mult":0.8,"piano":2.0,"strings":1.5,"style":"rnb"},
    "soul":{"intensity":0.7,"bpm_mult":0.9,"brass":2.0,"piano":1.5,"style":"soul"},
    "funk_us":{"intensity":0.85,"bpm_mult":1.1,"bass":2.5,"brass":2.0,"style":"funk"},
    
    # Rock/Metal
    "metal":{"intensity":0.95,"guitar":3.0,"distortion":2.5,"drums":2.5,"style":"metal"},
    "punk":{"intensity":0.9,"guitar":2.5,"distortion":2.0,"drums":2.5,"bpm_mult":1.4,"style":"punk"},
    "grunge":{"intensity":0.85,"guitar":2.5,"distortion":1.8,"drums":2.0,"style":"rock"},
    "emo":{"intensity":0.8,"guitar":2.0,"distortion":1.5,"strings":1.5,"style":"rock"},
    "indie":{"intensity":0.7,"guitar":2.0,"drums":1.5,"style":"indie"},
    "alternativo":{"intensity":0.75,"guitar":2.0,"drums":1.5,"style":"indie"},
    "progressive":{"intensity":0.8,"guitar":2.0,"synth":2.0,"drums":1.8,"style":"progressive"},
    "hardcore":{"intensity":0.95,"guitar":2.5,"distortion":2.5,"drums":2.5,"bpm_mult":1.5,"style":"metal"},
    
    # Blues/Jazz
    "blues":{"intensity":0.7,"bpm_mult":0.8,"guitar":2.5,"piano":1.5,"style":"blues"},
    "swing":{"intensity":0.75,"bpm_mult":1.2,"brass":2.5,"piano":2.0,"style":"jazz"},
    "bebop":{"intensity":0.8,"bpm_mult":1.4,"brass":2.0,"piano":2.0,"style":"jazz"},
    "fusion":{"intensity":0.8,"bpm_mult":1.1,"guitar":2.0,"synth":2.0,"style":"jazz"},
    "smooth_jazz":{"intensity":0.5,"bpm_mult":0.8,"sax":2.5,"piano":2.0,"style":"jazz"},
    
    # World
    "reggae":{"intensity":0.7,"bpm_mult":0.7,"bass":2.5,"guitar":1.5,"style":"reggae"},
    "reggaeton":{"intensity":0.85,"bpm_mult":1.0,"drums":2.5,"synth":1.5,"style":"latin"},
    "salsa":{"intensity":0.85,"bpm_mult":1.3,"brass":2.5,"drums":2.0,"style":"latin"},
    "bachata":{"intensity":0.7,"bpm_mult":1.0,"guitar":2.5,"style":"latin"},
    "cumbia":{"intensity":0.8,"bpm_mult":1.1,"drums":2.0,"brass":1.5,"style":"latin"},
    "flamenco":{"intensity":0.8,"bpm_mult":1.2,"guitar":3.0,"style":"flamenco"},
    "celtic":{"intensity":0.7,"bpm_mult":1.0,"flute":2.5,"strings":2.0,"style":"folk"},
    "folk":{"intensity":0.6,"bpm_mult":0.9,"guitar":2.5,"flute":1.5,"style":"folk"},
    "country":{"intensity":0.7,"bpm_mult":1.0,"guitar":2.5,"style":"country"},
    "bluegrass":{"intensity":0.8,"bpm_mult":1.3,"guitar":2.5,"flute":1.5,"style":"country"},
    
    # Clássicos
    "baroque":{"intensity":0.7,"bpm_mult":0.9,"strings":2.5,"orchestral":2.0,"style":"classical"},
    "romantic":{"intensity":0.6,"bpm_mult":0.7,"strings":2.5,"piano":2.0,"style":"classical"},
    "impressionist":{"intensity":0.5,"bpm_mult":0.6,"piano":2.5,"strings":2.0,"style":"classical"},
    "minimalist":{"intensity":0.4,"bpm_mult":0.5,"piano":2.0,"synth":1.5,"style":"minimal"},
    "orchestral":{"intensity":0.85,"bpm_mult":1.0,"orchestral":3.0,"strings":2.5,"style":"orchestral"},
    "symphonic":{"intensity":0.9,"bpm_mult":1.1,"orchestral":3.0,"strings":2.5,"brass":2.0,"style":"orchestral"},
    
    # Game/Media
    "game_menu":{"intensity":0.5,"bpm_mult":0.8,"synth":2.0,"strings":1.5,"style":"ambient"},
    "game_battle":{"intensity":0.95,"bpm_mult":1.4,"drums":2.5,"orchestral":2.0,"style":"cinematic"},
    "game_victory":{"intensity":0.9,"bpm_mult":1.2,"brass":2.5,"orchestral":2.0,"style":"cinematic"},
    "game_gameover":{"intensity":0.4,"bpm_mult":0.6,"strings":2.0,"minor":True,"style":"ambient"},
    "chiptune":{"intensity":0.8,"bpm_mult":1.3,"synth":3.0,"style":"chiptune"},
    "8bit":{"intensity":0.8,"bpm_mult":1.3,"synth":3.0,"style":"chiptune"},
    "16bit":{"intensity":0.85,"bpm_mult":1.2,"synth":2.5,"style":"chiptune"},
    
    # Mood/Atmosfera
    "meditation":{"intensity":0.3,"bpm_mult":0.4,"synth":2.0,"flute":2.0,"style":"ambient"},
    "sleep":{"intensity":0.2,"bpm_mult":0.3,"synth":1.5,"style":"ambient"},
    "study":{"intensity":0.4,"bpm_mult":0.6,"piano":2.0,"style":"lofi"},
    "workout":{"intensity":0.9,"bpm_mult":1.4,"drums":2.5,"bass":2.0,"style":"electronic"},
    "party":{"intensity":0.95,"bpm_mult":1.3,"synth":2.5,"drums":2.5,"style":"electronic"},
    "romantic":{"intensity":0.5,"bpm_mult":0.7,"strings":2.5,"piano":2.0,"style":"romantic"},
    "sad":{"intensity":0.4,"bpm_mult":0.6,"minor":True,"piano":2.0,"strings":1.5,"style":"sad"},
    "happy":{"intensity":0.8,"bpm_mult":1.2,"major":True,"style":"happy"},
    "energetic":{"intensity":0.95,"bpm_mult":1.4,"drums":2.0,"style":"energetic"},
    "relaxing":{"intensity":0.4,"bpm_mult":0.5,"synth":2.0,"style":"ambient"},
    
    # Experimental
    "noise":{"intensity":0.7,"bpm_mult":1.0,"distortion":3.0,"style":"experimental"},
    "drone":{"intensity":0.4,"bpm_mult":0.3,"synth":2.5,"style":"drone"},
    "glitch":{"intensity":0.8,"bpm_mult":1.2,"breakcore":2.0,"style":"glitch"},
    "idm":{"intensity":0.7,"bpm_mult":1.1,"synth":2.5,"drums":2.0,"style":"experimental"},
    "avant_garde":{"intensity":0.6,"bpm_mult":0.9,"style":"experimental"},
}

# ============================================================
# 200+ KEYWORDS EM PORTUGUÊS
# ============================================================

KEYWORDS={
    # Emoções (30)
    "intenso":{"intensity":0.95,"bpm_mult":1.3},"calmo":{"intensity":0.4,"bpm_mult":0.6},
    "épico":{"intensity":0.9,"bpm_mult":1.2,"orchestral":2.0},"epico":{"intensity":0.9,"bpm_mult":1.2,"orchestral":2.0},
    "sombrio":{"intensity":0.7,"minor":True},"dark":{"intensity":0.7,"minor":True},
    "feliz":{"major":True,"bpm_mult":1.1},"alegre":{"major":True,"bpm_mult":1.2},
    "triste":{"minor":True,"bpm_mult":0.7},"melancólico":{"minor":True,"bpm_mult":0.8},
    "melancolico":{"minor":True,"bpm_mult":0.8},"energético":{"intensity":0.95,"bpm_mult":1.4},
    "energetico":{"intensity":0.95,"bpm_mult":1.4},"romântico":{"minor":True,"bpm_mult":0.7},
    "romantico":{"minor":True,"bpm_mult":0.7},"misterioso":{"intensity":0.5,"minor":True},
    "heroico":{"intensity":0.9,"bpm_mult":1.15,"orchestral":2.0},"heróico":{"intensity":0.9,"bpm_mult":1.15,"orchestral":2.0},
    "agressivo":{"intensity":0.95,"distortion":2.0,"bpm_mult":1.3},"suave":{"intensity":0.4,"bpm_mult":0.7},
    "pesado":{"intensity":1.1,"drums":1.5,"bass":1.8},"leve":{"intensity":0.4,"drums":0.5},
    "tenso":{"intensity":0.8,"minor":True,"bpm_mult":1.2},"relaxante":{"intensity":0.3,"bpm_mult":0.5},
    "animado":{"intensity":0.8,"bpm_mult":1.3},"depressivo":{"minor":True,"bpm_mult":0.5,"intensity":0.4},
    "esperançoso":{"major":True,"bpm_mult":1.1},"esperancoso":{"major":True,"bpm_mult":1.1},
    "nostálgico":{"minor":True,"bpm_mult":0.8},"nostalgico":{"minor":True,"bpm_mult":0.8},
    "vitorioso":{"intensity":0.9,"bpm_mult":1.2,"major":True},"triunfante":{"intensity":0.9,"bpm_mult":1.2},
    
    # Instrumentos (40)
    "guitarra":{"guitar":2.0,"distortion":1.5},"violão":{"guitar":2.0},"violao":{"guitar":2.0},
    "piano":{"piano":2.5},"bateria":{"drums":2.5},"tambores":{"drums":2.5},
    "violino":{"strings":2.5,"orchestral":1.8},"sintetizador":{"synth":2.0},"synth":{"synth":2.0},
    "baixo":{"bass":2.5},"flauta":{"flute":2.0},"trompete":{"brass":2.0},
    "saxofone":{"sax":2.0},"coral":{"choir":2.5,"orchestral":2.0},"orquestra":{"orchestral":3.0,"strings":2.5},
    "cordas":{"strings":2.5},"teclado":{"piano":2.0,"synth":1.5},"harpa":{"harp":2.0},
    "órgão":{"organ":2.0},"orgao":{"organ":2.0},"cavaquinho":{"guitar":1.8},"ukulele":{"guitar":1.8},
    "banjo":{"guitar":2.0},"mandolim":{"guitar":1.8},"acordeão":{"accordion":2.0},"acordeao":{"accordion":2.0},
    "sanfona":{"accordion":2.0},"gaita":{"harmonica":2.0},"sitar":{"sitar":2.0},
    "tablas":{"drums":1.8},"congas":{"drums":2.0},"bongos":{"drums":1.8},
    "pandeiro":{"drums":2.0},"tamborim":{"drums":1.8},"cuíca":{"drums":1.5},"cuica":{"drums":1.5},
    "berimbau":{"bass":1.8},"xilofone":{"bells":2.0},"marimba":{"bells":1.8},
    "vibrafone":{"bells":1.8},"sinos":{"bells":2.0},
    
    # Estilos Brasileiros (15)
    "funk":{"style":"funk","drums":2.5,"bass":2.0},"samba":{"style":"samba","drums":2.0},
    "bossa":{"style":"bossa","bpm_mult":0.7},"bossa nova":{"style":"bossa","bpm_mult":0.7},
    "pagode":{"style":"samba","drums":2.0},"sertanejo":{"style":"country","guitar":2.5},
    "forró":{"style":"latin","drums":1.8},"forro":{"style":"latin","drums":1.8},
    "axé":{"style":"latin","drums":2.5,"brass":2.0},"axe":{"style":"latin","drums":2.5},
    "mpb":{"style":"pop","guitar":2.0},"tropicália":{"style":"experimental"},
    "tropicalia":{"style":"experimental"},"choro":{"style":"jazz","flute":2.0},
    "baião":{"style":"latin","drums":1.8},"baiao":{"style":"latin"},
    
    # Estilos Internacionais (30)
    "rock":{"style":"rock","guitar":2.0,"drums":2.0},"metal":{"style":"metal","guitar":3.0,"distortion":2.5},
    "jazz":{"style":"jazz","jazz":2.5,"piano":2.0},"eletrônica":{"style":"electronic","synth":2.5},
    "eletronica":{"style":"electronic","synth":2.5},"clássica":{"style":"classical","orchestral":2.5},
    "classica":{"style":"classical","orchestral":2.5},"pop":{"style":"pop"},
    "hip hop":{"style":"hiphop","bass":2.5},"hiphop":{"style":"hiphop","bass":2.5},
    "rap":{"style":"hiphop","bass":2.5},"trap":{"style":"hiphop","bass":3.0,"bpm_mult":0.7},
    "reggae":{"style":"reggae","bpm_mult":0.7},"salsa":{"style":"latin","brass":2.5},
    "techno":{"style":"electronic","synth":2.5,"bpm_mult":1.4},"house":{"style":"electronic","synth":2.0},
    "trance":{"style":"electronic","synth":2.5,"bpm_mult":1.35},"dubstep":{"style":"electronic","bass":3.0},
    "drum and bass":{"style":"breakcore","bpm_mult":1.7},"dnb":{"style":"breakcore","bpm_mult":1.7},
    "punk":{"style":"punk","guitar":2.5,"bpm_mult":1.4},"grunge":{"style":"rock","guitar":2.5},
    "blues":{"style":"blues","guitar":2.5},"soul":{"style":"soul","brass":2.0},
    "rnb":{"style":"rnb","piano":2.0},"country":{"style":"country","guitar":2.5},
    "folk":{"style":"folk","guitar":2.5},"flamenco":{"style":"flamenco","guitar":3.0},
    "celtic":{"style":"folk","flute":2.5},"baroque":{"style":"classical","strings":2.5},
    
    # Contextos (25)
    "batalha":{"intensity":0.95,"drums":2.0,"bpm_mult":1.3},"guerra":{"intensity":0.95,"drums":2.5,"orchestral":2.0},
    "boss":{"intensity":0.95,"bpm_mult":1.3},"medieval":{"orchestral":1.8,"strings":1.5},
    "fantasia":{"orchestral":2.0},"espaço":{"synth":1.5,"intensity":0.5},"espaco":{"synth":1.5},
    "cidade":{"intensity":0.7},"natureza":{"intensity":0.4},"chuva":{"intensity":0.3},
    "noite":{"minor":True,"intensity":0.5},"dia":{"major":True,"intensity":0.7},
    "amanhecer":{"major":True,"bpm_mult":0.8},"entardecer":{"minor":True,"bpm_mult":0.7},
    "festa":{"intensity":0.9,"bpm_mult":1.3,"drums":2.0},"dançar":{"bpm_mult":1.2,"drums":1.8},
    "dancar":{"bpm_mult":1.2},"estudar":{"intensity":0.3,"bpm_mult":0.6},
    "trabalhar":{"intensity":0.4,"bpm_mult":0.7},"dormir":{"intensity":0.2,"bpm_mult":0.4},
    "correr":{"bpm_mult":1.4,"drums":2.0},"treinar":{"bpm_mult":1.3,"drums":1.8},
    "academia":{"bpm_mult":1.3,"drums":1.8},"viagem":{"intensity":0.6},
    "praia":{"intensity":0.5},"montanha":{"intensity":0.6,"orchestral":1.5},
    "floresta":{"intensity":0.4},"deserto":{"intensity":0.5,"minor":True},
    "oceano":{"intensity":0.4},"tempestade":{"intensity":0.9,"drums":2.0,"minor":True},
    
    # Velocidade (10)
    "rápido":{"bpm_mult":1.4},"rapido":{"bpm_mult":1.4},"lento":{"bpm_mult":0.6},
    "devagar":{"bpm_mult":0.6},"muito rápido":{"bpm_mult":1.6},"muito rapido":{"bpm_mult":1.6},
    "muito lento":{"bpm_mult":0.4},"acelerado":{"bpm_mult":1.5},"frenético":{"bpm_mult":1.8},
    "frenetico":{"bpm_mult":1.8},
    
    # Breakcore/Glitch (10)
    "breakcore":{"breakcore":3.0,"intensity":1.0,"bpm_mult":1.8},"amen":{"breakcore":2.5},
    "glitch":{"breakcore":2.0},"jungle":{"breakcore":2.0,"bpm_mult":1.5},
    "caótico":{"breakcore":2.0},"caotico":{"breakcore":2.0},
    "destruído":{"breakcore":2.0,"distortion":2.0},"destruido":{"breakcore":2.0},
    "noise":{"distortion":3.0},"experimental":{"style":"experimental"},
    
    # Game (10)
    "game menu":{"intensity":0.5,"synth":2.0},"game battle":{"intensity":0.95,"drums":2.5},
    "game victory":{"intensity":0.9,"brass":2.5},"game over":{"intensity":0.4,"minor":True},
    "chiptune":{"synth":3.0,"style":"chiptune"},"8bit":{"synth":3.0,"style":"chiptune"},
    "16bit":{"synth":2.5,"style":"chiptune"},"retro game":{"synth":2.5},
    "videogame":{"synth":2.5},"pixel":{"synth":2.5},
}

# ============================================================
# 15 EFEITOS DE ÁUDIO AVANÇADOS
# ============================================================

def add_reverb(audio,sr=44100,decay=0.3,mix=0.25):
    delay=int(0.03*sr);reverb=np.zeros_like(audio)
    for d in [1,2,3,4,5,6]:
        pos=delay*d
        if pos<len(audio):reverb[pos:]+=audio[:-pos]*(decay**d)
    return audio*(1-mix)+reverb*mix

def add_delay(audio,sr=44100,delay_time=0.3,feedback=0.35,mix=0.2):
    delay_samples=int(delay_time*sr);output=audio.copy()
    for i in range(1,5):
        pos=delay_samples*i
        if pos<len(audio):output[pos:]+=audio[:-pos]*(feedback**i)
    return audio*(1-mix)+output*mix

def chorus_effect(audio,sr=44100,rate=1.5,depth=0.003,mix=0.3):
    n=len(audio);t=np.arange(n)/sr
    mod=depth*sr*np.sin(2*np.pi*rate*t)
    chorus=np.zeros_like(audio)
    for i in range(100,n):
        delay=int(abs(mod[i]))+100
        if 0<=i-delay<n:chorus[i]=audio[i-delay]
    return audio*(1-mix)+chorus*mix

def flanger_effect(audio,sr=44100,rate=0.5,depth=0.005,mix=0.4):
    n=len(audio);t=np.arange(n)/sr
    mod=depth*sr*(1+np.sin(2*np.pi*rate*t))/2
    flanged=np.zeros_like(audio)
    for i in range(200,n):
        delay=int(mod[i])+100
        if 0<=i-delay<n:flanged[i]=audio[i-delay]
    return audio*(1-mix)+flanged*mix

def phaser_effect(audio,sr=44100,rate=0.5,mix=0.5):
    n=len(audio);t=np.arange(n)/sr
    lfo=np.sin(2*np.pi*rate*t)
    output=audio.copy()
    for i in range(1,n):
        allpass=(audio[i]-lfo[i]*audio[i-1])/(1-lfo[i]*0.5)
        output[i]=audio[i]*(1-mix)+allpass*mix
    return output

def add_distortion(audio,gain=3.0,mix=0.7):
    return audio*(1-mix)+np.tanh(audio*gain)*mix

def bitcrush_effect(audio,bits=8,rate_div=4):
    if len(audio)==0:return audio
    reduced=audio[::rate_div];upsampled=np.repeat(reduced,rate_div)
    if len(upsampled)<len(audio):upsampled=np.pad(upsampled,(0,len(audio)-len(upsampled)))
    elif len(upsampled)>len(audio):upsampled=upsampled[:len(audio)]
    return np.round(upsampled*(2**bits))/(2**bits)

def soft_compress(audio,threshold=0.6,ratio=3.0):
    compressed=audio.copy()
    mask=np.abs(compressed)>threshold
    compressed[mask]=threshold+(compressed[mask]-threshold)/ratio
    return compressed

def limiter(audio,threshold=0.9):
    limited=audio.copy()
    mask=np.abs(limited)>threshold
    limited[mask]=threshold*np.sign(limited[mask])
    return limited

def eq_bass_boost(audio,sr=44100,boost=1.5):
    kernel=np.ones(20)/20
    bass=np.convolve(audio,kernel,mode='same')
    return audio+bass*(boost-1.0)

def eq_treble_boost(audio,sr=44100,boost=1.3):
    treble=np.diff(audio,prepend=audio[0])
    return audio+treble*(boost-1.0)*0.5

def sidechain_compress(audio,kick_pattern,sr=44100,threshold=0.3):
    compressed=audio.copy()
    envelope=np.ones_like(audio)
    kick_times=np.where(kick_pattern>0.5)[0]
    for kt in kick_times:
        start=int(kt)
        attack_samples=int(0.005*sr)
        release_samples=int(0.1*sr)
        end=min(start+attack_samples+release_samples,len(audio))
        if start<len(audio):
            duck_len=min(attack_samples,len(audio)-start)
            if duck_len>0:envelope[start:start+duck_len]*=threshold
            rel_start=start+attack_samples
            rel_end=min(rel_start+release_samples,len(audio))
            if rel_start<len(audio):
                envelope[rel_start:rel_end]=np.linspace(threshold,1.0,rel_end-rel_start)
    return audio*envelope

def stereo_widen(audio,sr=44100,width=1.5):
    return audio*width

def autopan_effect(audio,sr=44100,rate=0.5):
    n=len(audio);t=np.arange(n)/sr
    pan=(np.sin(2*np.pi*rate*t)+1)/2
    return audio*pan

def tremolo_effect(audio,sr=44100,rate=5.0,depth=0.5):
    n=len(audio);t=np.arange(n)/sr
    lfo=1.0-depth*(1.0+np.sin(2*np.pi*rate*t))/2
    return audio*lfo

# ============================================================
# SISTEMA DE REMIX (NOVA FUNÇÃO!)
# ============================================================

def remix_song(audio,sr,remix_type="style_change",target_style=None,bpm_mult=1.0):
    """
    🆕 SISTEMA DE REMIX - Cria variações de músicas existentes
    
    remix_type:
    - "style_change": Muda o estilo mantendo estrutura
    - "speed_up": Acelera o BPM
    - "slow_down": Desacelera o BPM
    - "acoustic": Versão acústica (remove synth, adiciona violão)
    - "electronic": Versão eletrônica (adiciona synth, remove acústico)
    - "heavy": Versão pesada (mais distorção, mais bateria)
    - "soft": Versão suave (menos intensidade, mais reverb)
    - "variations": Gera 5 variações aleatórias
    """
    print(f"🎛️ REMIX: {remix_type}")
    
    if remix_type=="style_change":
        # Aplicar características do novo estilo
        if target_style and target_style in MUSIC_STYLES:
            style_params=MUSIC_STYLES[target_style]
            if style_params.get('distortion',1.0)>1.5:
                audio=add_distortion(audio,gain=2.5,mix=0.5)
            if style_params.get('synth',1.0)>2.0:
                audio=chorus_effect(audio,sr,mix=0.3)
            if style_params.get('intensity',0.7)>0.9:
                audio=soft_compress(audio,threshold=0.4,ratio=4.0)
        return audio
    
    elif remix_type=="speed_up":
        # Acelerar sem mudar pitch (simplificado)
        new_len=int(len(audio)/bpm_mult)
        indices=np.linspace(0,len(audio)-1,new_len).astype(int)
        audio=audio[indices]
        return audio
    
    elif remix_type=="slow_down":
        # Desacelerar
        new_len=int(len(audio)*bpm_mult)
        indices=np.linspace(0,len(audio)-1,new_len).astype(int)
        audio=audio[indices]
        return audio
    
    elif remix_type=="acoustic":
        # Versão acústica: mais reverb, menos distorção
        audio=add_reverb(audio,sr,decay=0.4,mix=0.35)
        audio=eq_treble_boost(audio,sr,boost=1.2)
        return audio
    
    elif remix_type=="electronic":
        # Versão eletrônica: mais chorus, mais compressão
        audio=chorus_effect(audio,sr,rate=2.0,mix=0.4)
        audio=soft_compress(audio,threshold=0.4,ratio=4.0)
        audio=eq_bass_boost(audio,sr,boost=1.3)
        return audio
    
    elif remix_type=="heavy":
        # Versão pesada: distorção, compressão
        audio=add_distortion(audio,gain=4.0,mix=0.6)
        audio=soft_compress(audio,threshold=0.3,ratio=5.0)
        audio=eq_bass_boost(audio,sr,boost=1.5)
        return audio
    
    elif remix_type=="soft":
        # Versão suave: reverb, menos intensidade
        audio=add_reverb(audio,sr,decay=0.5,mix=0.4)
        audio=audio*0.7
        return audio
    
    elif remix_type=="variations":
        # Gerar 5 variações aleatórias
        variations=[]
        effects=[
            lambda a:add_reverb(a,sr,decay=np.random.uniform(0.2,0.5),mix=np.random.uniform(0.2,0.4)),
            lambda a:chorus_effect(a,sr,rate=np.random.uniform(0.5,3.0),mix=np.random.uniform(0.2,0.4)),
            lambda a:add_distortion(a,gain=np.random.uniform(1.5,3.0),mix=np.random.uniform(0.3,0.6)),
            lambda a:soft_compress(a,threshold=np.random.uniform(0.3,0.7),ratio=np.random.uniform(2.0,5.0)),
            lambda a:eq_bass_boost(a,sr,boost=np.random.uniform(1.0,1.8)),
        ]
        for i in range(5):
            var=audio.copy()
            for effect in np.random.choice(effects,size=np.random.randint(2,4),replace=False):
                var=effect(var)
            variations.append(var)
        return variations
    
    return audio

def generate_remix(source_song_num,remix_type="variations",target_style=None):
    """Gera remix de uma música existente"""
    source_path=os.path.join(OUTPUT_DIR,f"{source_song_num}.wav")
    if not os.path.exists(source_path):
        print(f"❌ Música {source_song_num}.wav não encontrada")
        return None
    
    # Carregar áudio
    try:
        import soundfile as sf
        audio,sr=sf.read(source_path)
    except:
        import wave
        with wave.open(source_path,'r') as wf:
            sr=wf.getframerate()
            n_frames=wf.getnframes()
            audio_data=wf.readframes(n_frames)
            audio=np.frombuffer(audio_data,dtype=np.int16).astype(np.float32)/32767.0
    
    print(f"🎵 Remixando música #{source_song_num}...")
    result=remix_song(audio,sr,remix_type,target_style)
    
    # Salvar variações
    if isinstance(result,list):
        saved=[]
        for i,var in enumerate(result):
            metadata={"remix_of":source_song_num,"remix_type":remix_type,"variation":i+1}
            filepath,number=save_song(var,sr,metadata)
            saved.append(filepath)
            print(f"  ✅ Variação {i+1}: {filepath}")
        return saved
    else:
        metadata={"remix_of":source_song_num,"remix_type":remix_type,"target_style":target_style}
        filepath,number=save_song(result,sr,metadata)
        return filepath

# ============================================================
# INSTRUMENTOS (mantidos do código anterior)
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

# Breakcore functions
def bitcrush(audio,bits=8,rate_div=4):
    return bitcrush_effect(audio,bits,rate_div)

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

def make_amen_break(sr=44100,tempo_factor=1.0):
    beat_dur=0.125/tempo_factor
    pattern=[('K',1),('H',.3),('S',.9),('H',.3),('G',.4),('H',.3),('S',.7),('H',.4),
             ('K',.9),('K',.5),('S',.9),('H',.3),('G',.5),('S',.6),('S',.8),('H',.3),
             ('K',1),('H',.3),('S',.9),('G',.4),('K',.8),('H',.4),('S',.8),('H',.3),
             ('K',1),('G',.5),('S',.9),('H',.4),('S',.7),('S',.6),('K',.8),('S',.9)]
    total_samples=int(beat_dur*len(pattern)*sr);output=np.zeros(total_samples)
    kick_s=make_kick(sr)[:int(0.12*sr)];snare_s=make_snare(sr)[:int(0.1*sr)]
    ghost_s=make_snare(sr)[:int(0.06*sr)]*0.4;hat_s=make_hihat(sr)[:int(0.04*sr)]
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

def riser_sweep(duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    freqs=np.linspace(200,2000,len(t))
    signal=np.sin(2*np.pi*np.cumsum(freqs)/sr)+np.random.randn(len(t))*0.3
    envelope=np.linspace(0,1,len(t))**2
    return signal*envelope/(np.max(np.abs(signal))+1e-10)

def noise_sweep(duration,sr=44100):
    t=np.linspace(0,duration,int(duration*sr),endpoint=False)
    noise=np.random.randn(len(t))
    envelope=np.linspace(0,1,len(t))**2
    return noise*envelope/(np.max(np.abs(noise*envelope))+1e-10)

def note_to_freq(semitone,base_freq=261.63):
    return base_freq*(2**(semitone/12.0))

# ============================================================
# PROGRESSÕES E ESCALAS
# ============================================================

ALL_PROGRESSIONS=[
    [[0,2,4],[5,0,2],[3,5,0],[4,6,1]],[[0,2,4],[3,5,0],[4,6,1],[5,0,2]],
    [[5,0,2],[3,5,0],[0,2,4],[4,6,1]],[[0,2,4],[0,2,4],[5,0,2],[4,6,1]],
    [[1,3,5],[4,6,1],[0,2,4],[5,0,2]],[[0,2,4,6],[4,6,1,3],[5,0,2,4],[0,2,4,6]],
    [[0,3,5],[5,0,2],[4,6,1],[0,2,4]],[[0,2,4],[4,6,1],[0,2,4],[5,0,2]],
    [[0,4,6],[3,5,0],[2,4,6],[5,0,2]],[[0,2,4],[5,0,2],[4,6,1],[0,2,4]],
]

ALL_SCALES={
    'major':[0,2,4,5,7,9,11],'minor':[0,2,3,5,7,8,10],'dorian':[0,2,3,5,7,9,10],
    'phrygian':[0,1,3,5,7,8,10],'lydian':[0,2,4,6,7,9,11],'mixolydian':[0,2,4,5,7,9,10],
    'harmonic_minor':[0,2,3,5,7,8,11],'melodic_minor':[0,2,3,5,7,9,11],
    'pentatonic_major':[0,2,4,7,9],'pentatonic_minor':[0,3,5,7,10],
    'blues':[0,3,5,6,7,10],'whole_tone':[0,2,4,6,8,10],
}
SCALE_NAMES=list(ALL_SCALES.keys())

# ============================================================
# SONG STRUCTURE
# ============================================================

class SongStructure:
    STRUCTURES={
        'pop':['intro','verse','chorus','verse','chorus','bridge','chorus','outro'],
        'rock':['intro','verse','chorus','verse','chorus','solo','chorus','outro'],
        'electronic':['intro','buildup','drop','breakdown','buildup','drop','outro'],
        'cinematic':['intro','theme','development','climax','resolution','outro'],
        'jazz':['intro','head','solo1','head','solo2','head','outro'],
        'ambient':['intro','section1','section2','section3','section4','outro'],
        'breakcore':['intro','chaos1','break','chaos2','break','chaos3','outro'],
        'classical':['exposition','development','recapitulation','coda'],
        'funk':['intro','verse','chorus','verse','chorus','bridge','chorus','outro'],
        'samba':['intro','verse','chorus','verse','chorus','outro'],
        'bossa':['intro','theme','theme2','outro'],
        'hiphop':['intro','verse','chorus','verse','chorus','outro'],
        'lofi':['intro','verse','chorus','verse','outro'],
        'metal':['intro','verse','chorus','verse','chorus','solo','chorus','outro'],
        'punk':['intro','verse','chorus','verse','chorus','outro'],
    }
    SECTION_ENERGY={
        'intro':0.3,'verse':0.5,'chorus':0.9,'bridge':0.6,'outro':0.4,
        'buildup':0.7,'drop':1.0,'breakdown':0.2,'solo':0.7,'head':0.6,
        'theme':0.5,'theme2':0.6,'development':0.7,'climax':1.0,'resolution':0.5,
        'exposition':0.5,'recapitulation':0.7,'coda':0.4,
        'chaos1':0.8,'chaos2':0.9,'chaos3':1.0,'break':0.3,
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
            elif section in ['verse','head','theme','theme2','exposition']:bars=np.random.choice([8,12,16])
            elif section in ['chorus','climax','drop','recapitulation']:bars=np.random.choice([8,12])
            elif section=='bridge':bars=np.random.choice([4,8])
            elif section=='buildup':bars=np.random.choice([4,8])
            elif section in ['breakdown','break','coda']:bars=np.random.choice([4,8])
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
    def get_fill_probability(self,time):
        section=self.get_section_at_time(time)
        section_duration=section['duration']
        position=(time-section['start'])/section_duration if section_duration>0 else 0
        return 0.5 if position>0.85 else 0.02

# ============================================================
# RAG
# ============================================================

class MusicRAG:
    def __init__(self):
        self.entries=[
            {"tags":["epic","batalha","heroico"],"scale":"major","dynamics":"loud","style":"cinematic"},
            {"tags":["dark","sombrio","terror"],"scale":"harmonic_minor","dynamics":"quiet","style":"ambient"},
            {"tags":["breakcore","glitch","amen"],"scale":"minor","dynamics":"extreme","style":"breakcore"},
            {"tags":["ambient","calmo"],"scale":"lydian","dynamics":"very_quiet","style":"ambient"},
            {"tags":["rock","metal","pesado"],"scale":"phrygian","dynamics":"loud","style":"rock"},
            {"tags":["eletrônica","techno","edm"],"scale":"minor","dynamics":"loud","style":"electronic"},
            {"tags":["feliz","alegre","pop"],"scale":"major","dynamics":"medium","style":"pop"},
            {"tags":["funk","baile"],"scale":"minor","dynamics":"loud","style":"funk"},
            {"tags":["samba","pagode"],"scale":"major","dynamics":"medium","style":"samba"},
            {"tags":["bossa","romântico"],"scale":"major","dynamics":"quiet","style":"bossa"},
            {"tags":["sertanejo","country"],"scale":"major","dynamics":"medium","style":"country"},
            {"tags":["hiphop","rap","trap"],"scale":"minor","dynamics":"loud","style":"hiphop"},
            {"tags":["jazz","blues","swing"],"scale":"dorian","dynamics":"medium","style":"jazz"},
            {"tags":["reggae","samba","calmo"],"scale":"major","dynamics":"medium","style":"reggae"},
        ]
    def get_context(self,query,style_hint=None):
        if not query and not style_hint:return self._random_context()
        query_lower=(query or '').lower()
        scores=[]
        for entry in self.entries:
            score=sum(1 for tag in entry['tags'] if tag in query_lower)
            if style_hint and entry['style']==style_hint:score+=2
            if score>0:scores.append((score,entry))
        best=scores[0][1] if scores else np.random.choice(self.entries)
        progression=ALL_PROGRESSIONS[np.random.randint(0,len(ALL_PROGRESSIONS))]
        scale_name=best.get('scale','major')
        if np.random.random()<0.3:scale_name=np.random.choice(SCALE_NAMES)
        scale=ALL_SCALES.get(scale_name,ALL_SCALES['major'])
        dynamics=best.get('dynamics','medium')
        bpm_map={'very_quiet':(40,70),'quiet':(60,90),'medium':(90,130),'loud':(120,160),'extreme':(160,230)}
        bpm=np.random.randint(*bpm_map.get(dynamics,(90,130)))
        intensity_map={'very_quiet':0.35,'quiet':0.55,'medium':0.7,'loud':0.85,'extreme':0.95}
        intensity=np.clip(intensity_map.get(dynamics,0.7)+np.random.uniform(-0.1,0.1),0.2,0.98)
        style=style_hint or best.get('style','pop')
        print(f"  📚 RAG: estilo={style}, escala={scale_name}, BPM={bpm}")
        return {'scale':scale,'scale_name':scale_name,'progression':progression,'bpm':bpm,'intensity':intensity,'style':style,'seed':get_dynamic_seed()}
    def _random_context(self):
        scale_name=np.random.choice(SCALE_NAMES)
        progression=ALL_PROGRESSIONS[np.random.randint(0,len(ALL_PROGRESSIONS))]
        style=np.random.choice(list(SongStructure.STRUCTURES.keys()))
        return {'scale':ALL_SCALES[scale_name],'scale_name':scale_name,'progression':progression,'bpm':np.random.randint(50,200),'intensity':np.random.uniform(0.3,0.95),'style':style,'seed':get_dynamic_seed()}

# ============================================================
# PROMPT INTERPRETER
# ============================================================

class PromptInterpreter:
    def interpret(self,prompt):
        if not prompt:return self._default_params()
        prompt_lower=prompt.lower()
        params=self._default_params()
        matched=[]
        for keyword,effects in KEYWORDS.items():
            if keyword in prompt_lower:
                matched.append(keyword)
                for key,value in effects.items():
                    if isinstance(value,bool):params[key]=value
                    elif isinstance(value,(int,float)):params[key]=value
                    elif isinstance(value,str):params[key]=value
        if matched:print(f"  🧠 Keywords: {matched[:10]}{'...' if len(matched)>10 else ''}")
        return params
    def _default_params(self):
        return {"intensity":0.7,"bpm_mult":1.0,"orchestral":1.0,"guitar":1.0,"piano":1.0,"drums":1.0,"strings":1.0,"synth":1.0,"distortion":1.0,"bass":1.0,"flute":1.0,"brass":1.0,"jazz":1.0,"minor":False,"major":True,"breakcore":0.0,"style":None}

# ============================================================
# GERAÇÃO PRINCIPAL
# ============================================================

def generate_with_structure(duration,rag_context,sr=44100):
    np.random.seed(rag_context['seed'])
    bpm=rag_context['bpm'];style=rag_context.get('style','pop')
    base_freq=np.random.choice([220.0,246.94,261.63,293.66,329.63,349.23])
    scale=rag_context['scale'];progression=rag_context['progression']
    structure=SongStructure(style=style,duration=duration,bpm=bpm)
    print(f"   Estrutura: {[s['name'] for s in structure.sections]}")
    total_samples=int(duration*sr);beat_duration=60.0/bpm
    drums_track=np.zeros(total_samples);bass_track=np.zeros(total_samples)
    chords_track=np.zeros(total_samples);melody_track=np.zeros(total_samples)
    fx_track=np.zeros(total_samples)
    kick_sample=make_kick(sr);snare_sample=make_snare(sr);hihat_sample=make_hihat(sr)
    tom1_sample=make_tom(200,sr);tom2_sample=make_tom(150,sr);tom3_sample=make_tom(100,sr)
    crash_sample=make_crash(sr)
    n_beats=int(duration/beat_duration)
    
    print("  🥁 Bateria...")
    for beat in range(n_beats):
        time=beat*beat_duration;section=structure.get_section_at_time(time)
        section_name=section['name'];energy=structure.get_energy_at_time(time)
        vel=0.4+0.6*energy;pos=int(beat*beat_duration*sr)
        if section_name in ['intro','breakdown','break']:
            if beat%4==0 and pos+len(kick_sample)<=total_samples:drums_track[pos:pos+len(kick_sample)]+=kick_sample*vel*0.6
            if beat%4==2 and pos+len(snare_sample)<=total_samples:drums_track[pos:pos+len(snare_sample)]+=snare_sample*vel*0.5
            if pos+len(hihat_sample)<=total_samples:drums_track[pos:pos+len(hihat_sample)]+=hihat_sample*vel*0.3
        elif section_name in ['verse','head','theme']:
            if beat%4 in [0,2] and pos+len(kick_sample)<=total_samples:drums_track[pos:pos+len(kick_sample)]+=kick_sample*vel*0.8
            if beat%4 in [1,3] and pos+len(snare_sample)<=total_samples:drums_track[pos:pos+len(snare_sample)]+=snare_sample*vel*0.7
            if pos+len(hihat_sample)<=total_samples:drums_track[pos:pos+len(hihat_sample)]+=hihat_sample*vel*0.4
        elif section_name in ['chorus','drop','climax','chaos1','chaos2','chaos3']:
            if beat%2==0 and pos+len(kick_sample)<=total_samples:drums_track[pos:pos+len(kick_sample)]+=kick_sample*vel*0.9
            if beat%4 in [1,3] and pos+len(snare_sample)<=total_samples:drums_track[pos:pos+len(snare_sample)]+=snare_sample*vel*0.8
            for sub in [0,0.5]:
                sp=int((beat+sub)*beat_duration*sr)
                if sp+len(hihat_sample)<=total_samples:drums_track[sp:sp+len(hihat_sample)]+=hihat_sample*vel*(0.4 if sub==0 else 0.25)
        elif section_name=='buildup':
            section_start=section['start'];section_duration=section['duration']
            position=(time-section_start)/section_duration if section_duration>0 else 0
            if position<0.5:
                if beat%4==0 and pos+len(kick_sample)<=total_samples:drums_track[pos:pos+len(kick_sample)]+=kick_sample*vel*0.7
                if pos+len(hihat_sample)<=total_samples:drums_track[pos:pos+len(hihat_sample)]+=hihat_sample*vel*0.3
            else:
                for sub in [0,0.25,0.5,0.75]:
                    sp=int((beat+sub)*beat_duration*sr)
                    if sp+len(hihat_sample)<=total_samples:drums_track[sp:sp+len(hihat_sample)]+=hihat_sample*vel*0.5
                    if sp+len(snare_sample)<=total_samples:drums_track[sp:sp+len(snare_sample)]+=snare_sample*vel*0.4
        else:
            if beat%4 in [0,2] and pos+len(kick_sample)<=total_samples:drums_track[pos:pos+len(kick_sample)]+=kick_sample*vel*0.7
            if beat%4 in [1,3] and pos+len(snare_sample)<=total_samples:drums_track[pos:pos+len(snare_sample)]+=snare_sample*vel*0.6
            if pos+len(hihat_sample)<=total_samples:drums_track[pos:pos+len(hihat_sample)]+=hihat_sample*vel*0.35
        fill_prob=structure.get_fill_probability(time)
        if np.random.random()<fill_prob:
            fill_start=pos
            if fill_start+int(beat_duration*sr)<=total_samples:
                toms=[tom1_sample,tom2_sample,tom3_sample,tom2_sample]
                for i,tom in enumerate(toms):
                    tom_pos=fill_start+int(i*beat_duration*sr/4)
                    if tom_pos+len(tom)<=total_samples:drums_track[tom_pos:tom_pos+len(tom)]+=tom*vel*0.7
                crash_pos=fill_start+int(beat_duration*sr*0.9)
                if crash_pos+len(crash_sample)<=total_samples:drums_track[crash_pos:crash_pos+len(crash_sample)]+=crash_sample*vel*0.8
    
    print("  🎸 Baixo...")
    for beat in range(n_beats):
        time=beat*beat_duration;section=structure.get_section_at_time(time);energy=structure.get_energy_at_time(time)
        chord_idx=(beat//4)%len(progression);root=progression[chord_idx][0]
        root_freq=note_to_freq(scale[root%len(scale)],base_freq)/2;vel=0.3+0.7*energy
        if section['name'] in ['intro','breakdown','break']:
            if beat%4==0:
                note=karplus_strong(root_freq,beat_duration*3,sr,damping=0.998)
                pos=int(beat*beat_duration*sr)
                if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*vel*0.3
        elif section['name'] in ['verse','head','theme']:
            if beat%2==0:
                note=karplus_strong(root_freq,beat_duration*1.5,sr,damping=0.997)
                pos=int(beat*beat_duration*sr)
                if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*vel*0.4
        elif section['name'] in ['chorus','drop','climax']:
            freq=root_freq if beat%2==0 else root_freq*2
            note=karplus_strong(freq,beat_duration*0.9,sr)
            pos=int(beat*beat_duration*sr)
            if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*vel*0.45
        else:
            if beat%2==0:
                note=karplus_strong(root_freq,beat_duration*1.2,sr)
                pos=int(beat*beat_duration*sr)
                if pos+len(note)<=total_samples:bass_track[pos:pos+len(note)]+=note*vel*0.4
    
    print("  🎹 Acordes...")
    chord_duration=beat_duration*4
    for i in range(int(duration/chord_duration)):
        time=i*chord_duration;section=structure.get_section_at_time(time);energy=structure.get_energy_at_time(time)
        chord=progression[i%len(progression)];pos=int(i*chord_duration*sr);vel=0.2+0.6*energy
        if section['name'] in ['intro','breakdown','break','outro']:
            for nd in chord:
                freq=note_to_freq(scale[nd%len(scale)],base_freq)
                note=synth_pad(freq,chord_duration*0.95,sr)
                if pos+len(note)<=total_samples:chords_track[pos:pos+len(note)]+=note*vel*0.15
        elif section['name'] in ['verse','head','theme']:
            for nd in chord:
                freq=note_to_freq(scale[nd%len(scale)],base_freq)
                note=piano_note(freq,chord_duration*0.9,sr)
                if pos+len(note)<=total_samples:chords_track[pos:pos+len(note)]+=note*vel*0.2
        elif section['name'] in ['chorus','drop','climax']:
            for nd in chord:
                freq=note_to_freq(scale[nd%len(scale)],base_freq)
                note_p=piano_note(freq,chord_duration*0.9,sr)
                if pos+len(note_p)<=total_samples:chords_track[pos:pos+len(note_p)]+=note_p*vel*0.15
                note_s=violin_note(freq,chord_duration*0.95,sr)
                if pos+len(note_s)<=total_samples:chords_track[pos:pos+len(note_s)]+=note_s*vel*0.12
        else:
            for nd in chord:
                freq=note_to_freq(scale[nd%len(scale)],base_freq)
                note=piano_note(freq,chord_duration*0.9,sr)
                if pos+len(note)<=total_samples:chords_track[pos:pos+len(note)]+=note*vel*0.18
    
    print("  🎶 Melodia...")
    note_duration=beat_duration/2;current_degree=0;prev_section=None
    for i in range(int(duration/note_duration)):
        time=i*note_duration;section=structure.get_section_at_time(time)
        section_name=section['name'];energy=structure.get_energy_at_time(time)
        if section!=prev_section:current_degree=np.random.randint(0,len(scale));prev_section=section
        if section_name in ['intro','breakdown','break']:
            if np.random.random()<0.3:
                step=np.random.choice([-1,0,1])
                current_degree=max(0,min(current_degree+step,len(scale)-1))
                freq=note_to_freq(scale[current_degree],base_freq)
                pos=int(time*sr);nlen=int(note_duration*sr*3)
                note=flute_note(freq,nlen/sr,sr)
                if pos+len(note)<=total_samples:melody_track[pos:pos+len(note)]+=note*energy*0.25
        elif section_name in ['verse','head','theme']:
            if np.random.random()<0.7:
                step=np.random.choice([-2,-1,0,1,2],p=[0.1,0.25,0.3,0.25,0.1])
                current_degree=max(0,min(current_degree+step,len(scale)*2-1))
                octave=current_degree//len(scale);degree=current_degree%len(scale)
                freq=note_to_freq(scale[degree],base_freq)*(2**octave)
                pos=int(time*sr);nlen=int(note_duration*sr*1.5)
                note=piano_note(freq,nlen/sr,sr)
                if pos+len(note)<=total_samples:melody_track[pos:pos+len(note)]+=note*energy*0.3
        elif section_name in ['chorus','drop','climax']:
            if np.random.random()<0.85:
                step=np.random.choice([-3,-2,-1,0,1,2,3],p=[0.05,0.15,0.2,0.2,0.2,0.15,0.05])
                current_degree=max(0,min(current_degree+step,len(scale)*3-1))
                octave=current_degree//len(scale);degree=current_degree%len(scale)
                freq=note_to_freq(scale[degree],base_freq)*(2**octave)
                pos=int(time*sr)
                dur_mult=np.random.choice([0.5,1.0,1.5,2.0],p=[0.3,0.4,0.2,0.1])
                nlen=int(note_duration*sr*1.8*dur_mult)
                inst_choice=np.random.choice(['piano','violin','brass','synth'])
                if inst_choice=='piano':note=piano_note(freq,nlen/sr,sr)
                elif inst_choice=='violin':note=violin_note(freq,nlen/sr,sr)
                elif inst_choice=='brass':note=brass_note(freq,nlen/sr,sr)
                else:note=synth_pad(freq,nlen/sr,sr)
                if pos+len(note)<=total_samples:melody_track[pos:pos+len(note)]+=note*energy*0.35
        else:
            if np.random.random()<0.6:
                step=np.random.choice([-1,0,1,2])
                current_degree=max(0,min(current_degree+step,len(scale)*2-1))
                octave=current_degree//len(scale);degree=current_degree%len(scale)
                freq=note_to_freq(scale[degree],base_freq)*(2**octave)
                pos=int(time*sr);nlen=int(note_duration*sr*1.5)
                note=piano_note(freq,nlen/sr,sr)
                if pos+len(note)<=total_samples:melody_track[pos:pos+len(note)]+=note*energy*0.3
    
    print("  ✨ Efeitos...")
    for section in structure.section_times:
        if section['name']=='buildup':
            riser_start=int((section['end']-2.0)*sr)
            if riser_start>0 and riser_start+int(2.0*sr)<=total_samples:
                riser=riser_sweep(2.0,sr)
                fx_track[riser_start:riser_start+len(riser)]+=riser*0.15
    for i in range(len(structure.section_times)-1):
        transition_time=structure.section_times[i]['end']
        sweep_start=int((transition_time-0.5)*sr)
        if sweep_start>0 and sweep_start+int(0.5*sr)<=total_samples:
            sweep=noise_sweep(0.5,sr)
            fx_track[sweep_start:sweep_start+len(sweep)]+=sweep*0.1
    
    print("  🎛️ Mixagem...")
    mix=drums_track+bass_track+chords_track+melody_track+fx_track
    mix=add_reverb(mix,sr,decay=0.35,mix=np.random.uniform(0.15,0.3))
    if np.random.random()<0.3:mix=chorus_effect(mix,sr,mix=0.2)
    if np.random.random()<0.2:mix=flanger_effect(mix,sr,mix=0.15)
    mix=soft_compress(mix,threshold=0.5,ratio=3.0)
    mix=limiter(mix,threshold=0.95)
    mix=mix/(np.max(np.abs(mix))+1e-10)*0.9
    fade_in=int(0.5*sr);fade_out=int(1.5*sr)
    if fade_in<len(mix):mix[:fade_in]*=np.linspace(0,1,fade_in)
    if fade_out<len(mix):mix[-fade_out:]*=np.linspace(1,0,fade_out)
    return mix,sr

def generate_breakcore(duration,sr=44100,intensity=1.0):
    seed=get_dynamic_seed();np.random.seed(seed)
    total_samples=int(duration*sr);bpm=np.random.randint(180,230)
    print(f"  💥 BREAKCORE: BPM={bpm}")
    break_source=make_amen_break(sr,tempo_factor=bpm/180)
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
        if pos<total_samples and len(chunk)>0:drums_track[pos:end]+=chunk[:end-pos]*np.random.uniform(0.5,1.0)*intensity*0.8
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

def generate_music(prompt=None,style=None,duration=45,use_rag=True):
    print("="*60)
    print("🎵 IA MUSIC GENERATOR PRO - UPGRADE COMPLETO")
    print("="*60)
    interpreter=PromptInterpreter()
    prompt_params=interpreter.interpret(prompt)
    final_style=style
    if prompt_params.get('style'):final_style=prompt_params['style']
    if not final_style:final_style='pop'
    
    # Verificar se é breakcore
    if prompt_params.get('breakcore',0)>1.5 or final_style=='breakcore':
        print("💥 Modo BREAKCORE")
        return generate_breakcore(duration)
    
    rag=MusicRAG() if use_rag else None
    if use_rag and rag:rag_context=rag.get_context(prompt,style_hint=final_style)
    else:rag_context=rag._random_context() if rag else {'scale':[0,2,4,5,7,9,11],'scale_name':'major','progression':ALL_PROGRESSIONS[0],'bpm':120,'intensity':0.7,'style':final_style,'seed':get_dynamic_seed()}
    
    print(f"\n🎵 Gerando música final...")
    return generate_with_structure(duration,rag_context)

# ============================================================
# BATCH E MAIN
# ============================================================

def batch_generate():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--prompt",type=str,default="")
    parser.add_argument("--style",type=str,default="epic")
    parser.add_argument("--duration",type=int,default=45)
    parser.add_argument("--use-rag",type=str,default="true")
    parser.add_argument("--remix",type=str,default="")
    parser.add_argument("--remix-type",type=str,default="variations")
    parser.add_argument("--batch",action="store_true")
    args=parser.parse_args()
    
    use_rag=args.use_rag.lower()=="true"
    
    # Modo remix
    if args.remix:
        print("="*60)
        print("🎛️ MODO REMIX")
        print("="*60)
        result=generate_remix(int(args.remix),args.remix_type,args.style)
        if result:print(f"\n✅ Remix completo!")
        return
    
    prompt=args.prompt if args.prompt else None
    style=args.style if not prompt else None
    audio,sr=generate_music(prompt=prompt,style=style,duration=args.duration,use_rag=use_rag)
    metadata={"prompt":args.prompt,"style":args.style,"duration":args.duration,"use_rag":use_rag,"version":"upgrade_completo"}
    filepath,number=save_song(audio,sr,metadata)
    print(f"\n✅ Música #{number}: {filepath}")

def main():
    while True:
        print("="*60)
        print("🎵 IA MUSIC PRO - UPGRADE COMPLETO + REMIX")
        print("="*60)
        os.makedirs(OUTPUT_DIR,exist_ok=True)
        songs=[f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
        print(f"📁 Músicas geradas: {len(songs)}\n")
        print("1. 🎵 Gerar com PROMPT")
        print("2. 🎼 Gerar com ESTILO")
        print("3. 💥 Gerar BREAKCORE")
        print("4. 🎛️ REMIX (nova função!)")
        print("5. 📁 Ver músicas")
        print("6. ❌ Sair")
        choice=input("\nOpção: ").strip()
        
        if choice=='1':
            prompt=input("Prompt: ").strip()
            duration=input("Duração (30/45/60/90): ").strip()
            if duration not in ["30","45","60","90"]:duration="45"
            audio,sr=generate_music(prompt=prompt,duration=int(duration))
            fp,num=save_song(audio,sr,{"prompt":prompt,"duration":int(duration)})
            print(f"\n✅ Música #{num}: {fp}")
            input("ENTER...")
        elif choice=='2':
            print("\nEstilos disponíveis:")
            styles_list=list(MUSIC_STYLES.keys())
            for i in range(0,len(styles_list),4):
                print("  "+" | ".join(styles_list[i:i+4]))
            s=input("\nEstilo: ").strip().lower()
            duration=input("Duração (30/45/60/90): ").strip()
            if duration not in ["30","45","60","90"]:duration="45"
            audio,sr=generate_music(style=s,duration=int(duration))
            fp,num=save_song(audio,sr,{"style":s,"duration":int(duration)})
            print(f"\n✅ Música #{num}: {fp}")
            input("ENTER...")
        elif choice=='3':
            duration=input("Duração (30/45/60/90): ").strip()
            if duration not in ["30","45","60","90"]:duration="45"
            audio,sr=generate_breakcore(int(duration))
            fp,num=save_song(audio,sr,{"style":"breakcore","duration":int(duration)})
            print(f"\n✅ Breakcore #{num}: {fp}")
            input("ENTER...")
        elif choice=='4':
            print("\n🎛️ SISTEMA DE REMIX")
            print("Músicas disponíveis:")
            songs=sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')])
            for s in songs:print(f"  {s}")
            if not songs:
                print("Nenhuma música para remixar.")
                input("ENTER...")
                continue
            num=input("\nNúmero da música para remixar: ").strip()
            print("\nTipo de remix:")
            print("  1. variations (5 variações aleatórias)")
            print("  2. style_change (mudar estilo)")
            print("  3. speed_up (acelerar)")
            print("  4. slow_down (desacelerar)")
            print("  5. acoustic (versão acústica)")
            print("  6. electronic (versão eletrônica)")
            print("  7. heavy (versão pesada)")
            print("  8. soft (versão suave)")
            remix_type=input("\nTipo (1-8): ").strip()
            remix_map={"1":"variations","2":"style_change","3":"speed_up","4":"slow_down","5":"acoustic","6":"electronic","7":"heavy","8":"soft"}
            remix_type=remix_map.get(remix_type,"variations")
            target_style=None
            if remix_type=="style_change":
                target_style=input("Novo estilo: ").strip().lower()
            result=generate_remix(int(num),remix_type,target_style)
            if result:print(f"\n✅ Remix completo!")
            input("ENTER...")
        elif choice=='5':
            songs=sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')])
            if not songs:print("Nenhuma música.")
            else:
                print(f"\n📁 Total: {len(songs)} músicas")
                for s in songs:
                    size=os.path.getsize(os.path.join(OUTPUT_DIR,s))/1024/1024
                    print(f"  🎵 {s} ({size:.1f}MB)")
            input("\nENTER...")
        elif choice=='6':
            print("\n👋 Até a próxima! 🎵")
            break

if __name__=="__main__":
    if "--batch" in sys.argv:batch_generate()
    else:main()
