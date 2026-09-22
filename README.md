# IA Music Pro

Gerador de música com Inteligência Artificial em português.

## Características

- **Interpretação de prompts em português**: "samba com cavaquinho, pandeiro e surdo"
- **SoundFont realista**: Renderização MIDI com instrumentos reais via FluidSynth
- **Fallback sintético**: Funciona sem SoundFont usando síntese procedural
- **48+ instrumentos**: Piano, cordas, metais, instrumentos brasileiros
- **Composição MIDI**: Gera arquivos MIDI padrão
- **Interface web**: Upload e gestão de arquivos via navegador

## Instalação

### Python

```bash
pip install -r requirements.txt
```

### FluidSynth (opcional)

**Linux:**
```bash
sudo apt-get install fluidsynth
```

**macOS:**
```bash
brew install fluidsynth
```

### SoundFont

1. Baixe um SoundFont com licença compatível (CC0, CC-BY)
2. Coloque em `soundfonts/default.sf2`
3. Registre a licença em `soundfont_config.json`

## Uso

```bash
python music_generator.py --prompt "música cinematográfica com piano e violino"
python music_generator.py --prompt "samba com cavaquinho, pandeiro e surdo" --duration 10
python music_generator.py --soundfont soundfonts/default.sf2 --midi-output
```

### Listar instrumentos

```bash
python music_generator.py --list-instruments
python music_generator.py --list-families
```

## Fallback Sintético

Se o SoundFont não estiver disponível, o sistema automaticamente usa síntese procedural.

## Licenças

- **Código**: Licença do projeto
- **SoundFonts**: Verificar licença individual de cada .sf2
- **Samples**: Verificar licença individual

**AVISO**: Não inclua arquivos de áudio protegidos por direitos autorais sem permissão.

## Limitações

1. A qualidade do som depende do SoundFont utilizado
2. Instrumentos brasileiros geralmente não existem em SoundFonts padrão
3. A IA não "compõe" de forma criativa, usa regras e aleatoriedade controlada
4. O treinamento com FFT aprende padrões espectrais, não teoria musical
