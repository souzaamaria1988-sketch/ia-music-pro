# IA Music Pro

Gerador de música com IA em português.

## SoundFont

A pasta soundfonts/ é mantida no Git por soundfonts/.gitkeep.

Arquivos .sf2 e .sf3 são ignorados pelo .gitignore.

Não use SoundFont sem licença verificada.

## Uso

python music_generator.py --prompt "samba com cavaquinho, pandeiro e surdo" --duration 10

## Fallback

Se FluidSynth ou SoundFont não estiverem disponíveis, o sistema usa síntese procedural.
