"""
Para usar este script, como ele é de uma versão 

https://github.com/coqui-ai/TTS/tree/dev#installation
"""

# A biblioteca TTS.api e o sounddevice necessitam de Python 3.10, então por isso criei esse .venv nessa versão.

# Primeiro: ativar o .venv com source .venv/bin/activate
# Deve estar (.venv) antes do nome do terminal somente. Se estiver (base), rode conda deactivate.

# Vamos ver se funciona primeiro:
# pip install torch==2.2.1 torchaudio==2.2.1
# pip install torchcodec==0.1.0
# pip install TTS==0.22.0

#Para rodar: python3.10 -u "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/xtts_audio_fake.py";        

import os
import torch
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from TTS.api import TTS

# ==========================================
# CORREÇÃO PARA PYTORCH 2.6+ (Weights Only)
# ==========================================
try:
    if hasattr(torch, 'load'):
        _original_load = torch.load

        def _safe_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)

        torch.load = _safe_load
except:
    pass

os.environ["COQUI_TOS_AGREED"] = "1"

AUDIO_REF = "audio_original.wav"
AUDIO_FAKE = "audio_fake.wav"
FS = 16000


def gravar_audio(duracao=12):
    print(f"\n--- GRAVAÇÃO ({duracao}s) ---")
    print("Fale de maneira natural, como se estivesse conversando.")

    audio = sd.rec(int(duracao * FS), samplerate=FS, channels=1, dtype="float32")
    sd.wait()

    audio_int16 = (audio * 32767).astype(np.int16)
    write(AUDIO_REF, FS, audio_int16)

    print(f"Áudio salvo em: {os.path.abspath(AUDIO_REF)}")


def gerar_fake(texto):
    print("\n--- GERANDO FAKE ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")

    MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

    tts = TTS(MODEL).to(device)

    print("Gerando com voz clonada (isso pode levar alguns segundos)...")

    tts.tts_to_file(
        text=texto,
        file_path=AUDIO_FAKE,
        speaker_wav=AUDIO_REF,
        language="pt"
    )

    print(f"\nFAKE GERADO EM: {os.path.abspath(AUDIO_FAKE)}")


TEXTO = (
    "Oi! Estou fazendo um teste da minha voz clonada. "
    "Quero ver se agora ela fica mais natural, com um ritmo parecido com o meu jeito de falar."
)

if __name__ == "__main__":
    gravar_audio(duracao=12)
    gerar_fake(TEXTO)
