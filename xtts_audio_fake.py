"""
import os
import torch
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from TTS.api import TTS

# ==========================================
# CORREÇÃO PARA PYTORCH 2.6+ (Weights Only)
# ==========================================
# Isso resolve o erro: "_pickle.UnpicklingError: Weights only load failed"
# sem precisar fazer downgrade do PyTorch.
try:
    if hasattr(torch, 'load'):
        _original_load = torch.load


        def _safe_load(*args, **kwargs):
            # Se a versão do torch suportar 'weights_only', forçamos False
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)


        torch.load = _safe_load
except Exception as e:
    print(f"Aviso: Não foi possível aplicar o patch do PyTorch: {e}")
# ==========================================

# Aceitar termos de uso do Coqui automaticamente (evita travar o terminal)
os.environ["COQUI_TOS_AGREED"] = "1"

# Configurações
AUDIO_REF = "audio_original_MATHEUS_1.wav"
AUDIO_FAKE = "audio_fake_xtts_MATHEUS_1.wav"
FS = 16000  # Taxa de amostragem (16kHz é padrão para muitos modelos)


def gravar_audio(duracao=5):
    #Grava áudio do microfone e salva em WAV.
    print(f"\n--- GRAVAÇÃO ---")
    print(f"Gravando por {duracao} segundos... FALE AGORA.")

    # Grava
    audio = sd.rec(int(duracao * FS), samplerate=FS, channels=1, dtype="float32")
    sd.wait()  # Aguarda o fim da gravação

    print("Gravação finalizada.")

    # Converte float32 para int16 (formato padrão de WAV)
    audio_int16 = (audio * 32767).astype(np.int16)
    write(AUDIO_REF, FS, audio_int16)
    print(f"Áudio original salvo em: {os.path.abspath(AUDIO_REF)}")

def gerar_fake(texto):
    #Carrega o XTTS, extrai embedding e gera áudio clonado com maior naturalidade.
    print(f"\n--- GERAÇÃO DE FAKE ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")

    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

    try:
        print("Carregando modelo...")
        tts = TTS(MODEL_NAME).to(device)

        print("Extraindo embeddings...")
        speaker_emb = tts.get_speaker_embeddings(AUDIO_REF)

        print("Gerando áudio...")
        tts.tts_to_file(
            text=texto,
            file_path=AUDIO_FAKE,
            speaker_wav=AUDIO_REF,
            speaker_embeddings=speaker_emb,
            language="pt",
        )

        print(f"\nSUCESSO! Áudio fake gerado em: {os.path.abspath(AUDIO_FAKE)}")

    except Exception as e:
        print("\nERRO NA GERAÇÃO:")
        print(e)


if __name__ == "__main__":
    # 1. Grava a voz do usuário (Referência)
    gravar_audio(duracao=10)

    # 2. Define o texto que será falado com a voz clonada
    texto_alvo = "Olá! Este é um teste de clonagem de voz para o meu TCC na Poli. Minha voz foi clonada com sucesso."

    # 3. Gera o fake
    gerar_fake(texto_alvo)
    
"""
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
