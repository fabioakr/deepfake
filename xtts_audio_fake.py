"""
Este script coleta um áudio de referência da sua voz, utilizando a biblioteca sounddevice,
e então gera um áudio falso (deepfake) utilizando a biblioteca TTS com o modelo XTTS v2. 
O áudio falso é salvo com o discurso presente na variável TEXTO, com a voz do locutor inicial.

Alunos:
Fábio Akira Yonamine - 11805398
Maria Monique de Menezes Cavalcanti - 11807935

Como o modelo XTTS v2 necessita de Python 3.10, sugerimos criar um ambiente virtual (.venv), 
utilizando obrigatoriamente as bibliotecas nas versões indicadas abaixo.

COMANDOS PARA CRIAR E ATIVAR .VENV NO TERMINAL (MAC OS):
python3.10 -m venv .venv
source .venv/bin/activate

COMANDOS PARA INSTALAR AS DEPENDÊNCIAS NO .VENV:
pip install torch==2.2.1 torchaudio==2.2.1
pip install torchcodec==0.1.0
pip install TTS==0.22.0
pip install sounddevice
pip install "transformers==4.36.2"
python3.10 -u "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/xtts_audio_fake.py"

COMANDOS PARA DESATIVAR O VENV:
deactivate

COMANDOS PARA REMOVER O VENV, DEPOIS DE DESATIVADO:
rm -rf .venv

Para mais informações sobre a biblioteca TTS, acesse:
https://github.com/coqui-ai/TTS/tree/dev#installation
"""

###  Os avisos são suprimidos para evitar poluição no output do programa.  ###
###  Eles surgem por estarmos utilizando comandos em estado "deprecated",  ###
###  mas que ainda funcionam para nossos propósitos.                       ###
import warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

###  Importação de bibliotecas  ###
import os
import torch
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from TTS.api import TTS

###  Abaixo, constam os nomes dos arquivos de áudio gerados           ###
###  (original e fake). Caso deseje, mude os nomes entre aspas.       ###
###  FS altera a frequência de amostragem dos dois arquivos, mas      ###
###  já estão ajustados para 16 [kHz], padrão usado neste trabalho.   ###
###  Por fim, a variável TEXTO é o discurso a ser proferido no        ###
###  áudio fake.                                                      ##############################
AUDIO_REF = "audio_original.wav"
AUDIO_FAKE = "audio_fake.wav"
FS = 16000
TEXTO = (
    "Oi! Estou fazendo um teste da minha voz clonada."
    "Quero ver se agora ela fica mais natural, com um ritmo parecido com o meu jeito de falar."
)
####################################################################################################


# ==========================================
# CORREÇÃO PARA PYTORCH 2.6+ (Weights Only)
# ==========================================
"""
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
"""

# Aceita os termos de uso da Coqui TTS, mandatórios para utilização da biblioteca
os.environ["COQUI_TOS_AGREED"] = "1"

def gravar_audio(duracao=12):
    """
    Grava um áudio de 12 segundos da sua voz, utilizando a biblioteca sounddevice.
    O áudio é salvo no arquivo definido na variável AUDIO_REF, na mesma pasta.
    """

    print(f"\n--- GRAVAÇÃO ({duracao}s) ---")
    print("Fale de maneira natural, como se estivesse conversando.")

    audio = sd.rec(int(duracao * FS), samplerate=FS, channels=1, dtype="float32")
    sd.wait()

    audio_int16 = (audio * 32767).astype(np.int16)
    write(AUDIO_REF, FS, audio_int16)

    print(f"Áudio salvo em: {os.path.abspath(AUDIO_REF)}")


def gerar_fake(texto):
    """
    Lê o arquivo de áudio de referência (AUDIO_REF), também recebe o discurso 
    na variável e gera um áudio falso (AUDIO_FAKE) com o modelo do XTTS v2.
    """

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

if __name__ == "__main__":
    gravar_audio(duracao=12)
    gerar_fake(TEXTO)
