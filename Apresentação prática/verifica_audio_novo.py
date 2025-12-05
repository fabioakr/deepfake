"""
Este script será utilizado durante a apresentação prática do TCC, 
verificando se os áudios real e fake gerados ao vivo, anteriormente,
são detectados corretamente pelos modelos treinados, a partir dos 
arquivos de extensão .pkl salvos. Ao fim, ele gera uma tabela com 
as probabilidades de cada áudio ser fake, segundo cada modelo.

Alunos:
Fábio Akira Yonamine - 11805398
Maria Monique de Menezes Cavalcanti - 11807935

Como os modelos foram treinados utilizando scikit-learn na versão 1.6.1,
sugerimos criar um ambiente virtual (.venv), utilizando obrigatoriamente
essa versão da biblioteca. Caso outra versão mais nova do scikit-learn 
seja usada, o Terminal emite um warning, dizendo que os resultados podem
não ser conclusivos. A versão do Python utilizada nesse venv é 3.13.7.

COMANDOS PARA CRIAR E ATIVAR .VENV NO TERMINAL (MAC OS):
python3.13 -m venv .venv
source .venv/bin/activate

COMANDOS PARA INSTALAR A DEPENDÊNCIA NO .VENV:
pip install scikit-learn==1.6.1

COMANDOS PARA RODAR O SCRIPT:
python3 verifica_audio_novo.py
"""

import os
import sys
import numpy as np
import librosa
import joblib
import soundfile as sf
import scipy
import matplotlib.pyplot as plt

# --- Configurações (mesmas do treino) ---
N_MFCC = 40
TARGET_SR = 16000

MODEL_FOLDERS = [
    "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/KNN/knn_results_mfcc",
    "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/KNN/knn_results_cqcc",
    "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/KNN/knn_results_lfcc",
    "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Regressão Logística/logistic_regression_results_mfcc40",
    "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Regressão Logística/logistic_regression_results_cqcc40",
    "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Regressão Logística/logistic_regression_results_lfcc40"
]

wav_files = [
    "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/audio_original.wav",
    "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/audio_fake.wav"
    ]

#input_path = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake"


def extract_mfcc(filepath, n_mfcc=N_MFCC):
    """
    Extrai a média e o desvio padrão dos MFCCs.
    """
    y, sr = librosa.load(filepath, sr=TARGET_SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat_mean = np.mean(mfcc, axis=1)
    feat_std = np.std(mfcc, axis=1)
    return np.concatenate([feat_mean, feat_std])

def extract_cqcc(path, sr=TARGET_SR, n_cqcc=N_MFCC, bins_per_octave=96):
    """
    CQCC = DCT(log(CQT^2))
    CQT → log-power → DCT → coeficientes
    """
    #wave = load_audio(path)
    wave = load_audio2(path)

    # 1) CQT complex
    CQT = librosa.cqt(
        wave,
        sr=sr,
        hop_length=128,
        fmin=20,
        n_bins=8 * bins_per_octave,
        bins_per_octave=bins_per_octave,
        pad_mode="reflect"     # evita janelas curtas
    )

    # 2) Espectro de potência
    power = np.abs(CQT)**2

    # 3) Log-power
    log_power = np.log(power + 1e-12)

    # 4) DCT → cepstrum
    cqcc = scipy.fft.dct(log_power, axis=0, norm='ortho')

    # 5) Mantém primeiros coeficientes
    cqcc = cqcc[:n_cqcc, :]

    feat_mean = np.mean(cqcc, axis=1)
    feat_std = np.std(cqcc, axis=1)

    return np.concatenate([feat_mean, feat_std])

def load_audio2(path, sr=TARGET_SR):
    y, _ = librosa.load(path, sr=sr)
    return y

def load_audio(path, sr=TARGET_SR):
    wav, fs = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if fs != sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=fs, target_sr=sr)
    return wav.astype(np.float32)

def linear_filter_banks(sr, n_fft, n_filters, fmin=0, fmax=None):
    if fmax is None:
        fmax = sr / 2

    # Frequências reais geradas pela FFT
    freqs = np.linspace(0, sr / 2, 1 + n_fft // 2)

    # Bordas das bandas lineares
    edges = np.linspace(fmin, fmax, n_filters + 2)

    fbanks = np.zeros((n_filters, len(freqs)))

    for i in range(n_filters):
        left = edges[i]
        center = edges[i + 1]
        right = edges[i + 2]

        # Subida triangular
        left_slope = (freqs - left) / (center - left)
        # Descida triangular
        right_slope = (right - freqs) / (right - center)

        fbanks[i] = np.maximum(0, np.minimum(left_slope, right_slope))

    return fbanks

def extract_lfcc(path, sr=TARGET_SR, n_lfcc=N_MFCC): ### FUSAO DE EXTRACT_LFCC E EXTRACT_LFCC_MEAN
    wave = load_audio(path)

    # STFT → power spectrum
    S = np.abs(librosa.stft(wave, n_fft=512, hop_length=160, win_length=400))**2

    # Filtros triangulares lineares
    fbanks = linear_filter_banks(sr=sr, n_fft=512, n_filters=n_lfcc)

    # Aplica os filtros → espectro filtrado
    filtered = np.dot(fbanks, S)

    # log-energy
    logS = np.log(filtered + 1e-10)

    # Cepstrum via DCT
    lfcc = scipy.fft.dct(logS, axis=0, norm="ortho")

    feat_mean = np.mean(lfcc, axis=1)
    feat_std = np.std(lfcc, axis=1)

    return np.concatenate([feat_mean, feat_std])

FEATURE_EXTRACTORS = {
    "knn_results_mfcc": extract_mfcc,
    "knn_results_cqcc": extract_cqcc,
    "knn_results_lfcc": extract_lfcc,
    "logistic_regression_results_mfcc40": extract_mfcc,
    "logistic_regression_results_cqcc40": extract_cqcc,
    "logistic_regression_results_lfcc40": extract_lfcc
}

MODEL_LABELS = {
    "knn_results_mfcc": "KNN (MFCC)",
    "knn_results_cqcc": "KNN (CQCC)",
    "knn_results_lfcc": "KNN (LFCC)",
    "logistic_regression_results_mfcc40": "LogReg (MFCC)",
    "logistic_regression_results_cqcc40": "LogReg (CQCC)",
    "logistic_regression_results_lfcc40": "LogReg (LFCC)"
}

# ============================================================
# Funções de inferência
# ============================================================

def load_model_and_scalers():
    """Carrega múltiplos modelos KNN e scalers de cada subpasta."""
    models = []
    for folder in MODEL_FOLDERS:
        folder_name = os.path.basename(folder)

        if folder_name.startswith("knn_"):
            model_file = "knn_model.pkl"
            scaler_file = "scaler_knn.pkl"
        elif folder_name.startswith("lr_") or folder_name.startswith("logistic_regression_"):
            model_file = "logreg_model.pkl"
            scaler_file = "scaler_logreg.pkl"
        else:
            raise ValueError(f"Não sei qual modelo usar para a pasta: {folder_name}")

        model_path = os.path.join(folder, model_file)
        scaler_path = os.path.join(folder, scaler_file)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler não encontrado em: {scaler_path}")

        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        models.append((folder, clf, scaler))
    return models

def classify_file(filepath, clf, scaler, feature_fn):
    """
    Classifica um único arquivo .wav como Real (0) ou Fake (1).

    Retorna:
        label (int): 0 ou 1
        prob_fake (float): probabilidade de ser Fake (classe 1)
    """
    #features = extract_mfcc(filepath)   # MESMA feature do treino
    features = feature_fn(filepath)   # MESMA feature do treino
    X = features.reshape(1, -1)         # 1 amostra, n_features
    X_scaled = scaler.transform(X)

    prob = clf.predict_proba(X_scaled)[0]   # [prob_real, prob_fake]
    y_pred = clf.predict(X_scaled)[0]       # 0 ou 1

    prob_fake = float(prob[1])
    return int(y_pred), prob_fake

def collect_wav_files(path):
    """Se path for pasta, percorre recursivamente; se for arquivo, retorna só ele."""
    if os.path.isfile(path):
        if path.lower().endswith(".wav"):
            return [path]
        else:
            raise ValueError("O caminho fornecido é um arquivo, mas não é .wav.")
    elif os.path.isdir(path):
        wav_files = []
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                if f.lower().endswith(".wav"):
                    wav_files.append(os.path.join(dirpath, f))
        return sorted(wav_files)
    else:
        raise FileNotFoundError(f"Caminho não encontrado: {path}")

def main():
    # Carrega modelos e scalers
    models = load_model_and_scalers()
    results = {}

    if not wav_files:
        print("Nenhum arquivo .wav encontrado.")
        sys.exit(1)

    print(f"\nEncontrados {len(wav_files)} arquivo(s) .wav.\n")
    for wav_path in wav_files:
        print(f"\nArquivo: {wav_path}")
        results[wav_path] = {}
        for folder, clf, scaler in models:
            try:
                model_name = os.path.basename(folder)
                feature_fn = FEATURE_EXTRACTORS[model_name]
                label, prob_fake = classify_file(wav_path, clf, scaler, feature_fn)
                label_str = "Real" if label == 0 else "Fake"
                print(f"  Modelo: {folder}")
                print(f"    → Predição: {label_str} (classe {label})")
                print(f"    → Probabilidade de ser Fake: {prob_fake:.3f}")
                results[wav_path][model_name] = prob_fake
            except Exception as e:
                print(f"  ⚠️ Erro no modelo '{folder}': {e}")
        print("-" * 60)

    for wav_path, model_probs in results.items():
        models_list = [MODEL_LABELS[m] for m in model_probs.keys()]
        probs = list(model_probs.values())
        plt.figure(figsize=(10,4))
        plt.bar(models_list, probs)
        plt.title(f"Probabilidade de ser Fake - {os.path.basename(wav_path)}")
        plt.ylabel("Probabilidade")
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # --- Gerar uma única imagem com todas as tabelas ---
    fig, axes = plt.subplots(len(results), 1, figsize=(8, 2.5 * len(results)))
    if len(results) == 1:
        axes = [axes]

    for ax, (wav_path, model_probs) in zip(axes, results.items()):
        ax.axis('off')
        models_list = [MODEL_LABELS[m] for m in model_probs.keys()]
        probs = list(model_probs.values())
        table_data = [["Modelo", "Probabilidade Fake"]] + list(zip(models_list, [f"{p*100:.1f}%" for p in probs]))
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)
        for i, prob in enumerate(probs, start=1):
            if prob >= 0.5:
                table[(i, 1)].set_facecolor('#FF7F7F')
            else:
                table[(i, 1)].set_facecolor('#90EE90')
        ax.set_title(f"Teste - {os.path.basename(wav_path)}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
