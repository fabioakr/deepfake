"""
Script de inferência para o modelo KNN treinado em knn_v3.py.

Usa os arquivos:
- knn_results/knn_model.pkl
- knn_results/scaler_knn.pkl

Entrada:
    - Caminho para 1 arquivo .wav
    - OU caminho para uma pasta contendo vários .wav (procura recursivamente)

Exemplos de uso:

    python knn_predict.py /caminho/para/arquivo.wav
    python knn_predict.py /caminho/para/pasta_com_audios


pip install scikit-learn==1.6.1
Precisa ser essa versão!!!!
"""

import os
import sys
import numpy as np
import librosa
import joblib
import soundfile as sf

# --- Configurações (mesmas do treino) ---
N_MFCC = 40
TARGET_SR = 16000

MODEL_FOLDERS = [
    "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/KNN/knn_results_cqcc",
    "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/KNN/knn_results_lfcc",
    "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/KNN/knn_results_mfcc"
]

wav_files = [
    "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/audio_original.wav",
    "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/audio_fake.wav"
    ]

input_path = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake"


def extract_mfcc(filepath, n_mfcc=N_MFCC):
    """
    Extrai a média e o desvio padrão dos MFCCs (mesma lógica do knn_v3.py).
    """
    y, sr = librosa.load(filepath, sr=TARGET_SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat_mean = np.mean(mfcc, axis=1)
    feat_std = np.std(mfcc, axis=1)
    return np.concatenate([feat_mean, feat_std])


# (Opcional) Se quiser usar LFCC no futuro, pode reaproveitar do knn_v3.py:
def load_audio(path, sr=TARGET_SR):
    wav, fs = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if fs != sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=fs, target_sr=sr)
    return wav.astype(np.float32)

# ============================================================
# Funções de inferência
# ============================================================

def load_model_and_scalers():
    """Carrega múltiplos modelos KNN e scalers de cada subpasta."""
    models = []
    for folder in MODEL_FOLDERS:
        model_path = os.path.join(folder, "knn_model.pkl")
        scaler_path = os.path.join(folder, "scaler_knn.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler não encontrado em: {scaler_path}")

        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        models.append((folder, clf, scaler))
    return models

def classify_file(filepath, clf, scaler):
    """
    Classifica um único arquivo .wav como Real (0) ou Fake (1).

    Retorna:
        label (int): 0 ou 1
        prob_fake (float): probabilidade de ser Fake (classe 1)
    """
    features = extract_mfcc(filepath)   # MESMA feature do treino
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

    if not wav_files:
        print("Nenhum arquivo .wav encontrado.")
        sys.exit(1)

    print(f"\nEncontrados {len(wav_files)} arquivo(s) .wav.\n")
    for wav_path in wav_files:
        print(f"\nArquivo: {wav_path}")
        for folder, clf, scaler in models:
            try:
                label, prob_fake = classify_file(wav_path, clf, scaler)
                label_str = "Real" if label == 0 else "Fake"
                print(f"  Modelo: {folder}")
                print(f"    → Predição: {label_str} (classe {label})")
                print(f"    → Probabilidade de ser Fake: {prob_fake:.3f}")
            except Exception as e:
                print(f"  ⚠️ Erro no modelo '{folder}': {e}")
        print("-" * 60)


if __name__ == "__main__":
    main()
