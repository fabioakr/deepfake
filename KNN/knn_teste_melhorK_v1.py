"""
Versão com seleção automática do melhor K para KNN.
Testa vários valores de K e escolhe aquele com melhor acurácia.
"""

import os
import time
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import scipy.fft
import soundfile as sf

# --- Configurações ---
folder_train_true = "/Users/fabioakira/Downloads/reais_train"
folder_train_fake = "/Users/fabioakira/Downloads/fakes_train"
folder_test_true = "/Users/fabioakira/Downloads/reais_test"
folder_test_fake = "/Users/fabioakira/Downloads/fakes_test"

SAVE_FOLDER = "knn_results"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Parâmetros KNN
LISTA_K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30]

# Parâmetros Áudio
N_MFCC = 40
TARGET_SR = 16000

def _process_folder(root_folder, label):
    features_list, labels_list = [], []

    if isinstance(root_folder, list):
        file_list = root_folder
    else:
        file_list = []
        for dirpath, _, filenames in os.walk(root_folder):
            for f in filenames:
                if f.lower().endswith(".wav"):
                    file_list.append(os.path.join(dirpath, f))

    for filepath in file_list:
        try:
            features = extract_mfcc(filepath)  # pode trocar para LFCC se quiser
            features_list.append(features)
            labels_list.append(label)
        except Exception as e:
            print(f"⚠️ Erro no arquivo {filepath}: {e}")

    return features_list, labels_list

def _process_folder_with_extractor(root_folder, label, extractor_fn):
    features_list, labels_list = [], []

    if isinstance(root_folder, list):
        file_list = root_folder
    else:
        file_list = []
        for dirpath, _, filenames in os.walk(root_folder):
            for f in filenames:
                if f.lower().endswith(".wav"):
                    file_list.append(os.path.join(dirpath, f))

    for filepath in file_list:
        try:
            features = extractor_fn(filepath)
            features_list.append(features)
            labels_list.append(label)
        except Exception as e:
            print(f"⚠️ Erro no arquivo {filepath}: {e}")

    return features_list, labels_list

def extract_mfcc(filepath, n_mfcc=N_MFCC):
    y, sr = librosa.load(filepath, sr=TARGET_SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat_mean = np.mean(mfcc, axis=1)
    feat_std = np.std(mfcc, axis=1)
    return np.concatenate([feat_mean, feat_std])

def load_audio(path, sr=TARGET_SR):
    wav, fs = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if fs != sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=fs, target_sr=sr)
    return wav.astype(np.float32)

# ---------------------------------------
# 2. Função para extrair LFCC completo
# ---------------------------------------

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

def load_manual_dataset(real_train, fake_train, real_test, fake_test, extractor_fn):
    print("Carregando treino...")
    X_rt, y_rt = _process_folder_with_extractor(real_train, 0, extractor_fn)
    X_ft, y_ft = _process_folder_with_extractor(fake_train, 1, extractor_fn)

    def collect_files(folder):
        return [os.path.join(dp, f) for dp, _, files in os.walk(folder)
                for f in files if f.lower().endswith(".wav")]

    real_files = collect_files(real_test)
    fake_files = collect_files(fake_test)

    real_train_part, real_test_part = train_test_split(real_files, test_size=0.2, random_state=42)
    fake_train_part, fake_test_part = train_test_split(fake_files, test_size=0.2, random_state=42)

    X_rv, y_rv = _process_folder_with_extractor(real_test_part, 0, extractor_fn)
    X_fv, y_fv = _process_folder_with_extractor(fake_test_part, 1, extractor_fn)

    X_train = np.array(X_rt + X_ft)
    y_train = np.array(y_rt + y_ft)
    X_test = np.array(X_rv + X_fv)
    y_test = np.array(y_rv + y_fv)

    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    return X_train, y_train, X_test, y_test


def train_and_select_best_k(extractor_fn, label):
    start = time.time()
    X_train, y_train, X_test, y_test = load_manual_dataset(
        folder_train_true, folder_train_fake,
        folder_test_true, folder_test_fake,
        extractor_fn
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    resultados = []
    melhor_k = None
    melhor_acc = -1

    print(f"\n🔍 Testando valores de K para {label}:")
    for k in LISTA_K:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train_scaled, y_train)

        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)

        resultados.append((k, acc))
        print(f"K={k} → Accuracy = {acc:.3f}")

        if acc > melhor_acc:  # escolhe melhor K
            melhor_acc = acc
            melhor_k = k

    print(f"\n🏆 Melhor K encontrado para {label}:", melhor_k)

    # Treina versão final com o melhor K
    clf = KNeighborsClassifier(n_neighbors=melhor_k)
    clf.fit(X_train_scaled, y_train)

    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== RESULTADOS FINAIS para {label} ===")
    print(f"K ótimo: {melhor_k}")
    print(f"Accuracy: {melhor_acc:.3f}")
    print(f"AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    joblib.dump(clf, os.path.join(SAVE_FOLDER, f"knn_model_{label.lower()}.pkl"))
    joblib.dump(scaler, os.path.join(SAVE_FOLDER, f"scaler_knn_{label.lower()}.pkl"))
    print("💾 Modelo salvo.")

    ks, accs = zip(*resultados)
    print("\n⏱️ Tempo total:", round(time.time() - start, 2), "seg")
    return ks, accs, melhor_k, melhor_acc


if __name__ == "__main__":
    ks1, accs1, kbest_mfcc, amfcc = train_and_select_best_k(extract_mfcc, "MFCC")
    ks2, accs2, kbest_lfcc, alfcc = train_and_select_best_k(extract_lfcc, "LFCC")

    plt.figure(figsize=(7,5))
    plt.plot(ks1, accs1, marker='o', label='MFCC')
    plt.plot(ks2, accs2, marker='o', label='LFCC')
    plt.title("Comparação MFCC vs LFCC por valor de K")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_FOLDER, "comparacao_mfcc_lfcc.png"), dpi=300)
    plt.show()
