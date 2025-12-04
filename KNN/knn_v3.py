"""
Esta versão utiliza KNN com 20% das amostras de teste, para se adequar ao CNN e SVM.

As funções auxiliares load_audio(), linear_filter_banks() e extract_lfcc() são exclusivas
para a extração de LFCC.
"""

import os
import time
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib

# Importações específicas do KNN
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split
import scipy.fft
import soundfile as sf

# --- Configurações ---
folder_train_true = "/Users/fabioakira/Downloads/reais_train"
folder_train_fake = "/Users/fabioakira/Downloads/fakes_train"
folder_test_true = "/Users/fabioakira/Downloads/reais_test"
folder_test_fake = "/Users/fabioakira/Downloads/fakes_test"

# Pasta para salvar resultados
SAVE_FOLDER = "knn_results"
#os.makedirs(SAVE_FOLDER, exist_ok=True)

# Parâmetros de ML (Ajuste para KNN)
N_VIZINHOS = 5    # Número de vizinhos (K) para o KNN

# Parâmetros de Áudio
N_MFCC = 40       # Número de coeficientes MFCC
TARGET_SR = 16000 # Sample rate alvo (16000 Hz)

def _process_folder(root_folder, label):
    """
    Função auxiliar para caminhar recursivamente por uma pasta,
    encontrar arquivos .wav e extrair features.
    """
    features_list = []
    labels_list = []

    if isinstance(root_folder, list):  # lista de arquivos já coletados
        file_list = root_folder
    else:  # caminho de pasta
        file_list = []
        for dirpath, _, filenames in os.walk(root_folder):
            for f in filenames:
                if f.lower().endswith(".wav"):
                    file_list.append(os.path.join(dirpath, f))

    for filepath in file_list:
        print(f"Abrindo arquivo: {filepath}")
        try:
            features = extract_mfcc(filepath) ## MUDE AQUI ENTRE LFCC E MFCC
            features_list.append(features)
            labels_list.append(label)
        except Exception as e:
            print(f"⚠️ Erro ao processar o arquivo {filepath}: {e}")

    return features_list, labels_list

def extract_mfcc(filepath, n_mfcc=N_MFCC):
    """
    Extrai a média e o desvio padrão dos MFCCs.
    """
    y, sr = librosa.load(filepath, sr=TARGET_SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat_mean = np.mean(mfcc, axis=1)
    feat_std = np.std(mfcc, axis=1)
    return np.concatenate([feat_mean, feat_std])

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


def load_manual_dataset(real_train, fake_train, real_test, fake_test):
    print("Carregando dados de treino...")
    X_rt, y_rt = _process_folder(real_train, 0)
    X_ft, y_ft = _process_folder(fake_train, 1)

    print("\nCarregando dados de teste... (apenas 20% sorteados)")

    # Coleta todos os arquivos das pastas de teste
    def collect_files(folder):
        paths = []
        for dirpath, _, filenames in os.walk(folder):
            for f in filenames:
                if f.lower().endswith(".wav"):
                    paths.append(os.path.join(dirpath, f))
        return paths

    real_files = collect_files(real_test)
    fake_files = collect_files(fake_test)

    # Sorteia apenas 20% dos arquivos
    real_train_part, real_test_part = train_test_split(real_files, test_size=0.2, random_state=42)
    fake_train_part, fake_test_part = train_test_split(fake_files, test_size=0.2, random_state=42)

    # Processa somente as amostras sorteadas
    X_rv, y_rv = _process_folder(real_test_part, 0)
    X_fv, y_fv = _process_folder(fake_test_part, 1)

    X_train = np.array(X_rt + X_ft)
    y_train = np.array(y_rt + y_ft)

    X_test = np.array(X_rv + X_fv)
    y_test = np.array(y_rv + y_fv)

    print(f"\nTreino: {len(X_train)} arquivos | Teste: {len(X_test)} arquivos")
    return X_train, y_train, X_test, y_test

def train_and_save_model():
    start_time = time.time()
    os.makedirs(SAVE_FOLDER, exist_ok=True) # Precisa estar aqui, para criar a pasta corretamente

    X_train, y_train, X_test, y_test = load_manual_dataset(
        folder_train_true, folder_train_fake,
        folder_test_true, folder_test_fake
    )

    if len(X_train) == 0 or len(X_test) == 0:
        print("\nERRO: Nenhum arquivo .wav foi encontrado nos conjuntos de treino ou teste!")
        return 

    # Normalização do KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinamento do KNN
    clf = KNeighborsClassifier(n_neighbors=N_VIZINHOS)
    clf.fit(X_train_scaled, y_train)

    # Roda a avaliação com amostras de teste
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    end_time = time.time()
    elapsed_time_sec = end_time - start_time

    # Tempo gasto para o script rodar
    minutes = int(elapsed_time_sec // 60)
    seconds = int(elapsed_time_sec % 60)
    time_display = f"{minutes} min {seconds} seg" if minutes > 0 else f"{elapsed_time_sec:.2f} seg"

    print("\n=== RESULTADOS DO TREINO (KNN) ===")
    print(f"Usando {N_MFCC} MFCCs e K={N_VIZINHOS}")
    print(f"Accuracy: {acc:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"Precision:  {precision:.3f}")
    print(f"Recall:     {recall:.3f}")
    print(f"F1-score:   {f1:.3f}")
    print(f"Tempo Total: {time_display}")
    print("Matriz de Confusão:\n", cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    # Gera o gráfico: matriz de confusão
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real","Fake"],
        yticklabels=["Real","Fake"])
    plt.title("Confusion Matrix - KNN, MFCC")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(SAVE_FOLDER, "matriz_confusao_knn.png"), dpi=300)
    print("💾 Matriz de confusão salva como matriz_confusao_knn.png")
    plt.show()

    # Gera o gráfico: curva ROC
    plt.figure(figsize=(6,5))
    # Set major ticks every 0.2 on both axes
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.plot([0,1], [0,1], "--", color="gray", label="Aleatório (AUC=0.5)")
    plt.title("Curva ROC - KNN, MFCC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(os.path.join(SAVE_FOLDER, "curva_roc_knn.png"), dpi=300)
    print("💾 Curva ROC salva como curva_roc_knn.png")
    plt.show()

    # Salva modelo + scaler em arquivos .pkl
    joblib.dump(clf, os.path.join(SAVE_FOLDER, "knn_model.pkl"))
    joblib.dump(scaler, os.path.join(SAVE_FOLDER, "scaler_knn.pkl"))
    print("💾 Modelo KNN salvo como 'knn_model.pkl' e scaler 'scaler_knn.pkl'.")

# Execução (main)
if __name__ == "__main__":
    train_and_save_model()
