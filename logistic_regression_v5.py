"""
Esta versão utiliza LogRes com 20% das amostras de teste, para se adequar ao CNN e SVM.

As funções auxiliares load_audio(), linear_filter_banks() e extract_lfcc() são exclusivas
para a extração de LFCC.
"""

import os
import time  # Módulo para medir o tempo
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import scipy.fft
import soundfile as sf

# --- Configurações ---
folder_train_true = "/Users/fabioakira/Downloads/reais_train"
folder_train_fake = "/Users/fabioakira/Downloads/fakes_train"
folder_test_true = "/Users/fabioakira/Downloads/reais_test"
folder_test_fake = "/Users/fabioakira/Downloads/fakes_test"

# Pasta para salvar resultados
SAVE_FOLDER = "logistic_regression_results"

# --- Parâmetros de Áudio (AJUSTE AQUI) ---
N_MFCC = 5       # <<-- AQUI: Número de coeficientes MFCC desejado
TARGET_SR = 16000 # <<-- AQUI: Sample rate alvo (ex: 16000). Use None para usar o original.

## Notes: 1. Primeiro, tentei fazer o código com sample rate original dos áudios, assim como n_mfcc =13.
# acuracia 0.956 e AUC 0.991 (v2)

## Em seguida, forçando sample rate para 16 kHz e n_mfcc=20, a acuracia foi para 0.981 e AUC para 0.998

## Em seguida, forçando sample rate para 16 kHz e n_mfcc=40, a acuracia caiu para 0.988 e AUC para 0.999

## Mesma acuracia e AUC, se variar uso de amostras de teste para 20%.

# --- Funções Auxiliares ---

def _process_folder(root_folder, label):
    """
    Função auxiliar para caminhar recursivamente por uma pasta,
    encontrar arquivos .wav e extrair features.
    """
    features_list = []
    labels_list = []

    # os.walk é o segredo: ele desce por todas as subpastas
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                # Constrói o caminho completo do arquivo
                filepath = os.path.join(dirpath, filename)
                try:
                    # Extrai as features do arquivo
                    # A função extract_features usará os valores globais N_MFCC e TARGET_SR
                    features = extract_mfcc(filepath) ## MUDE AQUI ENTRE LFCC E MFCC E CQCC
                    features_list.append(features)
                    labels_list.append(label)
                except Exception as e:
                    # Adiciona um try/except para o caso de um áudio falhar
                    print(f"⚠️ Erro ao processar o arquivo {filepath}: {e}")

    return features_list, labels_list

# --- Extração de Features ---
def extract_mfcc(filepath, n_mfcc=N_MFCC):
    """
    Extrai a média e o desvio padrão dos MFCCs para um arquivo de áudio.
    """
    # Carrega o áudio, forçando o sample rate para TARGET_SR
    y, sr = librosa.load(filepath, sr=TARGET_SR)

    # Extrai os MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Calcula média e desvio padrão
    feat_mean = np.mean(mfcc, axis=1)
    feat_std = np.std(mfcc, axis=1)

    # A forma final será (2 * n_mfcc,)
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

def load_audio2(path, sr=TARGET_SR):
    y, _ = librosa.load(path, sr=sr)
    return y

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

# ============================================================
# 1. Função para extrair CQCC
# ============================================================
def extract_cqcc(path, sr=TARGET_SR, n_cqcc=N_MFCC, bins_per_octave=96):
    """
    CQCC = DCT(log(CQT^2))
    CQT → log-power → DCT → coeficientes
    """
    wave = load_audio(path)

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

#def extract_cqcc_mean(wave, sr=16000, n_cqcc=40):
#    cq = extract_cqcc(wave, sr=sr, n_cqcc=n_cqcc)
#    return np.mean(cq, axis=1)

def load_dataset(real_folder, fake_folder):
    """
    Carrega o dataset usando a função auxiliar _process_folder
    para lidar com as subpastas.
    """
    print(f"Iniciando varredura com N_MFCC={N_MFCC} e Sample Rate={TARGET_SR}Hz")
    print(f"Procurando áudios reais em: {real_folder}")
    X_real, y_real = _process_folder(real_folder, 0) # Label 0 = Real

    print(f"\nProcurando áudios falsos em: {fake_folder}")
    X_fake, y_fake = _process_folder(fake_folder, 1) # Label 1 = Fake

    # Combina as listas de áudios reais e falsos
    X = np.array(X_real + X_fake)
    y = np.array(y_real + y_fake)

    # Verifica se algum arquivo foi carregado
    if len(X) == 0:
        print("\n❌ ERRO: Nenhum arquivo .wav foi encontrado!")
        print("Verifique os caminhos 'folder_true' e 'folder_fake' no seu script.")
        return np.array([]), np.array([])

    print(f"\n✅ Carregados {len(X)} arquivos ({len(y_real)} reais, {len(y_fake)} falsos)")
    print(f"Shape das features (X): {X.shape}") # (n_amostras, 2 * N_MFCC)
    return X, y

def load_train_dataset(real_folder, fake_folder):
    print("\n=== Carregando dados de treino ===")
    X_real, y_real = _process_folder(real_folder, 0)
    X_fake, y_fake = _process_folder(fake_folder, 1)

    X = np.array(X_real + X_fake)
    y = np.array(y_real + y_fake)

    print(f"Treino: {len(y_real)} reais + {len(y_fake)} falsos = {len(X)} amostras")
    return shuffle(X, y, random_state=42)

def load_test_dataset(real_folder, fake_folder, fraction=0.2):
    print("\n=== Carregando dados de teste ===")
    X_real, y_real = _process_folder(real_folder, 0)
    X_fake, y_fake = _process_folder(fake_folder, 1)

    # Seleciona apenas 20% dos arquivos
    n_real = max(1, int(len(X_real) * fraction))
    n_fake = max(1, int(len(X_fake) * fraction))

    X_real_sel = X_real[:n_real]
    y_real_sel = y_real[:n_real]

    X_fake_sel = X_fake[:n_fake]
    y_fake_sel = y_fake[:n_fake]

    X = np.array(X_real_sel + X_fake_sel)
    y = np.array(y_real_sel + y_fake_sel)

    print(f"Teste: {n_real} reais + {n_fake} falsos = {len(X)} amostras selecionadas")
    return shuffle(X, y, random_state=42)

# --- Treinamento do Modelo ---
def train_and_save_model():
    # Começa o timer, antes de começar o processamento
    start_time = time.time()
    os.makedirs(SAVE_FOLDER, exist_ok=True) # Precisa estar aqui, para criar a pasta corretamente

    X_train, y_train = load_train_dataset(folder_train_true, folder_train_fake)
    X_test, y_test = load_test_dataset(folder_test_true, folder_test_fake, fraction=0.2)

    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinamento do modelo
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    # Avaliação
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Captura o tempo final e calcula a duração
    end_time = time.time()
    elapsed_time_sec = end_time - start_time # Tempo total em segundos
    minutes = int(elapsed_time_sec // 60) # formata em mm:ss
    seconds = int(elapsed_time_sec % 60)

    # Formatação da string de tempo
    if minutes > 0:
        time_display = f"{minutes} min {seconds} seg"
    else:
        time_display = f"{elapsed_time_sec:.2f} seg"

    print("\n=== RESULTADOS DO TREINO ===")
    print(f"Usando {N_MFCC} coeficientes MFCC.")
    print(f"Accuracy: {acc:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"Precision:  {precision:.3f}")
    print(f"Recall:     {recall:.3f}")
    print(f"F1-score:   {f1:.3f}")
    print(f"⏱️ Tempo Total de Execução (Extração + Treino): {time_display}")
    print("Matriz de Confusão:\n", cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    # Gera o gráfico: matriz de confusão
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real","Fake"],
        yticklabels=["Real","Fake"])
    plt.title(f"Confusion Matrix - LogReg, MFCC={N_MFCC}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(SAVE_FOLDER, "matriz_confusao_logreg.png"), dpi=300)
    print("💾 Matriz de confusão salva como matriz_confusao_logreg.png")
    plt.show()

    # Gera o gráfico: curva ROC
    plt.figure(figsize=(6,5))
    ax = plt.gca()
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.plot([0,1], [0,1], "--", color="gray", label="Aleatório (AUC=0.5)")
    plt.title(f"Curva ROC - LogReg, MFCC={N_MFCC}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(os.path.join(SAVE_FOLDER, "curva_roc_logreg.png"), dpi=300)
    print("💾 Curva ROC salva como curva_roc_logreg.png")
    plt.show()

    # Salva modelo + scaler
    joblib.dump(clf, os.path.join(SAVE_FOLDER, "logreg_model.pkl"))
    joblib.dump(scaler, os.path.join(SAVE_FOLDER, "scaler_logreg.pkl"))
    print("\n💾 Modelo e scaler salvos como 'logreg_model.pkl' e 'scaler_logreg.pkl'.")

    # Visualização original
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob[y_test == 0], bins=15, alpha=0.6, label="Real (label=0)")
    plt.hist(y_prob[y_test == 1], bins=15, alpha=0.6, label="Fake (label=1)")
    plt.axvline(0.5, color="k", linestyle="--", label="Decision boundary (0.5)")
    plt.xlabel("Probabilidade Prevista de ser Falso")
    plt.ylabel("Número de Amostras")
    plt.title(f"Regressão Logística (MFCCs={N_MFCC}) – Separação de Probabilidade")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# Execução (main)
if __name__ == "__main__":
    train_and_save_model()
