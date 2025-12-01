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

# --- Configurações ---
#folder_true = "/Users/fabioakira/Downloads/reais"
#folder_fake = "/Users/fabioakira/Downloads/fakes"
folder_train_true = "/Users/fabioakira/Downloads/reais_train"
folder_train_fake = "/Users/fabioakira/Downloads/fakes_train"
folder_test_true = "/Users/fabioakira/Downloads/reais_test"
folder_test_fake = "/Users/fabioakira/Downloads/fakes_test"

# Pasta para salvar resultados
SAVE_FOLDER = "logistic_regression_results"

# --- Parâmetros de Áudio (AJUSTE AQUI) ---
N_MFCC = 40       # <<-- AQUI: Número de coeficientes MFCC desejado
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
                    features = extract_features_mfcc(filepath)
                    features_list.append(features)
                    labels_list.append(label)
                except Exception as e:
                    # Adiciona um try/except para o caso de um áudio falhar
                    print(f"⚠️ Erro ao processar o arquivo {filepath}: {e}")

    return features_list, labels_list

# --- Extração de Features ---
def extract_features_mfcc(filepath, n_mfcc=N_MFCC):
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
    print("\n=== Carregando dados de TREINO ===")
    X_real, y_real = _process_folder(real_folder, 0)
    X_fake, y_fake = _process_folder(fake_folder, 1)

    X = np.array(X_real + X_fake)
    y = np.array(y_real + y_fake)

    print(f"Treino: {len(y_real)} reais + {len(y_fake)} falsos = {len(X)} amostras")
    return shuffle(X, y, random_state=42)

def load_test_dataset(real_folder, fake_folder, fraction=0.2):
    print("\n=== Carregando dados de TESTE ===")
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
    # INÍCIO: Captura o tempo antes de começar o processamento
    start_time = time.time()
    os.makedirs(SAVE_FOLDER, exist_ok=True) # Precisa estar aqui, para criar a pasta corretamente

    #X, y = load_dataset(folder_true, folder_fake)
    X_train, y_train = load_train_dataset(folder_train_true, folder_train_fake)
    X_test, y_test = load_test_dataset(folder_test_true, folder_test_fake, fraction=0.2)

    #if len(X) == 0:
    #    return # Para a execução se nenhum arquivo foi carregado

    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, stratify=y, test_size=0.2, random_state=42 # antes, era 0.5
    #)

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

    # FIM: Captura o tempo final e calcula a duração
    end_time = time.time()
    elapsed_time_sec = end_time - start_time # Tempo total em segundos

    # CÁLCULO E FORMATAÇÃO DO TEMPO
    minutes = int(elapsed_time_sec // 60)
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
    plt.title("Confusion Matrix - LogReg, MFCC")
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
    plt.title("Curva ROC - LogReg, MFCC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(os.path.join(SAVE_FOLDER, "curva_roc_logreg.png"), dpi=300)
    print("💾 Curva ROC salva como curva_roc_logreg.png")
    plt.show()

    # Salvar modelo + scaler
    joblib.dump(clf, "logreg_model.pkl")
    joblib.dump(scaler, "scaler_logreg.pkl")
    print("\n💾 Modelo e scaler salvos como 'logreg_model.pkl' e 'scaler_logreg.pkl'.")

    # Visualização
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
    # 1. Treina o modelo
    train_and_save_model()
