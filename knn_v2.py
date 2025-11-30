import os
import time
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib

# Importações específicas do KNN
from sklearn.neighbors import KNeighborsClassifier 
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# --- Configurações ---
folder_train_true = "/Users/fabioakira/Downloads/reais_train"
folder_train_fake = "/Users/fabioakira/Downloads/fakes_train"
folder_test_true = "/Users/fabioakira/Downloads/reais_test"
folder_test_fake = "/Users/fabioakira/Downloads/fakes_test"

# --- Parâmetros de Áudio ---
N_MFCC = 40       # Número de coeficientes MFCC
TARGET_SR = 16000 # Sample rate alvo (16000 Hz)

# --- Parâmetros de ML (Ajuste para KNN) ---
N_VIZINHOS = 5    # Número de vizinhos (K) para o KNN

SAVE_FOLDER = "knn_results"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# --- Funções Auxiliares ---

def _process_folder(root_folder, label):
    """
    Função auxiliar para caminhar recursivamente por uma pasta,
    encontrar arquivos .wav e extrair features.
    """
    features_list = []
    labels_list = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                filepath = os.path.join(dirpath, filename)
                try:
                    features = extract_features(filepath) 
                    features_list.append(features)
                    labels_list.append(label)
                except Exception as e:
                    print(f"⚠️ Erro ao processar o arquivo {filepath}: {e}")

    return features_list, labels_list

def extract_features(filepath, n_mfcc=N_MFCC):
    """
    Extrai a média e o desvio padrão dos MFCCs.
    """
    y, sr = librosa.load(filepath, sr=TARGET_SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat_mean = np.mean(mfcc, axis=1)
    feat_std = np.std(mfcc, axis=1)
    return np.concatenate([feat_mean, feat_std])

def load_manual_dataset(real_train, fake_train, real_test, fake_test):
    print("Carregando dados de TREINO...")
    X_rt, y_rt = _process_folder(real_train, 0)
    X_ft, y_ft = _process_folder(fake_train, 1)

    print("\nCarregando dados de TESTE...")
    X_rv, y_rv = _process_folder(real_test, 0)
    X_fv, y_fv = _process_folder(fake_test, 1)

    X_train = np.array(X_rt + X_ft)
    y_train = np.array(y_rt + y_ft)

    X_test = np.array(X_rv + X_fv)
    y_test = np.array(y_rv + y_fv)

    print(f"\nTreino: {len(X_train)} arquivos | Teste: {len(X_test)} arquivos")
    return X_train, y_train, X_test, y_test

# --- Treinamento do Modelo (KNN) ---
def train_and_save_model():
    start_time = time.time() 

    X_train, y_train, X_test, y_test = load_manual_dataset(
        folder_train_true, folder_train_fake,
        folder_test_true, folder_test_fake
    )

    if len(X_train) == 0 or len(X_test) == 0:
        print("\n❌ ERRO: Nenhum arquivo .wav foi encontrado nos conjuntos de treino ou teste!")
        return 

    # Normalização é crucial para KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinamento do modelo KNN
    clf = KNeighborsClassifier(n_neighbors=N_VIZINHOS)
    clf.fit(X_train_scaled, y_train)

    # Avaliação
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    end_time = time.time()
    elapsed_time_sec = end_time - start_time

    # CÁLCULO E FORMATAÇÃO DO TEMPO
    minutes = int(elapsed_time_sec // 60)
    seconds = int(elapsed_time_sec % 60)    
    time_display = f"{minutes} min {seconds} seg" if minutes > 0 else f"{elapsed_time_sec:.2f} seg"

    print("\n=== RESULTADOS DO TREINO (KNN) ===")
    print(f"Usando {N_MFCC} MFCCs e K={N_VIZINHOS}")
    print(f"Accuracy: {acc:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"⏱️ Tempo Total: {time_display}")
    print("Matriz de Confusão:\n", cm)

    # --- Gráfico da Matriz de Confusão ---
    plt.figure(figsize=(6,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(values_format='d')
    plt.title("Matriz de Confusão - KNN")
    plt.savefig(os.path.join(SAVE_FOLDER, "matriz_confusao_knn.png"), dpi=300)
    print("🖼️ Matriz de confusão salva como matriz_confusao_knn.png")
    plt.show()

    # --- Curva ROC ---
    plt.figure(figsize=(6,5))
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("Curva ROC - KNN")
    plt.savefig(os.path.join(SAVE_FOLDER, "curva_roc_knn.png"), dpi=300)
    print("🖼️ Curva ROC salva como curva_roc_knn.png")
    plt.show()

    # Salvar modelo + scaler
    joblib.dump(clf, os.path.join(SAVE_FOLDER, "knn_model.pkl"))
    joblib.dump(scaler, os.path.join(SAVE_FOLDER, "scaler_knn.pkl"))
    print("\n💾 Modelo KNN salvo como 'knn_model.pkl' e scaler 'scaler_knn.pkl'.")

    # --- Visualização ---
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob[y_test == 0], bins=15, alpha=0.6, label="Real (0)")
    plt.hist(y_prob[y_test == 1], bins=15, alpha=0.6, label="Fake (1)")
    plt.axvline(0.5, color="k", linestyle="--", label="Limite (0.5)")
    plt.xlabel("Probabilidade Prevista de ser Falso")
    plt.ylabel("Quantidade")
    plt.title(f"KNN - Distribuição de Probabilidade (K={N_VIZINHOS})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# --- Execução ---
if __name__ == "__main__":
    train_and_save_model()
