import os
import time
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib

# Importações específicas do KNN
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# --- Configurações ---
folder_true = "/Users/fabioakira/Downloads/reais"
folder_fake = "/Users/fabioakira/Downloads/fakes"

# Arquivo para testar após o treino
test_audio_path = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/audio.wav"

# --- Parâmetros de Áudio ---
N_MFCC = 40       # Número de coeficientes MFCC
TARGET_SR = 16000 # Sample rate alvo (16000 Hz)

# --- Parâmetros de ML (Ajuste para KNN) ---
N_VIZINHOS = 5    # <<-- AQUI: Número de vizinhos (K) para o KNN

# --- Funções Auxiliares (mantidas as mesmas) ---

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

def load_dataset(real_folder, fake_folder):
    """
    Carrega o dataset.
    """
    print(f"Iniciando varredura com N_MFCC={N_MFCC} e Sample Rate={TARGET_SR}Hz")
    print(f"Procurando áudios reais em: {real_folder}")
    X_real, y_real = _process_folder(real_folder, 0) # Label 0 = Real

    print(f"\nProcurando áudios falsos em: {fake_folder}")
    X_fake, y_fake = _process_folder(fake_folder, 1) # Label 1 = Fake

    X = np.array(X_real + X_fake)
    y = np.array(y_real + y_fake)

    if len(X) == 0:
        print("\n❌ ERRO: Nenhum arquivo .wav foi encontrado!")
        print("Verifique os caminhos 'folder_true' e 'folder_fake' no seu script.")
        return np.array([]), np.array([])

    print(f"\n✅ Carregados {len(X)} arquivos ({len(y_real)} reais, {len(y_fake)} falsos)")
    print(f"Shape das features (X): {X.shape}")
    return X, y

# --- Treinamento do Modelo (AGORA USANDO KNN) ---
def train_and_save_model():
    start_time = time.time() 

    X, y = load_dataset(folder_true, folder_fake)

    if len(X) == 0:
        return 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Normalização é crucial para KNN, pois ele usa distâncias
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinamento do modelo KNN
    clf = KNeighborsClassifier(n_neighbors=N_VIZINHOS) # <--- MUDANÇA PRINCIPAL
    clf.fit(X_train_scaled, y_train)

    # Avaliação
    # Note: KNN tem predict_proba, o que permite calcular AUC
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
    
    if minutes > 0:
        time_display = f"{minutes} min {seconds} seg"
    else:
        time_display = f"{elapsed_time_sec:.2f} seg"

    print("\n=== RESULTADOS DO TREINO (KNN) ===")
    print(f"Usando {N_MFCC} coeficientes MFCC e K={N_VIZINHOS}.")
    print(f"Accuracy: {acc:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"⏱️ Tempo Total de Execução (Extração + Treino): {time_display}")
    print("Matriz de Confusão:\n", cm)

    # Salvar modelo + scaler (renomeando os arquivos)
    joblib.dump(clf, "knn_model.pkl")
    joblib.dump(scaler, "scaler_knn.pkl")
    print("\n💾 Modelo KNN e scaler salvos como 'knn_model.pkl' e 'scaler_knn.pkl'.")

    # --- Visualização ---
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob[y_test == 0], bins=15, alpha=0.6, label="Real (label=0)")
    plt.hist(y_prob[y_test == 1], bins=15, alpha=0.6, label="Fake (label=1)")
    plt.axvline(0.5, color="k", linestyle="--", label="Decision boundary (0.5)")
    plt.xlabel("Probabilidade Prevista de ser Falso")
    plt.ylabel("Número de Amostras")
    plt.title(f"KNN (MFCCs={N_MFCC}, K={N_VIZINHOS}) – Separação de Probabilidade")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# --- Teste de novo arquivo (Atualizado para KNN) ---
def classify_new_audio(audio_path):
    print("\n--- CLASSIFICANDO NOVO ARQUIVO (usando KNN) ---")
    
    model_file = "knn_model.pkl"
    scaler_file = "scaler_knn.pkl"

    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        print(f"⚠️ Você precisa treinar o modelo primeiro! Execute o script sem comentar 'train_and_save_model()'. (Esperando {model_file})")
        return

    if not os.path.exists(audio_path):
        print(f"⚠️ Arquivo de teste não encontrado: {audio_path}")
        return

    clf = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    try:
        features = extract_features(audio_path).reshape(1, -1)
    except Exception as e:
        print(f"Erro ao processar o arquivo de teste {audio_path}: {e}")
        return

    features_scaled = scaler.transform(features)

    # Faz a predição
    prob = clf.predict_proba(features_scaled)[0, 1]
    label = "FAKE" if prob >= 0.5 else "REAL"

    print(f"\n🔎 Arquivo: {os.path.basename(audio_path)}")
    print(f"→ Veredito: {label}")
    print(f"→ Probabilidade de ser FAKE: {prob:.3f}")

# --- Execução ---
if __name__ == "__main__":
    # 1. Treina o modelo
    train_and_save_model()

    # 2. Classifica o arquivo de teste
    classify_new_audio(test_audio_path)