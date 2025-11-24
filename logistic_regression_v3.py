import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# --- Configurações ---
folder_true = "/Users/fabioakira/Downloads/reais"
folder_fake = "/Users/fabioakira/Downloads/fakes"

# Arquivo para testar após o treino
test_audio_path = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/audio.wav"  # mude para seu arquivo

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
                    features = extract_features(filepath) 
                    features_list.append(features)
                    labels_list.append(label)
                except Exception as e:
                    # Adiciona um try/except para o caso de um áudio falhar
                    print(f"⚠️ Erro ao processar o arquivo {filepath}: {e}")

    return features_list, labels_list

# --- Extração de Features ---
def extract_features(filepath, n_mfcc=N_MFCC): # <--- ALTERADO AQUI
    """
    Extrai a média e o desvio padrão dos MFCCs para um arquivo de áudio.
    Usa as variáveis globais N_MFCC e TARGET_SR por padrão.
    """
    # Carrega o áudio, forçando o sample rate para TARGET_SR
    y, sr = librosa.load(filepath, sr=TARGET_SR) # <--- ALTERADO AQUI

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

# --- Treinamento do Modelo ---
def train_and_save_model():
    X, y = load_dataset(folder_true, folder_fake)

    if len(X) == 0:
        return # Para a execução se nenhum arquivo foi carregado

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42 # antes, era 0.5
    )

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

    print("\n=== RESULTADOS DO TREINO ===")
    print(f"Usando {N_MFCC} coeficientes MFCC.")
    print(f"Accuracy: {acc:.3f}")
    print(f"AUC: {auc:.3f}")
    print("Matriz de Confusão:\n", cm)

    # Salvar modelo + scaler
    joblib.dump(clf, "logreg_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\n💾 Modelo e scaler salvos como 'logreg_model.pkl' e 'scaler.pkl'.")

    # --- Visualização ---
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

# --- Teste de novo arquivo ---
def classify_new_audio(audio_path):
    print("\n--- CLASSIFICANDO NOVO ARQUIVO ---")
    if not os.path.exists("logreg_model.pkl") or not os.path.exists("scaler.pkl"):
        print("⚠️ Você precisa treinar o modelo primeiro! Execute o script sem comentar 'train_and_save_model()'.")
        return

    if not os.path.exists(audio_path):
        print(f"⚠️ Arquivo de teste não encontrado: {audio_path}")
        return

    clf = joblib.load("logreg_model.pkl")
    scaler = joblib.load("scaler.pkl")

    try:
        # Extrai features usando as mesmas configurações (N_MFCC, TARGET_SR)
        features = extract_features(audio_path).reshape(1, -1)
    except Exception as e:
        print(f"Erro ao processar o arquivo de teste {audio_path}: {e}")
        return

    # Normaliza as features com o scaler salvo
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