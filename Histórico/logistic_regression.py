import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# --- Folders ---
#folder_true = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Benita_F008"
#folder_fake = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Benita_F008_Fake"

folder_true = "/Users/fabioakira/Downloads/reais"
folder_fake = "/Users/fabioakira/Downloads/fakes"

# File to test after training
test_audio_path = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/audio.wav"  # change to your test file

# --- Feature extraction ---
def extract_features(filepath, n_mfcc=13):
    """Extract MFCC mean and std features for one audio file."""
    y, sr = librosa.load(filepath, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat_mean = np.mean(mfcc, axis=1)
    feat_std = np.std(mfcc, axis=1)
    return np.concatenate([feat_mean, feat_std])  # shape (26,)

# --- Load dataset ---
def load_dataset(real_folder, fake_folder):
    X, y = [], []
    for f in os.listdir(real_folder):
        if f.lower().endswith(".wav"):
            X.append(extract_features(os.path.join(real_folder, f)))
            y.append(0)
    for f in os.listdir(fake_folder):
        if f.lower().endswith(".wav"):
            X.append(extract_features(os.path.join(fake_folder, f)))
            y.append(1)
    X = np.array(X)
    y = np.array(y)
    print(f"✅ Loaded {len(X)} files ({sum(y==0)} real, {sum(y==1)} fake)")
    return X, y

# --- Train model ---
def train_and_save_model():
    X, y = load_dataset(folder_true, folder_fake)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.5, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== RESULTS ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"AUC: {auc:.3f}")
    print("Confusion matrix:\n", cm)

    # Save model + scaler
    joblib.dump(clf, "logreg_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\n💾 Model and scaler saved as 'logreg_model.pkl' and 'scaler.pkl'.")

    # --- Visualization ---
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob[y_test == 0], bins=15, alpha=0.6, label="Real (label=0)")
    plt.hist(y_prob[y_test == 1], bins=15, alpha=0.6, label="Fake (label=1)")
    plt.axvline(0.5, color="k", linestyle="--", label="Decision boundary (0.5)")
    plt.xlabel("Predicted Probability of Being Fake")
    plt.ylabel("Number of Samples")
    plt.title("Logistic Regression – Probability Separation")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# --- Test new file ---
def classify_new_audio(audio_path):
    if not os.path.exists("logreg_model.pkl") or not os.path.exists("scaler.pkl"):
        print("⚠️ You need to train the model first!")
        return

    clf = joblib.load("logreg_model.pkl")
    scaler = joblib.load("scaler.pkl")

    features = extract_features(audio_path).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prob = clf.predict_proba(features_scaled)[0, 1]
    label = "FAKE" if prob >= 0.5 else "REAL"

    print(f"\n🔎 File: {os.path.basename(audio_path)}")
    print(f"→ Predicted: {label}")
    print(f"→ Probability of being FAKE: {prob:.3f}")

# --- Run ---
if __name__ == "__main__":
    train_and_save_model()
    if os.path.exists(test_audio_path):
        classify_new_audio(test_audio_path)
    else:
        print(f"\n⚠️ Test file not found: {test_audio_path}")