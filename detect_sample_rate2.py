import os
import librosa

# --- Your directories ---
folder_true = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Benita_F008"
folder_fake = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Benita_F008_Fake"

def load_audios_and_check_sr(folder_path, label_name=""):
    """
    Loads all .wav files from a folder,
    prints their sample rates, and checks consistency.
    Returns: dict {filepath: (audio_array, sample_rate)}
    """
    audios = {}
    srs = []

    print(f"\n🔍 Checking folder: {label_name or folder_path}")

    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(".wav"):
                fpath = os.path.join(root, fname)
                try:
                    # Load audio with native sample rate (no resampling)
                    y, sr = librosa.load(fpath, sr=None)
                    audios[fpath] = (y, sr)
                    srs.append(sr)
                    print(f"✅ {fname}: {sr} Hz ({len(y)} samples)")
                except Exception as e:
                    print(f"⚠️ Error reading {fname}: {e}")

    if not audios:
        print("No .wav files found.")
        return {}, set()

    unique_srs = set(srs)
    print(f"\n📊 Summary for {label_name}:")
    print(f"Found {len(audios)} files")
    print("Unique sample rates:", unique_srs)

    if len(unique_srs) == 1:
        print(f"✅ All files have the same sample rate: {list(unique_srs)[0]} Hz")
    else:
        print("❌ Mismatch detected! Some files have different sample rates.")

    return audios, unique_srs


# --- Run checks for both datasets ---
if __name__ == "__main__":
    audios_true, srs_true = load_audios_and_check_sr(folder_true, label_name="REAL (Benita_F008)")
    audios_fake, srs_fake = load_audios_and_check_sr(folder_fake, label_name="FAKE (Benita_F008_Fake)")

    print("\n=== 🧠 OVERALL CHECK ===")
    combined_srs = srs_true.union(srs_fake)
    if len(combined_srs) == 1:
        print(f"✅ All real + fake files share the same sample rate: {list(combined_srs)[0]} Hz")
    else:
        print(f"❌ Mismatch between folders! Found sample rates: {combined_srs}")