import os
import soundfile as sf  # safer and faster than librosa.load for metadata

def check_sample_rates(folder_path):
    """
    Checks if all .wav files in a folder have the same sample rate.
    Prints mismatches and returns True if all are identical.
    """
    sample_rates = {}
    
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(".wav"):
                fpath = os.path.join(root, fname)
                try:
                    # Read only metadata (no audio data)
                    info = sf.info(fpath)
                    sr = info.samplerate
                    sample_rates[fpath] = sr
                except Exception as e:
                    print(f"⚠️ Error reading {fpath}: {e}")

    if not sample_rates:
        print("No .wav files found.")
        return False

    # Get all unique sample rates
    unique_rates = set(sample_rates.values())
    print("\n--- Sample Rate Summary ---")
    for f, sr in sample_rates.items():
        print(f"{f}: {sr} Hz")

    if len(unique_rates) == 1:
        sampling_rate = unique_rates.pop()
        print("\n✅ All files have the same sample rate:", sampling_rate, "Hz")
        return True, sampling_rate
    else:
        print("\n❌ Mismatch detected! Different sample rates found:", unique_rates)
        return False


if __name__ == "__main__":
    folder_true = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Benita_F008"   # <-- change to your folder path
    folder_fake = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Benita_F008_Fake"   # <-- change to your folder pat
    a,b = check_sample_rates(folder_true)
    c,d = check_sample_rates(folder_fake)
    
    print(a,b,c,d)