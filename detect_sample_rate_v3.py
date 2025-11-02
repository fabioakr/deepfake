import os
import librosa
import numpy as np # librosa returns numpy arrays

# --- Your directories ---
#folder_true = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Benita_F008"
#folder_fake = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Benita_F008_Fake"

folder_true = "/Users/fabioakira/Downloads/reais"
folder_fake = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Benita_F008_Fake"

def load_audios_and_check_properties(folder_path, label_name=""):
    """
    Loads all .wav files from a folder and all its subfolders,
    prints their sample rates and channel counts, and checks consistency.
    
    Returns: 
        dict {filepath: (audio_array, sample_rate)},
        set {unique_sample_rates},
        set {unique_channel_counts}
    """
    audios = {}
    properties = [] # Will store (sr, channels) tuples

    print(f"\n🔍 Checking folder: {label_name or folder_path}")

    # os.walk will traverse the top folder AND all subfolders
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(".wav"):
                # fpath will be the full path, including the subfolder
                fpath = os.path.join(root, fname)
                try:
                    # Load audio with native SR and without mixing to mono
                    y, sr = librosa.load(fpath, sr=None, mono=False)
                    
                    # Determine channel count from the array shape
                    # y.ndim == 1 -> Mono
                    # y.ndim == 2 -> Stereo/Multi-channel, shape is (channels, samples)
                    channels = y.shape[0] if y.ndim == 2 else 1
                    ch_str = 'Mono' if channels == 1 else 'Stereo' if channels == 2 else f'{channels} channels'

                    # Get sample count (length)
                    num_samples = y.shape[1] if y.ndim == 2 else y.shape[0]

                    audios[fpath] = (y, sr)
                    properties.append((sr, channels))
                    
                    # Get relative path for cleaner printing
                    relative_path = os.path.relpath(fpath, folder_path)
                    print(f"✅ {relative_path}: {sr} Hz, {ch_str} ({num_samples} samples)")
                    
                except Exception as e:
                    print(f"⚠️ Error reading {fpath}: {e}")

    if not audios:
        print("No .wav files found.")
        return {}, set(), set()

    # --- Get Unique Properties ---
    unique_srs = set(p[0] for p in properties)
    unique_chs = set(p[1] for p in properties)

    print(f"\n📊 Summary for {label_name}:")
    print(f"Found {len(audios)} files")

    # --- Sample Rate Summary ---
    print("Unique sample rates:", unique_srs)
    if len(unique_srs) == 1:
        print(f"✅ All files have the same sample rate: {list(unique_srs)[0]} Hz")
    else:
        print("❌ Mismatch detected! Some files have different sample rates.")
        
    # --- Channel Summary ---
    print("Unique channel counts:", unique_chs)
    if len(unique_chs) == 1:
        ch = list(unique_chs)[0]
        ch_str = 'Mono' if ch == 1 else 'Stereo' if ch == 2 else f'{ch} channels'
        print(f"✅ All files have the same channel count: {ch_str}")
    else:
        print("❌ Mismatch detected! Some files have different channel counts.")
        # Add friendly names for common cases
        channel_descs = ['Mono' if c == 1 else 'Stereo' if c == 2 else f'{c} channels' for c in unique_chs]
        print(f"   (Found: {', '.join(channel_descs)})")

    return audios, unique_srs, unique_chs


# --- Run checks for both datasets ---
if __name__ == "__main__":
    audios_true, srs_true, chs_true = load_audios_and_check_properties(folder_true, label_name="REAL (Benita_F008)")
    audios_fake, srs_fake, chs_fake = load_audios_and_check_properties(folder_fake, label_name="FAKE (Benita_F008_Fake)")

    print("\n" + "="*30)
    print("=== 🧠 OVERALL CHECK ===")
    
    # --- Check Sample Rates ---
    combined_srs = srs_true.union(srs_fake)
    if len(combined_srs) == 1:
        print(f"✅ All real + fake files share the same sample rate: {list(combined_srs)[0]} Hz")
    elif not combined_srs:
        print("🤷 No files found in either folder.")
    else:
        print(f"❌ Mismatch between folders! Found sample rates: {combined_srs}") 

    # --- Check Channels ---
    combined_chs = chs_true.union(chs_fake)
    if len(combined_chs) == 1:
        ch = list(combined_chs)[0]
        ch_str = 'Mono' if ch == 1 else 'Stereo' if ch == 2 else f'{ch} channels'
        print(f"✅ All real + fake files share the same channel count: {ch_str}")
    elif not combined_chs:
        pass # Already printed 'No files found'
    else:
        print(f"❌ Mismatch between folders! Found channel counts: {combined_chs}")
        channel_descs = ['Mono' if c == 1 else 'Stereo' if c == 2 else f'{c} channels' for c in combined_chs]
        print(f"   (Found: {', '.join(channel_descs)})")
    
    print("="*30)