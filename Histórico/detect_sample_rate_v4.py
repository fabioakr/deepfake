import os
import librosa
import numpy as np

# Path das pastas, com áudios reais e gerados artificialmente
#folder_true = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/reais"
#folder_fake = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/fake"

folder_true = "/Users/fabioakira/Downloads/reais"
folder_fake = "/Users/fabioakira/Downloads/fakes"

def load_audios_and_check_properties(folder_path, label_name=""):
    """
    Loads each .wav file one-by-one, checks its properties (SR, channels, duration),
    and then discards the audio data from memory before loading the next.
    
    This checks consistency and calculates duration stats without crashing.
    
    Returns: 
        set {unique_sample_rates},
        set {unique_channel_counts},
        list [all_durations_in_seconds]
    """
    properties = [] # Will store (sr, channels) tuples
    durations = []  # Will store durations in seconds
    file_count = 0

    print(f"\n🔍 Checking folder: {label_name or folder_path}")

    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(".wav"):
                fpath = os.path.join(root, fname)
                file_count += 1
                try:
                    # --- Memory Intensive Step ---
                    # Load the *entire* file into memory
                    y, sr = librosa.load(fpath, sr=None, mono=False)
                    
                    # --- Get Properties ---
                    # Channels
                    channels = y.shape[0] if y.ndim == 2 else 1
                    ch_str = 'Mono' if channels == 1 else 'Stereo' if channels == 2 else f'{channels} channels'
                    
                    # Duration
                    duration_sec = librosa.get_duration(y=y, sr=sr)
                    durations.append(duration_sec)

                    # Store metadata
                    properties.append((sr, channels))
                    
                    # --- Print File Info ---
                    num_samples = y.shape[1] if y.ndim == 2 else y.shape[0]
                    relative_path = os.path.relpath(fpath, folder_path)
                    print(f"✅ Loaded {relative_path}: {sr} Hz, {ch_str} ({num_samples} samples, {duration_sec:.2f}s)")
                    
                    # --- Memory "Clear" Step ---
                    # 'y' and 'sr' are now discarded as the loop continues
                    
                except Exception as e:
                    print(f"⚠️ Error reading {fpath}: {e}")

    if file_count == 0:
        print("No .wav files found.")
        return set(), set(), []

    # --- Get Unique Properties ---
    unique_srs = set(p[0] for p in properties)
    unique_chs = set(p[1] for p in properties)

    print(f"\n📊 Summary for {label_name}:")
    print(f"Found and loaded {file_count} files")

    # --- Sample Rate Summary ---
    print("\n--- Sample Rate Stats ---")
    print("Unique sample rates:", unique_srs)
    if len(unique_srs) == 1:
        print(f"✅ All files have the same sample rate: {list(unique_srs)[0]} Hz")
    else:
        print("❌ Mismatch detected! Some files have different sample rates.")
        
    # --- Channel Summary ---
    print("\n--- Channel Stats ---")
    print("Unique channel counts:", unique_chs)
    if len(unique_chs) == 1:
        ch = list(unique_chs)[0]
        ch_str = 'Mono' if ch == 1 else 'Stereo' if ch == 2 else f'{ch} channels'
        print(f"✅ All files have the same channel count: {ch_str}")
    else:
        print("❌ Mismatch detected! Some files have different channel counts.")
        channel_descs = ['Mono' if c == 1 else 'Stereo' if c == 2 else f'{c} channels' for c in unique_chs]
        print(f"   (Found: {', '.join(channel_descs)})")

    # --- Duration Summary ---
    print("\n--- Duration Stats ---")
    if durations:
        durations_np = np.array(durations)
        avg_dur = np.mean(durations_np)
        var_dur = np.var(durations_np)
        std_dur = np.std(durations_np)
        min_dur = np.min(durations_np)
        max_dur = np.max(durations_np)
        total_dur = np.sum(durations_np)
        
        print(f"Total Duration: {total_dur / 60:.2f} minutes")
        print(f"Average:  {avg_dur:.2f} s")
        print(f"Variance: {var_dur:.2f} s²")
        print(f"Std. Dev: {std_dur:.2f} s")
        print(f"Min: {min_dur:.2f} s, Max: {max_dur:.2f} s")
    else:
        print("No durations calculated.")

    return unique_srs, unique_chs, durations


# --- Run checks for both datasets ---
if __name__ == "__main__":
    # We now get the list of durations back from each function call
    srs_true, chs_true, durs_true = load_audios_and_check_properties(folder_true, label_name="REAL (Benita_F008)")
    srs_fake, chs_fake, durs_fake = load_audios_and_check_properties(folder_fake, label_name="FAKE (Benita_F008_Fake)")

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
    
    # --- Combined Duration Stats ---
    print("\n--- Combined Duration Stats (All Files) ---")
    combined_durs = durs_true + durs_fake
    if combined_durs:
        durs_np = np.array(combined_durs)
        avg_dur = np.mean(durs_np)
        var_dur = np.var(durs_np)
        std_dur = np.std(durs_np)
        min_dur = np.min(durs_np)
        max_dur = np.max(durs_np)
        total_dur = np.sum(durs_np)

        print(f"Total Files: {len(durs_np)}")
        print(f"Total Duration: {total_dur / 3600:.2f} hours ({total_dur / 60:.2f} minutes)")
        print(f"Average:  {avg_dur:.2f} s")
        print(f"Variance: {var_dur:.2f} s²")
        print(f"Std. Dev: {std_dur:.2f} s")
        print(f"Min: {min_dur:.2f} s, Max: {max_dur:.2f} s")
    else:
        print("No durations calculated.")
        
    print("="*30)