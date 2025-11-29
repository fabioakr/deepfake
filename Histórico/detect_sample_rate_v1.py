import os
import soundfile as sf  # safer and faster than librosa.load for metadata

def check_audio_properties(folder_path):
    """
    Checks if all .wav files in a folder have the same sample rate
    and the same number of channels (e.g., all mono or all stereo).
    
    Prints mismatches and returns status for both checks.
    """
    audio_properties = {}  # Store tuples: (samplerate, channels)

    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(".wav"):
                fpath = os.path.join(root, fname)
                try:
                    # Read only metadata (no audio data)
                    info = sf.info(fpath)
                    sr = info.samplerate
                    ch = info.channels
                    audio_properties[fpath] = (sr, ch)
                except Exception as e:
                    print(f"⚠️ Error reading {fpath}: {e}")

    if not audio_properties:
        print("No .wav files found.")
        # Return: sr_match, sr_val, ch_match, ch_val
        return False, None, False, None

    # Get all unique sample rates and channel counts
    unique_rates = set(props[0] for props in audio_properties.values())
    unique_channels = set(props[1] for props in audio_properties.values())

    print("\n--- Audio Property Summary ---")
    # Optional: Uncomment to see details for every file
    #for f, (sr, ch) in audio_properties.items():
    #    ch_str = 'Mono' if ch == 1 else 'Stereo' if ch == 2 else f'{ch} channels'
    #    print(f"{f}: {sr} Hz, {ch_str}")

    # --- Check Sample Rates ---
    all_rates_match = False
    consistent_rate = None
    if len(unique_rates) == 1:
        consistent_rate = unique_rates.pop()
        print(f"✅ All files have the same sample rate: {consistent_rate} Hz")
        all_rates_match = True
    else:
        print(f"❌ Mismatch detected! Different sample rates found: {unique_rates}")

    # --- Check Channels ---
    all_channels_match = False
    consistent_channels = None
    if len(unique_channels) == 1:
        consistent_channels = unique_channels.pop()
        # Friendly name for channels
        ch_str = 'Mono' if consistent_channels == 1 else 'Stereo' if consistent_channels == 2 else f'{consistent_channels} channels'
        print(f"✅ All files have the same channel count: {consistent_channels} ({ch_str})")
        all_channels_match = True
    else:
        print(f"❌ Mismatch detected! Different channel counts found: {unique_channels}")
        # Add friendly names for common cases
        channel_descs = ['Mono' if c == 1 else 'Stereo' if c == 2 else f'{c} channels' for c in unique_channels]
        print(f"   (Found: {', '.join(channel_descs)})")

    return all_rates_match, consistent_rate, all_channels_match, consistent_channels


if __name__ == "__main__":
    folder_true = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Benita_F008"   # <-- change to your folder path
    folder_fake = "/Users/fabioakira/Desktop/POLI/TCC/deepfake/deepfake/Benita_F008_Fake"   # <-- change to your folder path
    
    print(f"--- Checking folder: {folder_true} ---")
    sr_match_true, rate_true, ch_match_true, channels_true = check_audio_properties(folder_true)
    
    print(f"\n--- Checking folder: {folder_fake} ---")
    sr_match_fake, rate_fake, ch_match_fake, channels_fake = check_audio_properties(folder_fake)

    print("\n" + "="*30)
    print("--- Final Summary ---")
    print(f"Folder 1 ({os.path.basename(folder_true)}):")
    print(f"  Sample Rate Match: {sr_match_true} (Value: {rate_true} Hz)")
    print(f"  Channel Match:     {ch_match_true} (Value: {channels_true} channels)")
    print(f"Folder 2 ({os.path.basename(folder_fake)}):")
    print(f"  Sample Rate Match: {sr_match_fake} (Value: {rate_fake} Hz)")
    print(f"  Channel Match:     {ch_match_fake} (Value: {channels_fake} channels)")
    print("="*30)
