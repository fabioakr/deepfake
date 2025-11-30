import os

folder_train_true = "/Users/fabioakira/Downloads/reais_train"
folder_train_fake = "/Users/fabioakira/Downloads/fakes_train"
folder_test_true  = "/Users/fabioakira/Downloads/reais_test"
folder_test_fake  = "/Users/fabioakira/Downloads/fakes_test"

def count_wav_files(folder):
    count = 0
    for _, _, files in os.walk(folder):
        count += sum(1 for f in files if f.lower().endswith(".wav"))
    return count

print("\n=== Contagem de Áudios ===")
print(f"Reais (Treino): {count_wav_files(folder_train_true)}")
print(f"Falsos (Treino): {count_wav_files(folder_train_fake)}")
print(f"Reais (Teste): {count_wav_files(folder_test_true)}")
print(f"Falsos (Teste): {count_wav_files(folder_test_fake)}")
print("===========================\n")