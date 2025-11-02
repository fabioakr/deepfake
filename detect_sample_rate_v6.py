"""
Script para analisar as propriedades da base de dados 
a ser utilizada para treinamento neste projeto de TCC.

Alunos:
Fábio Akira Yonamine
Monique Menezes XXXX
"""

import os
import librosa
import numpy as np

# Localização das pastas, com áudios reais / gerados artificialmente
folder_true = "/Users/fabioakira/Downloads/reais"
folder_fake = "/Users/fabioakira/Downloads/fakes"

# Função auxiliar para imprimir estatísticas
def print_summary_stats(label, durations_sec, sizes_bytes):
    """
    Imprime informações dos áudios do conjunto em análise, 
    como durações e tamanhos de arquivo.
    """
    print(f"\n" + "-"*30)
    print(f"SUMÁRIO DE ESTATÍSTICAS: {label}")
    print(f"Total de Arquivos: {len(durations_sec)}")

    # --- Sumário de Duração ---
    print("\n--- Duração ---")
    if durations_sec:
        durations_np = np.array(durations_sec)
        total_dur = np.sum(durations_np)

        print(f"Duração Total: {total_dur / 3600:.2f} horas ({total_dur / 60:.2f} minutos)")
        print(f"Média:    {np.mean(durations_np):.2f} s")
        print(f"Variância: {np.var(durations_np):.2f} s²")
        print(f"Desv. Padrão: {np.std(durations_np):.2f} s")
        print(f"Min: {np.min(durations_np):.2f} s, Max: {np.max(durations_np):.2f} s")
    else:
        print("Nenhuma duração calculada.")

    # --- Sumário de Tamanho de Arquivo ---
    print("\n--- Tamanho de Arquivo (Megabytes) ---")
    if sizes_bytes:
        sizes_np_mb = np.array(sizes_bytes) / (1024 * 1024) # Converte para MB
        total_size_mb = np.sum(sizes_np_mb)

        if total_size_mb > 1024: # Se for maior que 1 GB, mostre em GB
             print(f"Tamanho Total: {total_size_mb / 1024:.2f} GB ({total_size_mb:.2f} MB)")
        else:
             print(f"Tamanho Total: {total_size_mb:.2f} MB")

        print(f"Média:    {np.mean(sizes_np_mb):.2f} MB")
        print(f"Variância: {np.var(sizes_np_mb):.2f} MB²")
        print(f"Desv. Padrão: {np.std(sizes_np_mb):.2f} MB")
        print(f"Min: {np.min(sizes_np_mb):.2f} MB, Max: {np.max(sizes_np_mb):.2f} MB")
    else:
        print("Nenhum tamanho de arquivo calculado.")
    print(f"{"-"*30}\n")


def load_audios_and_check_properties(folder_path, label_name=""):
    """
    Carrega individualmente cada arquivo .wav na pasta informada 
    e verifica suas propriedades (SR, canais, duração, tamanho).
    
    Depois de ler as propriedades, descarta o vetor de áudio da
    memória, antes de carregar o próximo, para não haver estouro
    na RAM.
    
    Retorna: 
        set {unique_sample_rates},
        set {unique_channel_counts},
        list [all_durations_in_seconds],
        list [all_file_sizes_in_bytes]
    """
    properties = []
    durations = []
    file_sizes_bytes = []
    file_count = 0

    print(f"\n Verificando pasta: {label_name or folder_path}")

    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(".wav"):
                fpath = os.path.join(root, fname)
                file_count += 1
                try:
                    # --- Obtém o tamanho do arquivo (comandos leves para a RAM) ---
                    file_size_bytes = os.path.getsize(fpath)
                    file_sizes_bytes.append(file_size_bytes)
                    file_size_mb = file_size_bytes / (1024 * 1024)

                    # --- Carrega o arquivo no Librosa (usa bastante RAM) ---
                    y, sr = librosa.load(fpath, sr=None, mono=False)

                    # --- Librosa obtém propriedades do arquivo ---
                    channels = y.shape[0] if y.ndim == 2 else 1
                    ch_str = 'Mono' if channels == 1 else 'Stereo' if channels == 2 else f'{channels} channels'
                    duration_sec = librosa.get_duration(y=y, sr=sr)
                    durations.append(duration_sec)
                    properties.append((sr, channels))

                    # --- Imprime confirmação de leitura e informações individuais ---
                    num_samples = y.shape[1] if y.ndim == 2 else y.shape[0]
                    relative_path = os.path.relpath(fpath, folder_path)
                    print(f"🆗 Arquivo {relative_path}: {sr} Hz, {ch_str} ({num_samples} samples, {duration_sec:.2f}s, {file_size_mb:.2f} MB)")

                except Exception as e:
                    print(f"⚠️ Erro ao ler {fpath}: {e}")

    if file_count == 0:
        print("Nenhum arquivo .wav encontrado.")
        return set(), set(), [], []

    unique_srs = set(p[0] for p in properties)
    unique_chs = set(p[1] for p in properties)

    print(f"\n Sumário para {label_name}:")
    print(f"Encontrados e lidos {file_count} arquivos")

    # --- Sumário de Sample Rate ---
    print("\n--- Estatísticas de Sample Rate ---")
    print("Sample rates únicos:", unique_srs)
    if len(unique_srs) == 1:
        print(f"✅ Todos os arquivos têm a mesma sample rate: {list(unique_srs)[0]} Hz")
    else:
        print("❌ Incompatibilidade detectada! Alguns arquivos têm sample rates diferentes.")

    # --- Sumário de número de canais (Mono/Estéreo) ---
    print("\n--- Estatísticas de Canais ---")
    print("Contagens de canais únicas:", unique_chs)
    if len(unique_chs) == 1:
        ch = list(unique_chs)[0]
        ch_str = 'Mono' if ch == 1 else 'Stereo' if ch == 2 else f'{ch} canais'
        print(f"✅ Todos os arquivos têm a mesma contagem de canais: {ch_str}")
    else:
        print("❌ Incompatibilidade detectada! Alguns arquivos têm contagens de canais diferentes.")
        channel_descs = ['Mono' if c == 1 else 'Stereo' if c == 2 else f'{c} canais' for c in unique_chs]
        print(f"   (Encontrados: {', '.join(channel_descs)})")

    # Função para imprimir o sumário imediato ---
    print_summary_stats(f"{label_name}", durations, file_sizes_bytes)

    return unique_srs, unique_chs, durations, file_sizes_bytes


# --- Run checks for both datasets ---
if __name__ == "__main__":

    srs_true, chs_true, durs_true, sizes_true = load_audios_and_check_properties(folder_true, label_name="Áudios Reais")
    srs_fake, chs_fake, durs_fake, sizes_fake = load_audios_and_check_properties(folder_fake, label_name="Áudios Artificiais")

    print("\n" + "="*40)
    print("=== VERIFICAÇÃO GERAL E SUMÁRIO FINAL ===")
    print("="*40)

    # --- Verifica sample rates ---
    print("\n--- Verificação de Compatibilidade (Sample Rate) ---")
    combined_srs = srs_true.union(srs_fake)
    if len(combined_srs) == 1:
        print(f"✅ Todos os arquivos (real + fake) compartilham a mesma sample rate: {list(combined_srs)[0]} Hz")
    elif not combined_srs:
        print("❌ Nenhum arquivo encontrado nas pastas.")
    else:
        print(f"❌ Incompatibilidade entre pastas! Sample rates encontradas: {combined_srs}") 

    # --- Verifica se áudios são mono ou estéreo ---
    print("\n--- Verificação de Compatibilidade (Canais) ---")
    combined_chs = chs_true.union(chs_fake)
    if len(combined_chs) == 1:
        ch = list(combined_chs)[0]
        ch_str = 'Mono' if ch == 1 else 'Stereo' if ch == 2 else f'{ch} canais'
        print(f"✅ Todos os arquivos (real + fake) compartilham a mesma contagem de canais: {ch_str}")
    elif not combined_chs:
        pass # Aqui, ele já terial imprimido "Nenhum arquivo encontrado"
    else:
        print(f"❌ Incompatibilidade entre pastas! Contagens de canais encontradas: {combined_chs}")
        channel_descs = ['Mono' if c == 1 else 'Stereo' if c == 2 else f'{c} canais' for c in combined_chs]
        print(f"   (Encontrados: {', '.join(channel_descs)})")

    # 1. Sumário somente das amostras reais
    print_summary_stats("Áudios Reais", durs_true, sizes_true)

    # 2. Sumário somente das amostras artificiais
    print_summary_stats("Áudios Artificiais", durs_fake, sizes_fake)

    # 3. Sumário combinado de ambas as pastas
    combined_durs = durs_true + durs_fake
    combined_sizes = sizes_true + sizes_fake
    print_summary_stats("Base completa)", combined_durs, combined_sizes)

    print("="*40)
    print("Verificação concluída.")
    print("="*40)
