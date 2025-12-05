"""
Script para analisar as propriedades das bases de dados 
a serem utilizadas para treinamento neste projeto de TCC.

Alunos:
Fábio Akira Yonamine - 11805398
Maria Monique de Menezes Cavalcanti - 11807935

Antes de executar este código, ajuste a localização das pastas com áudios, 
tal como foram baixadas da internet, abaixo.
"""

import os
import librosa
import numpy as np

# Localização das pastas, com áudios reais / gerados artificialmente
folder_true = "/Users/fabioakira/Downloads/reais"
folder_fake = "/Users/fabioakira/Downloads/fakes"

# A variável abaixo controla se o programa imprime detalhes de cada arquivo lido
verboseMode = False

# Função auxiliar para imprimir estatísticas de duração e tamanho
def print_summary_stats(label, durations_sec, sizes_bytes):
    """
    Imprime informações estatísticas de duração e tamanho de arquivo.
    Estes dados foram obtidos pela função load_audios_and_check_properties().
    """

    print(f"\n" + "="*50)
    print(f"=== Estatísticas da base: {label} ===")
    print(f"Total de Arquivos: {len(durations_sec)}")

    # Sumário de Duração
    print("\n--- Duração ---")
    if durations_sec:
        durations_np = np.array(durations_sec)
        total_dur = np.sum(durations_np)

        print(f"Duração Total: {total_dur / 3600:.2f} horas ({total_dur / 60:.2f} minutos)")
        print(f"Média:        {np.mean(durations_np):.2f} s")
        print(f"Variância:    {np.var(durations_np):.2f} s²")
        print(f"Desv. Padrão: {np.std(durations_np):.2f} s")
        print(f"Min: {np.min(durations_np):.2f} s, Max: {np.max(durations_np):.2f} s")
    else:
        print("Nenhuma duração calculada.")

    # Sumário de Tamanho de Arquivo
    print("\n--- Tamanho de Arquivo (Megabytes) ---")
    if sizes_bytes:
        sizes_np_mb = np.array(sizes_bytes) / (1024 * 1024) # Converte para MB
        total_size_mb = np.sum(sizes_np_mb)

        unit = "MB"
        total_display = total_size_mb
        if total_size_mb > 1024:
            unit = "GB"
            total_display = total_size_mb / 1024
            print(f"Tamanho Total: {total_display:.2f} GB ({total_size_mb:.2f} MB)")
        else:
            print(f"Tamanho Total: {total_display:.2f} MB")

        print(f"Média:        {np.mean(sizes_np_mb):.2f} MB")
        print(f"Variância:    {np.var(sizes_np_mb):.2f} MB²")
        print(f"Desv. Padrão: {np.std(sizes_np_mb):.2f} MB")
        print(f"Min: {np.min(sizes_np_mb):.2f} MB, Max: {np.max(sizes_np_mb):.2f} MB")
    else:
        print("Nenhum tamanho de arquivo calculado.")
    print(f"{"="*50}\n")


def print_compatibility_stats(label, unique_srs, unique_chs, file_count):
    """
    Imprime informações de compatibilidade (Sample Rate e Canais).
    Estes dados foram obtidos pela função load_audios_and_check_properties().
    """

    print(f"\n--- Compatibilidade da base: {label} ({file_count} arquivos) ---")

    # Sample Rate
    print("\n- Estatísticas de Sample Rate -")
    print("Sample rates únicos:", unique_srs)
    if len(unique_srs) == 1 and file_count > 0:
        print(f"✅ SR uniforme: {list(unique_srs)[0]} Hz")
    elif file_count == 0:
        print("Nenhum arquivo encontrado para análise de SR.")
    else:
        print("❌ Incompatibilidade detectada! Sample rates diferentes.")

    # Canais
    print("\n- Estatísticas de Canais -")
    print("Contagens de canais únicas:", unique_chs)
    if len(unique_chs) == 1 and file_count > 0:
        ch = list(unique_chs)[0]
        ch_str = 'Mono' if ch == 1 else 'Stereo' if ch == 2 else f'{ch} canais'
        print(f"✅ Canais uniformes: {ch_str}")
    elif file_count == 0:
        print("Nenhum arquivo encontrado para análise de canais.")
    else:
        channel_descs = ['Mono' if c == 1 else 'Stereo' if c == 2 else f'{c} canais' for c in unique_chs]
        print(f"❌ Incompatibilidade detectada! Contagens de canais diferentes ({', '.join(channel_descs)}).")


def load_audios_and_check_properties(folder_path, label_name=""):
    """
    Carrega individualmente cada arquivo .wav na pasta informada 
    e retorna suas propriedades (SR, canais, duração, tamanho).
    """

    properties = []
    durations = []
    file_sizes_bytes = []
    file_count = 0

    print(f"\n-- Iniciando verificação de: {label_name or folder_path} --")

    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(".wav"):
                fpath = os.path.join(root, fname)
                file_count += 1
                try:
                    # Vê o tamanho bruto do arquivo
                    file_size_bytes = os.path.getsize(fpath)
                    file_sizes_bytes.append(file_size_bytes)
                    file_size_mb = file_size_bytes / (1024 * 1024)

                    # Carrega o arquivo no Librosa
                    y, sr = librosa.load(fpath, sr=None, mono=False)

                    # Librosa checa canais, duração e sample rate
                    channels = y.shape[0] if y.ndim == 2 else 1
                    ch_str = 'Mono' if channels == 1 else 'Stereo' if channels == 2 else f'{channels} canais'
                    duration_sec = librosa.get_duration(y=y, sr=sr)
                    durations.append(duration_sec)
                    properties.append((sr, channels))

                    # Imprime OK com nome do arquivo e propriedades, se desejado
                    if verboseMode:
                        relative_path = os.path.relpath(fpath, folder_path)
                        print(f"🆗 Arquivo {relative_path}: {sr} Hz, {ch_str} ({duration_sec:.2f}s, {file_size_mb:.2f} MB)")

                except Exception as e:
                    print(f"⚠️ Erro ao ler {fpath}: {e}")

    # Retorna dados vazios se nenhum arquivo foi encontrado
    if file_count == 0:
        print("Nenhum arquivo .wav encontrado.")
        return set(), set(), [], [], 0

    unique_srs = set(p[0] for p in properties)
    unique_chs = set(p[1] for p in properties)

    print(f"\n-- Concluída a leitura de {file_count} arquivos na base {label_name}. --")

    # Retorna os dados obtidos
    return unique_srs, unique_chs, durations, file_sizes_bytes, file_count


# Função main, que executa as verificações e imprime relatório final
if __name__ == "__main__":

    srs_true, chs_true, durs_true, sizes_true, count_true = load_audios_and_check_properties(folder_true, label_name="Áudios Reais")
    srs_fake, chs_fake, durs_fake, sizes_fake, count_fake = load_audios_and_check_properties(folder_fake, label_name="Áudios Artificiais")

    print("\n" + "#"*60)
    print("#### RELATÓRIO FINAL: SUMÁRIO DE PROPRIEDADES DAS BASES ####")
    print("#"*60)

    # Sumário / Compatibilidade dos Áudios Reais
    print_compatibility_stats("áudios reais", srs_true, chs_true, count_true)
    print_summary_stats("áudios reais", durs_true, sizes_true)

    # Sumário / Compatibilidade dos Áudios Artificiais
    print_compatibility_stats("áudios artificais", srs_fake, chs_fake, count_fake)
    print_summary_stats("áudios artificiais", durs_fake, sizes_fake)

    # Sumário / Compatibilidade Consolidada
    combined_durs = durs_true + durs_fake
    combined_sizes = sizes_true + sizes_fake
    combined_srs = srs_true.union(srs_fake)
    combined_chs = chs_true.union(chs_fake)
    combined_count = count_true + count_fake
    print_compatibility_stats("base consolidada (real + fake)", combined_srs, combined_chs, combined_count)
    print_summary_stats("base consolidada (real + fake)", combined_durs, combined_sizes)

    # Erro, se não houver arquivos em ambas as pastas
    if combined_count == 0:
        print("Nenhum arquivo encontrado em ambas as pastas.")

    print("\n" + "#"*60)
    print("########################### FIM. ###########################")
    print("#"*60)
