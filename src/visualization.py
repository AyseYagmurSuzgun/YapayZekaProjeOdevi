from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = Path("D:/Projelerim/TÜBİTAK 2209/Proje/İşlenmiş Veri Final")

X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")
channel_names = np.load(DATA_DIR / "channel_names.npy").astype(str)

output_dir = "EEG_Channel_Plots"
os.makedirs(output_dir, exist_ok=True)

class_labels = {0: "Healthy", 1: "Alzheimer", 2: "Parkinson"}

for ch in range(len(channel_names)):

    plt.figure(figsize=(12, 4))

    for label, name in class_labels.items():

        data = X[y == label][:, ch, :]

        signal = np.median(data, axis=0)

        plt.plot(signal, label=name)

    plt.title(f"EEG Kanal: {channel_names[ch]}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{channel_names[ch]}_median_signal.png"))
    plt.close()

print(f"Grafikler '{output_dir}' klasörüne kaydedildi.")