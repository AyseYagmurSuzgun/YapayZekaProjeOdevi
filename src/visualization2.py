from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("D:/Projelerim/TÜBİTAK 2209/Proje/İşlenmiş Veri Final")

X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")
channel_names = np.load(DATA_DIR / "channel_names.npy")

fs = 500
n_samples = X.shape[2]

alz = X[y == 1]
park = X[y == 2]
healthy = X[y == 0]

def compute_psd(data):
    psd = np.abs(np.fft.rfft(data, axis=2)) ** 2
    return psd.mean(axis=0)   # (channels, freq)

alz_psd = compute_psd(alz)
park_psd = compute_psd(park)
healthy_psd = compute_psd(healthy)

freqs = np.fft.rfftfreq(n_samples, 1/fs)

# EEG BANDS
bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 60)
}

n_channels = X.shape[1]

def extract_band_matrix(psd_data):

    band_matrix = np.zeros((n_channels, len(bands)))

    for i, (band, (fmin, fmax)) in enumerate(bands.items()):
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        band_matrix[:, i] = psd_data[:, idx].mean(axis=1)

    return band_matrix

alz_band = extract_band_matrix(alz_psd)
park_band = extract_band_matrix(park_psd)
healthy_band = extract_band_matrix(healthy_psd)

fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle("EEG Band Power Comparison", fontsize=16)

# Alzheimer
sns.heatmap(
    alz_band,
    cmap="Oranges",
    xticklabels=bands.keys(),
    yticklabels=channel_names,
    annot=True,
    fmt=".2f",
    ax=axes[0]
)
axes[0].set_title("Alzheimer")

# Parkinson
sns.heatmap(
    park_band,
    cmap="Greens",
    xticklabels=bands.keys(),
    yticklabels=False,
    annot=True,
    fmt=".2f",
    ax=axes[1]
)
axes[1].set_title("Parkinson")

# Healthy
sns.heatmap(
    healthy_band,
    cmap="Blues",
    xticklabels=bands.keys(),
    yticklabels=False,
    annot=True,
    fmt=".2f",
    ax=axes[2]
)
axes[2].set_title("Healthy")

plt.tight_layout()
plt.show()