import mne
import random
import numpy as np
from pathlib import Path
from mne.preprocessing import ICA, create_eog_epochs

BASE_DIR = Path(r"D:\Projelerim\TÜBİTAK 2209\Proje\Ham Veri") 
ALZ_DIR = BASE_DIR / "Alzheimer EEG Veri Seti" / "ds004504" / "derivatives" 
PARK_DIR = BASE_DIR / "Parkinson EEG Veri Seti" / "ds004584-download"

LABEL_MAP = {
    "HC": 0,
    "ALZ": 1,
    "PD": 2
}
TARGET_SUBJECT_COUNT = 36
EPOCHS_PER_SUBJECT = 60
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ds004504 veri seti alzheimer ve sağlıklı sınıflarının ayrımı
def get_subject_labels_ds004504():
    labels = {}
    for i in range(1, 37):
        labels[f"sub-{i:03d}"] = "ALZ"
    for i in range(37, 66):
        labels[f"sub-{i:03d}"] = "HC"
    return labels

# ds004584 veri seti parkinson ve sağlıklı sınıflarının ayrımı
def get_subject_labels_ds004584():
    labels = {}
    for i in range(1, 101):
        labels[f"sub-{i:03d}"] = "PD"
    for i in range(101, 150):
        labels[f"sub-{i:03d}"] = "HC"
    return labels

# Band-pass filtre
def preprocess_raw(raw):
    raw.filter(l_freq=1.0, h_freq=40.0)
    raw.set_eeg_reference("average")
    return raw

# 2 saniyelik epoch lar oluşturma
def create_epochs(raw, epoch_length=2.0):
    return mne.make_fixed_length_epochs(
        raw,
        duration=epoch_length,
        overlap=0.0,
        preload=True
    )

# z-score normalizasyonu
def zscore_epochs(epochs):
    data = epochs.get_data()
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True) + 1e-8
    data = (data - mean) / std
    epochs._data = data
    return epochs

# ICA
def apply_ica(raw, random_state=42):

    n_components = len(raw.ch_names) - 1

    ica = ICA(
        n_components=n_components,
        method='fastica',
        random_state=random_state,
        max_iter='auto'
    )

    ica.fit(raw)
    raw_clean = ica.apply(raw.copy())

    return raw_clean

# Ortak kanalları bulma
def get_common_channels():
    alz_file = next(ALZ_DIR.rglob("*.set"))
    park_file = next(PARK_DIR.rglob("*.set"))

    raw_alz = mne.io.read_raw_eeglab(alz_file, preload=False, verbose=False)
    raw_park = mne.io.read_raw_eeglab(park_file, preload=False, verbose=False)

    common = list(set(raw_alz.ch_names) & set(raw_park.ch_names))
    common.sort()

    print(f"Ortak kanal sayısı: {len(common)}")
    return common

# Veri setlerini ön işleme
def process_dataset(folder_path, common_channels, label_map, target_label, use_ica=True):
    subject_epochs = {}
    eeg_files = list(folder_path.rglob("*.set"))

    for eeg_file in eeg_files:
        subject_id = eeg_file.parts[-3]

        if subject_id not in label_map:
            continue

        if label_map[subject_id] != target_label:
            continue

        raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
        raw.pick(common_channels)
        raw = preprocess_raw(raw)

        if use_ica:
            raw = apply_ica(raw)

        epochs = create_epochs(raw)
        epochs = zscore_epochs(epochs)
        subject_epochs[subject_id] = epochs.get_data()

    return subject_epochs

def subject_to_epochs(subject_dict, label):

    X, y, subjects = [], [], []

    for subj, epochs in subject_dict.items():

        for ep in epochs:
            X.append(ep)
            y.append(label)
            subjects.append(subj)

    return X, y, subjects

# Ana fonksiyon
def run_preprocessing(use_ica=True):

    common_channels = get_common_channels()

    labels_504 = get_subject_labels_ds004504()
    labels_584 = get_subject_labels_ds004584()

    alz_dict = process_dataset(ALZ_DIR, common_channels, labels_504, "ALZ", use_ica)
    pd_dict = process_dataset(PARK_DIR, common_channels, labels_584, "PD", use_ica)

    hc_504 = process_dataset(ALZ_DIR, common_channels, labels_504, "HC", use_ica)
    hc_584 = process_dataset(PARK_DIR, common_channels, labels_584, "HC", use_ica)

    healthy_dict = {**hc_504, **hc_584}

    print("\nÖN İŞLEME ÖZETİ")
    print("Alzheimer:", len(alz_dict))
    print("Parkinson:", len(pd_dict))
    print("Healthy:", len(healthy_dict))

    alz_X, alz_y, alz_s = subject_to_epochs(alz_dict, 1)
    pd_X, pd_y, pd_s = subject_to_epochs(pd_dict, 2)
    hc_X, hc_y, hc_s = subject_to_epochs(healthy_dict, 0)

    X = np.array(hc_X + alz_X + pd_X)
    y = np.array(hc_y + alz_y + pd_y)
    subjects = np.array(hc_s + alz_s + pd_s)

    print("\nFINAL DATASET")
    print("Samples:", len(X))
    print("Classes:", np.unique(y, return_counts=True))
    print("Subjects:", len(set(subjects)))


    channel_names = np.array(common_channels)

    return X, y, subjects, channel_names