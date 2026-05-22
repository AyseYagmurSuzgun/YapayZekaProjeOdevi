from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():

    DATA_DIR = Path("D:/Projelerim/TÜBİTAK 2209/Proje/İşlenmiş Veri Final")

    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")
    subjects = np.load(DATA_DIR / "subject_ids.npy")

    # participants.tsv dosyalarını okuma
    alz_part = pd.read_csv(
        r"D:\Projelerim\TÜBİTAK 2209\Proje\Ham Veri\Alzheimer EEG Veri Seti\ds004504\participants.tsv",
        sep="\t"
    )

    pd_part = pd.read_csv(
        r"D:\Projelerim\TÜBİTAK 2209\Proje\Ham Veri\Parkinson EEG Veri Seti\participants.tsv",
        sep="\t"
    )

    alz_part = alz_part.rename(columns={
        "participant_id": "subject",
        "Gender": "gender",
        "Age": "age",
        "MMSE": "mmse"
    })

    pd_part = pd_part.rename(columns={
        "participant_id": "subject",
        "AGE": "age",
        "GENDER": "gender",
        "MOCA": "moca",
        "UPDRS": "updrs"
    })

    def encode_gender(g):
        return 1 if str(g).upper().startswith("M") else 0

    alz_part["gender"] = alz_part["gender"].apply(encode_gender)
    pd_part["gender"] = pd_part["gender"].apply(encode_gender)

    def extract_band_features(X, fs=500):

        freqs = np.fft.rfftfreq(X.shape[2], 1/fs)
        psd = np.abs(np.fft.rfft(X, axis=2)) ** 2

        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta":  (13, 30),
            "gamma": (30, 60)
        }

        features = []

        for _, (fmin, fmax) in bands.items():
            idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
            band_power = psd[:, :, idx].mean(axis=(1, 2))
            features.append(band_power)

        return np.stack(features, axis=1)

    band_features = extract_band_features(X)

    # klinik özellikler
    clinical_features = []

    for subj in subjects:

        subj_clean = subj.replace("ALZ_", "").replace("PD_", "").replace("HC_", "")

        if "ALZ" in subj:
            row = alz_part[alz_part["subject"] == subj_clean]

            if len(row) > 0:
                clinical_features.append([
                    row["age"].values[0],
                    row["gender"].values[0],
                    row["mmse"].values[0],
                    0,
                    0
                ])
            else:
                clinical_features.append([0, 0, 0, 0, 0])

        elif "PD" in subj:
            row = pd_part[pd_part["subject"] == subj_clean]

            if len(row) > 0:
                updrs_val = row["updrs"].values[0]
                if pd.isna(updrs_val):
                    updrs_val = 0

                clinical_features.append([
                    row["age"].values[0],
                    row["gender"].values[0],
                    0,
                    row["moca"].values[0],
                    updrs_val
                ])
            else:
                clinical_features.append([0, 0, 0, 0, 0])

        else:
            clinical_features.append([0, 0, 0, 0, 0])

    clinical_features = np.array(clinical_features)

    X_features = np.concatenate([band_features, clinical_features], axis=1)

    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)

    assert len(X_features) == len(X), "Feature mismatch!"

    print("Band features:", band_features.shape)
    print("Clinical features:", clinical_features.shape)
    print("Final features:", X_features.shape)

    # subject split(%80 train - %20 test)
    unique_subjects = np.unique(subjects)

    train_subj, test_subj = train_test_split(
        unique_subjects,
        test_size=0.2,
        random_state=42
    )

    train_mask = np.isin(subjects, train_subj)
    test_mask = np.isin(subjects, test_subj)

    X_train_eeg = X[train_mask]
    X_test_eeg = X[test_mask]

    X_train_feat = X_features[train_mask]
    X_test_feat = X_features[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    np.save(DATA_DIR / "X_train_eeg.npy", X_train_eeg)
    np.save(DATA_DIR / "X_test_eeg.npy", X_test_eeg)

    np.save(DATA_DIR / "X_train_feat.npy", X_train_feat)
    np.save(DATA_DIR / "X_test_feat.npy", X_test_feat)

    np.save(DATA_DIR / "y_train.npy", y_train)
    np.save(DATA_DIR / "y_test.npy", y_test)

    print("\n✔ HER ŞEY HAZIR")
    print("Train EEG:", X_train_eeg.shape)
    print("Train Features:", X_train_feat.shape)
    print("Test EEG:", X_test_eeg.shape)

if __name__ == "__main__":
    main()