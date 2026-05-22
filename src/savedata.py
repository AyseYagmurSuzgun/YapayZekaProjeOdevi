import numpy as np
from pathlib import Path
from src.preprocess import run_preprocessing

print("Preprocessing...")

X, y, subjects, channel_names = run_preprocessing()

X = X.astype(np.float32)

perm = np.random.permutation(len(X))
X, y, subjects = X[perm], y[perm], subjects[perm]

SAVE_DIR = Path("D:/Projelerim/TÜBİTAK 2209/Proje/İşlenmiş Veri Final")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

np.save(SAVE_DIR / "X.npy", X)
np.save(SAVE_DIR / "y.npy", y)
np.save(SAVE_DIR / "subject_ids.npy", subjects)
np.save(SAVE_DIR / "channel_names.npy", channel_names)

print("DONE")
print(X.shape, y.shape)