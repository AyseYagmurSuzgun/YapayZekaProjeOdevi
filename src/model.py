import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as FNN
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


DATA_DIR = "D:/Projelerim/TÜBİTAK 2209/Proje/İşlenmiş Veri Final"

X_train = np.load(DATA_DIR + "/X_train_eeg.npy")
X_test  = np.load(DATA_DIR + "/X_test_eeg.npy")

F_train = np.load(DATA_DIR + "/X_train_feat.npy")
F_test  = np.load(DATA_DIR + "/X_test_feat.npy")

y_train = np.load(DATA_DIR + "/y_train.npy")
y_test  = np.load(DATA_DIR + "/y_test.npy")


scaler = StandardScaler()
F_train = scaler.fit_transform(F_train)
F_test = scaler.transform(F_test)

joblib.dump(scaler, "scaler.pkl")


X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)

F_train = torch.tensor(F_train, dtype=torch.float32)
F_test  = torch.tensor(F_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)


train_loader = DataLoader(
    TensorDataset(X_train, F_train, y_train),
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, F_test, y_test),
    batch_size=64
)

# focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = FNN.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

# EEG attention model
class EEG_CNN_Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(14, 32, 7, 2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.attn = nn.Sequential(
            nn.Conv1d(64, 32, 1),
            nn.Tanh(),
            nn.Conv1d(32, 64, 1),
            nn.Sigmoid()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 128)

    def forward(self, x):
        x = self.conv(x)

        attn = self.attn(x)
        x = x * attn

        x = self.pool(x).squeeze(-1)
        x = self.fc(x)

        return x

# tabular model
class TabularMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

# fusion model
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.eeg = EEG_CNN_Attention()
        self.tab = TabularMLP()

        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 3)
        )

    def forward(self, eeg, tab):
        eeg_feat = self.eeg(eeg)
        tab_feat = self.tab(tab)

        x = torch.cat([eeg_feat, tab_feat], dim=1)
        return self.classifier(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FusionModel().to(device)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train.numpy()),
    y=y_train.numpy()
)

weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = FocalLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# train loop
for epoch in range(25):

    model.train()
    total_loss = 0

    for eeg, tab, labels in train_loader:

        eeg = eeg.to(device)
        tab = tab.to(device)
        labels = labels.to(device)

        outputs = model(eeg, tab)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# evulation
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for eeg, tab, labels in test_loader:

        eeg = eeg.to(device)
        tab = tab.to(device)

        outputs = model(eeg, tab)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nCLASSIFICATION REPORT")
print(classification_report(y_true, y_pred))

print("\nF1 SCORE:", f1_score(y_true, y_pred, average="weighted"))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred))


cm = confusion_matrix(y_true, y_pred)

labels = ["Healthy", "Alzheimer", "Parkinson"]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size":12})

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.tight_layout()

# kaydet (POSTER İÇİN ÖNEMLİ)
plt.savefig("confusion_matrix.png", dpi=300)

plt.show()

plt.close()

print("Grafik kaydedildi")
torch.save(model.state_dict(), r"C:\Users\Asus\PycharmProjects\INUFEST\.venv\src\backend\model.pth")
print("Model backend klasörüne kaydedildi")
