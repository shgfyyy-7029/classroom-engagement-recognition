"""LR和MLP三种子验证"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np
from collections import Counter

SEQUENCES_DIR = r"D:\DIPSER\sequences"
SPLIT_DIR = r"C:\DIPSER"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 2024]
BATCH_SIZE = 64
EPOCHS = 30


def load_split_mean(split_file):
    with open(os.path.join(SPLIT_DIR, split_file), 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    all_feats, all_labels = [], []
    for fname in files:
        path = os.path.join(SEQUENCES_DIR, fname)
        if os.path.exists(path):
            data = torch.load(path)
            mean_feats = data['sequences'].mean(dim=1)
            labels = data['labels'].clone()
            labels[labels == 2] = 1
            all_feats.append(mean_feats)
            all_labels.append(labels)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def find_best_threshold(probs, labels):
    best_thresh, best_f1 = 0.5, 0
    for thresh in np.arange(0.20, 0.81, 0.02):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, pos_label=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


class MLP(nn.Module):
    def __init__(self, input_dim=1544, hidden=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)


# 加载并标准化数据（只做一次）
print("加载数据...")
X_train, y_train = load_split_mean('train_subjects.txt')
X_val, y_val = load_split_mean('val_subjects.txt')
X_test, y_test = load_split_mean('test_subjects.txt')

scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train.numpy())
X_val_np = scaler.transform(X_val.numpy())
X_test_np = scaler.transform(X_test.numpy())

y_train_np = y_train.numpy()
y_val_np = y_val.numpy()
y_test_np = y_test.numpy()


# ==================== LR ====================
print("\n" + "=" * 50)
print("LR 三种子")
print("=" * 50)

lr_results = []
for seed in SEEDS:
    lr = LogisticRegression(max_iter=2000, random_state=seed, class_weight=None)
    lr.fit(X_train_np, y_train_np)

    val_probs = lr.predict_proba(X_val_np)[:, 1]
    test_probs = lr.predict_proba(X_test_np)[:, 1]

    best_thresh = find_best_threshold(val_probs, y_val_np)
    preds = (test_probs >= best_thresh).astype(int)

    acc = accuracy_score(y_test_np, preds)
    f1_0 = f1_score(y_test_np, preds, pos_label=0)
    rec_0 = recall_score(y_test_np, preds, pos_label=0)
    prec_0 = precision_score(y_test_np, preds, pos_label=0)
    f1_1 = f1_score(y_test_np, preds, pos_label=1)

    print(f"Seed {seed}: 阈值={best_thresh:.2f}, 准确率={acc:.4f}, "
          f"不参与F1={f1_0:.4f}, 召回={rec_0:.4f}, 精确={prec_0:.4f}")
    lr_results.append([acc, f1_0, rec_0, prec_0, f1_1])

lr_results = np.array(lr_results)
print(f"\nLR 平均:")
print(f"  准确率:     {lr_results[:,0].mean():.4f} ± {lr_results[:,0].std():.4f}")
print(f"  不参与F1:   {lr_results[:,1].mean():.4f} ± {lr_results[:,1].std():.4f}")
print(f"  不参与召回: {lr_results[:,2].mean():.4f} ± {lr_results[:,2].std():.4f}")
print(f"  不参与精确: {lr_results[:,3].mean():.4f} ± {lr_results[:,3].std():.4f}")


# ==================== MLP ====================
print("\n" + "=" * 50)
print("MLP 三种子")
print("=" * 50)

X_train_t = torch.FloatTensor(X_train_np)
X_val_t = torch.FloatTensor(X_val_np)
X_test_t = torch.FloatTensor(X_test_np)

mlp_results = []
for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(x), y).item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"C:\\DIPSER\\mlp_seed{seed}.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 8:
                break

    model.load_state_dict(torch.load(f"C:\\DIPSER\\mlp_seed{seed}.pt"))
    model.eval()

    with torch.no_grad():
        val_probs = torch.softmax(model(X_val_t.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
        test_probs = torch.softmax(model(X_test_t.to(DEVICE)), dim=1)[:, 1].cpu().numpy()

    best_thresh = find_best_threshold(val_probs, y_val_np)
    preds = (test_probs >= best_thresh).astype(int)

    acc = accuracy_score(y_test_np, preds)
    f1_0 = f1_score(y_test_np, preds, pos_label=0)
    rec_0 = recall_score(y_test_np, preds, pos_label=0)
    prec_0 = precision_score(y_test_np, preds, pos_label=0)
    f1_1 = f1_score(y_test_np, preds, pos_label=1)

    print(f"Seed {seed}: 阈值={best_thresh:.2f}, 准确率={acc:.4f}, "
          f"不参与F1={f1_0:.4f}, 召回={rec_0:.4f}, 精确={prec_0:.4f}")
    mlp_results.append([acc, f1_0, rec_0, prec_0, f1_1])

mlp_results = np.array(mlp_results)
print(f"\nMLP 平均:")
print(f"  准确率:     {mlp_results[:,0].mean():.4f} ± {mlp_results[:,0].std():.4f}")
print(f"  不参与F1:   {mlp_results[:,1].mean():.4f} ± {mlp_results[:,1].std():.4f}")
print(f"  不参与召回: {mlp_results[:,2].mean():.4f} ± {mlp_results[:,2].std():.4f}")
print(f"  不参与精确: {mlp_results[:,3].mean():.4f} ± {mlp_results[:,3].std():.4f}")
