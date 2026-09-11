"""有监督特征选择实验：用互信息选出Top-N维度，对比全特征"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np

SEQUENCES_DIR = r"D:\DIPSER\sequences"
SPLIT_DIR = r"C:\DIPSER"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 8
SEEDS = [42, 123, 2024]
TOP_N_LIST = [10, 50, 200]


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


def load_split_seq(split_file):
    with open(os.path.join(SPLIT_DIR, split_file), 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    all_seqs, all_labels = [], []
    for fname in files:
        path = os.path.join(SEQUENCES_DIR, fname)
        if os.path.exists(path):
            data = torch.load(path)
            labels = data['labels'].clone()
            labels[labels == 2] = 1
            all_seqs.append(data['sequences'])
            all_labels.append(labels)
    return torch.cat(all_seqs, dim=0), torch.cat(all_labels, dim=0)


def get_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            p = torch.softmax(outputs, dim=1)[:, 1]
            probs.extend(p.cpu().tolist())
            labels.extend(y.cpu().tolist())
    return np.array(probs), np.array(labels)


def find_best_threshold(probs, labels):
    best_thresh, best_f1 = 0.5, 0
    for thresh in np.arange(0.20, 0.81, 0.02):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, pos_label=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


# ==================== 加载数据 ====================
print("加载10帧序列数据...")
X_train, y_train = load_split_seq('train_subjects.txt')
X_val, y_val = load_split_seq('val_subjects.txt')
X_test, y_test = load_split_seq('test_subjects.txt')

print(f"训练集: {X_train.shape}")

# ==================== 互信息特征选择 ====================
print("\n计算互信息（用训练集10帧均值）...")
X_train_mean = X_train.mean(dim=1).numpy()
y_train_np = y_train.numpy()

mi_scores = mutual_info_classif(X_train_mean, y_train_np, random_state=42)
top_indices = np.argsort(mi_scores)[::-1]  # 按互信息降序排列

print(f"互信息最高/最低值: {mi_scores.max():.4f} / {mi_scores.min():.4f}")
print(f"互信息Top-10维度索引: {top_indices[:10]}")
print(f"互信息Top-10得分: {mi_scores[top_indices[:10]]}")


def run_once(top_n, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    selected = np.array(top_indices[:top_n]).copy()

    X_train_sel = X_train[:, :, selected]
    X_val_sel = X_val[:, :, selected]
    X_test_sel = X_test[:, :, selected]

    train_loader = DataLoader(TensorDataset(X_train_sel, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_sel, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = GRUClassifier(input_dim=top_n).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
            torch.save(model.state_dict(), f"C:\\DIPSER\\mi{top_n}_seed{seed}.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(torch.load(f"C:\\DIPSER\\mi{top_n}_seed{seed}.pt"))

    val_probs, val_labels = get_probs(model, val_loader)
    best_thresh = find_best_threshold(val_probs, val_labels)

    test_probs, test_labels = get_probs(
        model, DataLoader(TensorDataset(X_test_sel, y_test), batch_size=BATCH_SIZE, shuffle=False)
    )
    preds = (test_probs >= best_thresh).astype(int)

    acc = accuracy_score(test_labels, preds)
    f1_0 = f1_score(test_labels, preds, pos_label=0)
    rec_0 = recall_score(test_labels, preds, pos_label=0)
    prec_0 = precision_score(test_labels, preds, pos_label=0)

    print(f"  MI-{top_n} seed{seed}: 阈值={best_thresh:.2f}, 准确率={acc:.4f}, "
          f"不参与F1={f1_0:.4f}, 召回={rec_0:.4f}, 精确={prec_0:.4f}")
    return [acc, f1_0, rec_0, prec_0]


# ==================== 主实验 ====================
print(f"\n{'='*60}")
print("互信息特征选择实验")
print(f"{'='*60}")

all_results = {}
for top_n in TOP_N_LIST:
    print(f"\nMI-{top_n}:")
    results = []
    for seed in SEEDS:
        results.append(run_once(top_n, seed))
    all_results[top_n] = np.array(results)

# ==================== 汇总 ====================
print(f"\n\n{'='*70}")
print("汇总")
print(f"{'='*70}")
print(f"{'配置':<15} {'准确率':>14} {'不参与F1':>14} {'召回率':>14} {'精确率':>14}")
print("-" * 75)

for top_n, results in all_results.items():
    means = results.mean(axis=0)
    stds = results.std(axis=0)
    print(f"MI-{top_n:<12} {means[0]:.4f}±{stds[0]:.4f} {means[1]:.4f}±{stds[1]:.4f} "
          f"{means[2]:.4f}±{stds[2]:.4f} {means[3]:.4f}±{stds[3]:.4f}")

print(f"\n{'PCA-10':<15} {'0.6839±0.0125':>14} {'0.3703±0.0219':>14} {'0.4390±0.0241':>14} {'0.3203±0.0199':>14}")
print(f"{'全特征1544':<15} {'0.7721±0.0071':>14} {'0.4979±0.0113':>14} {'0.5352±0.0407':>14} {'0.4676±0.0117':>14}")
