"""
verify_stability.py
headpose_only 稳定性验证（3个随机种子）
验证3维头部姿态特征的消融结果是否可复现
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np

# ==================== 配置 ====================
SEQUENCES_DIR = r"D:\DIPSER\sequences"
SPLIT_DIR = r"C:\DIPSER"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 8
SEEDS = [42, 123, 2024]

# headpose特征位置：1544维中的第1533-1535维
HEADPOSE_START = 1533
HEADPOSE_END = 1536
# =============================================


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


def load_split_headpose(split_file):
    """只加载headpose维度（1533:1536）"""
    with open(os.path.join(SPLIT_DIR, split_file), 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    all_seqs, all_labels = [], []
    for fname in files:
        path = os.path.join(SEQUENCES_DIR, fname)
        if os.path.exists(path):
            data = torch.load(path)
            seqs = data['sequences'][:, :, HEADPOSE_START:HEADPOSE_END]
            labels = data['labels'].clone()
            labels[labels == 2] = 1
            all_seqs.append(seqs)
            all_labels.append(labels)

    return torch.cat(all_seqs, dim=0), torch.cat(all_labels, dim=0)


def run_once(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = load_split_headpose('train_subjects.txt')
    X_val, y_val = load_split_headpose('val_subjects.txt')
    X_test, y_test = load_split_headpose('test_subjects.txt')

    train_counts = Counter(y_train.tolist())
    class_weights = torch.tensor([1.5, 0.8], dtype=torch.float).to(DEVICE)

    sample_weights = [2.0 if label == 0 else 1.0 for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = GRUClassifier(input_dim=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
            torch.save(model.state_dict(), f"C:\\DIPSER\\headpose_verify_seed{seed}.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(torch.load(f"C:\\DIPSER\\headpose_verify_seed{seed}.pt"))
    model.eval()

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds.extend(model(x).argmax(1).cpu().tolist())
            labels.extend(y.cpu().tolist())

    acc = accuracy_score(labels, preds)
    f1_0 = f1_score(labels, preds, pos_label=0)
    rec_0 = recall_score(labels, preds, pos_label=0)
    prec_0 = precision_score(labels, preds, pos_label=0)
    f1_1 = f1_score(labels, preds, pos_label=1)

    print(f"Seed {seed}: 准确率={acc:.4f}, 不参与F1={f1_0:.4f}, "
          f"召回={rec_0:.4f}, 精确={prec_0:.4f}, 参与F1={f1_1:.4f}")

    return [acc, f1_0, rec_0, prec_0, f1_1]


def main():
    print("=" * 60)
    print("headpose_only 稳定性验证（3个种子）")
    print(f"特征: 头部姿态 pitch/yaw/roll (3维)")
    print("=" * 60)

    all_results = []
    for seed in SEEDS:
        all_results.append(run_once(seed))

    results = np.array(all_results)

    print(f"\n平均结果:")
    print(f"  准确率:    {results[:,0].mean():.4f} ± {results[:,0].std():.4f}")
    print(f"  不参与F1:  {results[:,1].mean():.4f} ± {results[:,1].std():.4f}")
    print(f"  不参与召回: {results[:,2].mean():.4f} ± {results[:,2].std():.4f}")
    print(f"  不参与精确: {results[:,3].mean():.4f} ± {results[:,3].std():.4f}")
    print(f"  参与F1:    {results[:,4].mean():.4f} ± {results[:,4].std():.4f}")

    # 保存
    with open(r"C:\DIPSER\headpose_stability.txt", 'w') as f:
        f.write("headpose_only 稳定性验证\n")
        f.write("=" * 50 + "\n")
        for seed, res in zip(SEEDS, all_results):
            f.write(f"Seed {seed}: 准确率={res[0]:.4f}, 不参与F1={res[1]:.4f}, "
                    f"召回={res[2]:.4f}, 精确={res[3]:.4f}, 参与F1={res[4]:.4f}\n")
        f.write("\n平均:\n")
        f.write(f"  准确率:    {results[:,0].mean():.4f} ± {results[:,0].std():.4f}\n")
        f.write(f"  不参与F1:  {results[:,1].mean():.4f} ± {results[:,1].std():.4f}\n")
        f.write(f"  不参与召回: {results[:,2].mean():.4f} ± {results[:,2].std():.4f}\n")
        f.write(f"  不参与精确: {results[:,3].mean():.4f} ± {results[:,3].std():.4f}\n")
        f.write(f"  参与F1:    {results[:,4].mean():.4f} ± {results[:,4].std():.4f}\n")

    print(f"\n结果已保存到: C:\\DIPSER\\headpose_stability.txt")


if __name__ == '__main__':
    main()
