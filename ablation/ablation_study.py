"""
ablation_study.py
7种特征组合的消融实验，每组合跑1次（种子42）
特征维度划分（基于1544维顺序）：
  facemesh: 0:1434
  body_pose: 1434:1533
  headpose: 1533:1536
  bbox: 1536:1544
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
SEED = 42

# 7种特征组合
FEATURE_CONFIGS = {
    'facemesh_only':    [0, 1434],
    'body_pose_only':   [1434, 1533],
    'headpose_only':    [1533, 1536],
    'headpose_body':    [1434, 1536],
    'facemesh_head':    [0, 1434] + [1533, 1536],
    'facemesh_body':    [0, 1533],
    'full':             [0, 1544],
}
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


def select_features(seqs, config):
    """根据配置截取特征维度"""
    if config == 'facemesh_head':
        face = seqs[:, :, 0:1434]
        head = seqs[:, :, 1533:1536]
        return torch.cat([face, head], dim=2)
    else:
        start, end = FEATURE_CONFIGS[config]
        return seqs[:, :, start:end]


def load_split(split_file, config):
    """加载数据，标签2合并到标签1"""
    with open(os.path.join(SPLIT_DIR, split_file), 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    all_seqs, all_labels = [], []
    for fname in files:
        path = os.path.join(SEQUENCES_DIR, fname)
        if os.path.exists(path):
            data = torch.load(path)
            seqs = select_features(data['sequences'], config)
            labels = data['labels'].clone()
            labels[labels == 2] = 1
            all_seqs.append(seqs)
            all_labels.append(labels)

    return torch.cat(all_seqs, dim=0), torch.cat(all_labels, dim=0)


def train_single(config):
    print(f"\n{'='*60}")
    print(f"训练配置: {config}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train = load_split('train_subjects.txt', config)
    X_val, y_val = load_split('val_subjects.txt', config)
    X_test, y_test = load_split('test_subjects.txt', config)

    input_dim = X_train.shape[2]

    train_counts = Counter(y_train.tolist())
    class_weights = torch.tensor([1.5, 0.8], dtype=torch.float).to(DEVICE)

    sample_weights = [2.0 if label == 0 else 1.0 for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = GRUClassifier(input_dim=input_dim).to(DEVICE)
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
            torch.save(model.state_dict(), f"C:\\DIPSER\\ablation_{config}.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(torch.load(f"C:\\DIPSER\\ablation_{config}.pt"))
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

    print(f"结果: 准确率={acc:.4f}, 不参与F1={f1_0:.4f}, "
          f"召回={rec_0:.4f}, 精确={prec_0:.4f}, 参与F1={f1_1:.4f}")

    return {
        'config': config,
        'dim': input_dim,
        'accuracy': acc,
        'f1_0': f1_0,
        'recall_0': rec_0,
        'precision_0': prec_0,
        'f1_1': f1_1
    }


def main():
    print("=" * 60)
    print("消融实验：7种特征组合对比")
    print("=" * 60)

    results = []
    for config in FEATURE_CONFIGS:
        result = train_single(config)
        results.append(result)

    print(f"\n\n{'='*70}")
    print("消融实验汇总")
    print(f"{'='*70}")
    print(f"{'配置':<18} {'维度':>6} {'准确率':>8} {'不参与F1':>10} {'不参与召回':>10} {'不参与精确':>10} {'参与F1':>8}")
    print("-" * 72)
    for r in results:
        print(f"{r['config']:<18} {r['dim']:>6} {r['accuracy']:>8.4f} "
              f"{r['f1_0']:>10.4f} {r['recall_0']:>10.4f} {r['precision_0']:>10.4f} {r['f1_1']:>8.4f}")

    # 保存结果
    with open(r"C:\DIPSER\ablation_summary.txt", 'w') as f:
        f.write("消融实验汇总\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'配置':<18} {'维度':>6} {'准确率':>8} {'不参与F1':>10} {'不参与召回':>10} {'不参与精确':>10} {'参与F1':>8}\n")
        f.write("-" * 72 + "\n")
        for r in results:
            f.write(f"{r['config']:<18} {r['dim']:>6} {r['accuracy']:>8.4f} "
                    f"{r['f1_0']:>10.4f} {r['recall_0']:>10.4f} {r['precision_0']:>10.4f} {r['f1_1']:>8.4f}\n")

    print(f"\n结果已保存到: C:\\DIPSER\\ablation_summary.txt")


if __name__ == '__main__':
    main()
