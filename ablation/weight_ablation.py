"""类别权重与重采样策略消融（全特征1544维）"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter
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
SEED = 42

CONFIGS = {
    'A_无权重无重采样': {'use_class_weights': False, 'use_sampler': False},
    'B_仅类别权重': {'use_class_weights': True, 'use_sampler': False},
    'C_仅重采样': {'use_class_weights': False, 'use_sampler': True},
    'D_权重加重采样': {'use_class_weights': True, 'use_sampler': True},
}


class GRUClassifier(nn.Module):
    def __init__(self, input_dim=1544, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
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


def load_split(split_file):
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


def run_once(config_name, use_cw, use_sampler):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train = load_split('train_subjects.txt')
    X_val, y_val = load_split('val_subjects.txt')
    X_test, y_test = load_split('test_subjects.txt')

    # 损失函数
    if use_cw:
        train_counts = Counter(y_train.tolist())
        class_weights = torch.tensor([1.5, 0.8], dtype=torch.float).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # DataLoader
    if use_sampler:
        sample_weights = [2.0 if label == 0 else 1.0 for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = GRUClassifier(input_dim=1544).to(DEVICE)
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
            torch.save(model.state_dict(), f"C:\\DIPSER\\weight_ablation_{config_name}.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(torch.load(f"C:\\DIPSER\\weight_ablation_{config_name}.pt"))

    # 验证集选阈值
    val_probs, val_labels = get_probs(model, val_loader)
    best_thresh = 0.5
    best_val_f1 = 0
    for thresh in np.arange(0.20, 0.81, 0.02):
        preds = (val_probs >= thresh).astype(int)
        f1 = f1_score(val_labels, preds, pos_label=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = thresh

    # 测试集评估
    test_probs, test_labels = get_probs(
        model, DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    )
    preds = (test_probs >= best_thresh).astype(int)

    acc = accuracy_score(test_labels, preds)
    f1_0 = f1_score(test_labels, preds, pos_label=0)
    rec_0 = recall_score(test_labels, preds, pos_label=0)
    prec_0 = precision_score(test_labels, preds, pos_label=0)
    f1_1 = f1_score(test_labels, preds, pos_label=1)

    print(f"{config_name}: 阈值={best_thresh:.2f}, 准确率={acc:.4f}, "
          f"不参与F1={f1_0:.4f}, 召回={rec_0:.4f}, 精确={prec_0:.4f}, 参与F1={f1_1:.4f}")

    return [acc, f1_0, rec_0, prec_0, f1_1, best_thresh]


print("权重与重采样消融（全特征1544维，seed 42）")
print("=" * 50)

results = {}
for name, cfg in CONFIGS.items():
    results[name] = run_once(name, cfg['use_class_weights'], cfg['use_sampler'])

print(f"\n汇总:")
print(f"{'配置':<20} {'准确率':>8} {'不参与F1':>10} {'召回':>8} {'精确':>8} {'参与F1':>8}")
print("-" * 65)
for name, r in results.items():
    print(f"{name:<20} {r[0]:>8.4f} {r[1]:>10.4f} {r[2]:>8.4f} {r[3]:>8.4f} {r[4]:>8.4f}")

with open(r"C:\DIPSER\weight_ablation_results.txt", 'w') as f:
    f.write("权重与重采样消融（全特征1544维，seed 42）\n")
    f.write("=" * 50 + "\n\n")
    for name, r in results.items():
        f.write(f"{name}:\n")
        f.write(f"  准确率: {r[0]:.4f}\n")
        f.write(f"  不参与F1: {r[1]:.4f}\n")
        f.write(f"  不参与召回: {r[2]:.4f}\n")
        f.write(f"  不参与精确: {r[3]:.4f}\n")
        f.write(f"  参与F1: {r[4]:.4f}\n")
        f.write(f"  最优阈值: {r[5]:.2f}\n\n")

print(f"\n结果已保存: C:\\DIPSER\\weight_ablation_results.txt")
