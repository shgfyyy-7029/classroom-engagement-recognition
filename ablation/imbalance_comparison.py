"""
类别不平衡处理策略对比（完整版）
四种配置各3种子，验证集选阈值
配置A：无处理
配置B：标准类别权重（反频率公式）
配置C：2倍重采样
配置D：标准类别权重+2倍重采样
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from collections import Counter
import numpy as np

# ==================== 配置 ====================
SEQUENCES_DIR = r"D:\DIPSER\sequences"
SPLIT_DIR = r"C:\DIPSER"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 10
SEEDS = [42, 123, 2024]

CONFIGS = {
    'A_无处理':      {'use_class_weights': False, 'use_sampler': False},
    'B_标准权重':    {'use_class_weights': True,  'use_sampler': False},
    'C_2倍重采样':   {'use_class_weights': False, 'use_sampler': True},
    'D_权重加重采样': {'use_class_weights': True,  'use_sampler': True},
}
# =============================================


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


def find_best_threshold(probs, labels):
    best_thresh, best_f1 = 0.5, 0
    for thresh in np.arange(0.20, 0.81, 0.02):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, pos_label=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def run_once(config_name, use_cw, use_sampler, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = load_split('train_subjects.txt')
    X_val, y_val = load_split('val_subjects.txt')
    X_test, y_test = load_split('test_subjects.txt')

    # 损失函数
    if use_cw:
        train_counts = Counter(y_train.tolist())
        total = sum(train_counts.values())
        w0 = total / (2 * train_counts[0])
        w1 = total / (2 * train_counts[1])
        class_weights = torch.tensor([w0, w1], dtype=torch.float).to(DEVICE)
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
            torch.save(model.state_dict(), f"C:\\DIPSER\\imb_{config_name}_seed{seed}.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(torch.load(f"C:\\DIPSER\\imb_{config_name}_seed{seed}.pt"))

    val_probs, val_labels = get_probs(model, val_loader)
    best_thresh = find_best_threshold(val_probs, val_labels)

    test_probs, test_labels = get_probs(
        model, DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    )
    preds = (test_probs >= best_thresh).astype(int)

    acc = accuracy_score(test_labels, preds)
    f1_0 = f1_score(test_labels, preds, pos_label=0)
    rec_0 = recall_score(test_labels, preds, pos_label=0)
    prec_0 = precision_score(test_labels, preds, pos_label=0)
    auc = roc_auc_score(test_labels, test_probs)

    print(f"  {config_name} seed{seed}: 阈值={best_thresh:.2f}, 准确率={acc:.4f}, "
          f"不参与F1={f1_0:.4f}, 召回={rec_0:.4f}, 精确={prec_0:.4f}, AUC={auc:.4f}")
    return [acc, f1_0, rec_0, prec_0, auc]


def main():
    print("=" * 70)
    print("类别不平衡处理策略对比（4配置×3种子）")
    print("=" * 70)

    all_results = {}

    for config_name, cfg in CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"配置: {config_name}")
        print(f"{'='*50}")
        results = []
        for seed in SEEDS:
            results.append(run_once(config_name, cfg['use_class_weights'], cfg['use_sampler'], seed))
        all_results[config_name] = np.array(results)

    print(f"\n\n{'='*90}")
    print("最终汇总")
    print(f"{'='*90}")
    print(f"{'配置':<18} {'准确率':>14} {'不参与F1':>14} {'不参与召回':>14} {'不参与精确':>14} {'AUC':>12}")
    print("-" * 95)

    with open(r"C:\DIPSER\imbalance_comparison_results.txt", 'w') as f:
        f.write("类别不平衡处理策略对比结果\n\n")
        for config_name, results in all_results.items():
            means = results.mean(axis=0)
            stds = results.std(axis=0)
            print(f"{config_name:<18} {means[0]:.4f}±{stds[0]:.4f} {means[1]:.4f}±{stds[1]:.4f} "
                  f"{means[2]:.4f}±{stds[2]:.4f} {means[3]:.4f}±{stds[3]:.4f} {means[4]:.4f}±{stds[4]:.4f}")
            f.write(f"{config_name}:\n")
            f.write(f"  准确率: {means[0]:.4f} ± {stds[0]:.4f}\n")
            f.write(f"  不参与F1: {means[1]:.4f} ± {stds[1]:.4f}\n")
            f.write(f"  不参与召回: {means[2]:.4f} ± {stds[2]:.4f}\n")
            f.write(f"  不参与精确: {means[3]:.4f} ± {stds[3]:.4f}\n")
            f.write(f"  AUC: {means[4]:.4f} ± {stds[4]:.4f}\n\n")

    print(f"\n结果已保存: C:\\DIPSER\\imbalance_comparison_results.txt")


if __name__ == '__main__':
    main()
