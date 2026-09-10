"""窗口长度对比实验：5帧/20帧（10帧已有结果）"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np
from collections import Counter

PROCESSED_DIR = r"D:\DIPSER\dipser_processed"
SPLIT_DIR = r"C:\DIPSER"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 8
SEEDS = [42, 123, 2024]

WINDOW_CONFIGS = {
    'window_5': {'window': 5, 'stride': 3},
    'window_20': {'window': 20, 'stride': 10},
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


def build_sequences(features, labels, window, stride):
    """滑动窗口构建序列，窗口内标签一致才保留"""
    n = features.shape[0]
    if n < window:
        return None, None

    seqs, labs = [], []
    for start in range(0, n - window + 1, stride):
        end = start + window
        w_labels = labels[start:end]
        if torch.all(w_labels == w_labels[0]):
            seqs.append(features[start:end])
            labs.append(w_labels[0].item())

    if len(seqs) == 0:
        return None, None
    return torch.stack(seqs), torch.tensor(labs, dtype=torch.long)


def load_split(split_file, window, stride):
    """从逐帧数据构建指定窗口的序列"""
    with open(os.path.join(SPLIT_DIR, split_file), 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    all_seqs, all_labels = [], []
    for fname in files:
        path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            continue
        data = torch.load(path)
        features = data['features']
        labels = data['labels'].clone()
        labels[labels == 2] = 1

        seqs, labs = build_sequences(features, labels, window, stride)
        if seqs is not None:
            all_seqs.append(seqs)
            all_labels.append(labs)

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


def run_once(window_name, window, stride, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = load_split('train_subjects.txt', window, stride)
    X_val, y_val = load_split('val_subjects.txt', window, stride)
    X_test, y_test = load_split('test_subjects.txt', window, stride)

    print(f"    {window_name} seed{seed}: 训练{X_train.shape}, 验证{X_val.shape}, 测试{X_test.shape}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = GRUClassifier().to(DEVICE)
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
            torch.save(model.state_dict(), f"C:\\DIPSER\\{window_name}_seed{seed}.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(torch.load(f"C:\\DIPSER\\{window_name}_seed{seed}.pt"))

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
    f1_1 = f1_score(test_labels, preds, pos_label=1)

    print(f"    {window_name} seed{seed}: 阈值={best_thresh:.2f}, 准确率={acc:.4f}, "
          f"不参与F1={f1_0:.4f}, 召回={rec_0:.4f}, 精确={prec_0:.4f}")
    return [acc, f1_0, rec_0, prec_0, f1_1]


def main():
    print("=" * 60)
    print("窗口长度对比实验（5帧 / 20帧）")
    print("=" * 60)

    all_results = {}
    for window_name, cfg in WINDOW_CONFIGS.items():
        print(f"\n{'='*40}")
        print(f"{window_name}: 窗口={cfg['window']}, 步长={cfg['stride']}")
        print(f"{'='*40}")
        results = []
        for seed in SEEDS:
            results.append(run_once(window_name, cfg['window'], cfg['stride'], seed))
        all_results[window_name] = np.array(results)

    print(f"\n\n{'='*70}")
    print("最终汇总")
    print(f"{'='*70}")
    print(f"{'配置':<15} {'准确率':>14} {'不参与F1':>14} {'召回率':>14} {'精确率':>14}")
    print("-" * 75)

    for window_name, results in all_results.items():
        means = results.mean(axis=0)
        stds = results.std(axis=0)
        print(f"{window_name:<15} {means[0]:.4f}±{stds[0]:.4f} {means[1]:.4f}±{stds[1]:.4f} "
              f"{means[2]:.4f}±{stds[2]:.4f} {means[3]:.4f}±{stds[3]:.4f}")

    print(f"\n{'10帧(已有)':<15} {'0.7721±0.0071':>14} {'0.4979±0.0113':>14} {'0.5352±0.0407':>14} {'0.4676±0.0117':>14}")

    with open(r"C:\DIPSER\window_comparison.txt", 'w') as f:
        f.write("窗口长度对比实验\n")
        f.write("=" * 60 + "\n\n")
        for window_name, results in all_results.items():
            means = results.mean(axis=0)
            stds = results.std(axis=0)
            f.write(f"{window_name}:\n")
            f.write(f"  准确率: {means[0]:.4f} ± {stds[0]:.4f}\n")
            f.write(f"  不参与F1: {means[1]:.4f} ± {stds[1]:.4f}\n")
            f.write(f"  不参与召回: {means[2]:.4f} ± {stds[2]:.4f}\n")
            f.write(f"  不参与精确: {means[3]:.4f} ± {stds[3]:.4f}\n\n")
        f.write("10帧(已有):\n")
        f.write("  准确率: 0.7721 ± 0.0071\n")
        f.write("  不参与F1: 0.4979 ± 0.0113\n")
        f.write("  不参与召回: 0.5352 ± 0.0407\n")
        f.write("  不参与精确: 0.4676 ± 0.0117\n")

    print(f"\n结果已保存: C:\\DIPSER\\window_comparison.txt")


if __name__ == '__main__':
    main()
