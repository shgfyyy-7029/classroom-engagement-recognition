"""消融实验（正确版）：7配置×3种子，验证集选阈值"""
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
SEEDS = [42, 123, 2024]

FEATURE_CONFIGS = {
    'facemesh_only': [0, 1434],
    'body_pose_only': [1434, 1533],
    'headpose_only': [1533, 1536],
    'facemesh_body': [0, 1533],
    'facemesh_head': [0, 1434, 1533, 1536],
    'headpose_body': [1434, 1536],
    'full': [0, 1544],
}


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


def load_split(split_file, config_name):
    with open(os.path.join(SPLIT_DIR, split_file), 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    all_seqs, all_labels = [], []
    for fname in files:
        path = os.path.join(SEQUENCES_DIR, fname)
        if os.path.exists(path):
            data = torch.load(path)
            seqs = data['sequences']
            if config_name == 'facemesh_head':
                face = seqs[:, :, 0:1434]
                head = seqs[:, :, 1533:1536]
                seqs = torch.cat([face, head], dim=2)
            else:
                start, end = FEATURE_CONFIGS[config_name]
                seqs = seqs[:, :, start:end]
            labels = data['labels'].clone()
            labels[labels == 2] = 1
            all_seqs.append(seqs)
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


def run_once(config_name, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = load_split('train_subjects.txt', config_name)
    X_val, y_val = load_split('val_subjects.txt', config_name)
    X_test, y_test = load_split('test_subjects.txt', config_name)

    input_dim = X_train.shape[2]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = GRUClassifier(input_dim=input_dim).to(DEVICE)
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
            torch.save(model.state_dict(), f"C:\\DIPSER\\ablation_{config_name}_seed{seed}.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(torch.load(f"C:\\DIPSER\\ablation_{config_name}_seed{seed}.pt"))

    val_probs, val_labels = get_probs(model, val_loader)
    thresholds = np.arange(0.20, 0.81, 0.02)
    best_thresh = 0.5
    best_val_f1 = 0
    for thresh in thresholds:
        preds = (val_probs >= thresh).astype(int)
        f1 = f1_score(val_labels, preds, pos_label=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = thresh

    test_probs, test_labels = get_probs(
        model, DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    )
    preds = (test_probs >= best_thresh).astype(int)

    acc = accuracy_score(test_labels, preds)
    f1_0 = f1_score(test_labels, preds, pos_label=0)
    rec_0 = recall_score(test_labels, preds, pos_label=0)
    prec_0 = precision_score(test_labels, preds, pos_label=0)
    f1_1 = f1_score(test_labels, preds, pos_label=1)

    print(f"  {config_name} seed{seed}: 维度={input_dim}, 阈值={best_thresh:.2f}, "
          f"准确率={acc:.4f}, 不参与F1={f1_0:.4f}, 召回={rec_0:.4f}, 精确={prec_0:.4f}, 参与F1={f1_1:.4f}")
    return [acc, f1_0, rec_0, prec_0, f1_1, best_thresh]


def main():
    print("=" * 60)
    print("消融实验（正确版，7配置×3种子）")
    print("=" * 60)

    all_results = {}
    for config_name in FEATURE_CONFIGS:
        print(f"\n{'='*40}\n配置: {config_name}\n{'='*40}")
        config_results = [run_once(config_name, seed) for seed in SEEDS]
        all_results[config_name] = np.array(config_results)

    print(f"\n\n{'='*80}\n最终汇总\n{'='*80}")
    print(f"{'配置':<20} {'准确率':>12} {'不参与F1':>14} {'不参与召回':>14} {'不参与精确':>14} {'参与F1':>12}")
    print("-" * 80)

    with open(r"C:\DIPSER\ablation_correct_results.txt", 'w') as f:
        f.write("消融实验结果（正确版）\n\n")
        for config_name, results in all_results.items():
            means = results.mean(axis=0)
            stds = results.std(axis=0)
            print(f"{config_name:<20} {means[0]:>12.4f} {means[1]:>14.4f} {means[2]:>14.4f} {means[3]:>14.4f} {means[4]:>12.4f}")
            f.write(f"{config_name}:\n")
            f.write(f"  准确率: {means[0]:.4f} ± {stds[0]:.4f}\n")
            f.write(f"  不参与F1: {means[1]:.4f} ± {stds[1]:.4f}\n")
            f.write(f"  不参与召回: {means[2]:.4f} ± {stds[2]:.4f}\n")
            f.write(f"  不参与精确: {means[3]:.4f} ± {stds[3]:.4f}\n")
            f.write(f"  参与F1: {means[4]:.4f} ± {stds[4]:.4f}\n\n")

    print(f"\n结果已保存: C:\\DIPSER\\ablation_correct_results.txt")


if __name__ == '__main__':
    main()
