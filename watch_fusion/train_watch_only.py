"""纯手表特征训练：只用14维手表特征，看能到什么水平"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import numpy as np

FUSED_DIR = r"D:\DIPSER\sequences_fused"
SPLIT_DIR = r"C:\DIPSER"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 8
SEEDS = [42, 123, 2024]
MIN_MATCH_RATE = 0.5
VIS_DIM = 1544


class GRUClassifier(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
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
        path = os.path.join(FUSED_DIR, fname)
        if not os.path.exists(path):
            continue
        data = torch.load(path)
        match_rate = data['watch_matched'].float().mean().item()
        if match_rate < MIN_MATCH_RATE:
            continue
        # 只取手表部分
        watch_only = data['sequences'][:, :, VIS_DIM:]
        all_seqs.append(watch_only)
        all_labels.append(data['labels'])

    if len(all_seqs) == 0:
        return None, None
    return torch.cat(all_seqs, dim=0), torch.cat(all_labels, dim=0)


def standardize(X_train, X_val, X_test):
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


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


def run_once(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = load_split('train_subjects.txt')
    X_val, y_val = load_split('val_subjects.txt')
    X_test, y_test = load_split('test_subjects.txt')

    if X_train is None:
        return None

    X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = GRUClassifier(input_dim=14).to(DEVICE)
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
            torch.save(model.state_dict(), r"C:\DIPSER\watch_only_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(torch.load(r"C:\DIPSER\watch_only_best.pt"))

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

    print(f"Seed {seed}: 阈值={best_thresh:.2f}, 准确率={acc:.4f}, 不参与F1={f1_0:.4f}, "
          f"召回={rec_0:.4f}, 精确={prec_0:.4f}, AUC={auc:.4f}")
    return [acc, f1_0, rec_0, prec_0, auc]


print("纯手表特征训练 (14维)")
print("=" * 60)

results = []
for seed in SEEDS:
    r = run_once(seed)
    if r:
        results.append(r)

if results:
    results = np.array(results)
    print(f"\n{'='*60}")
    print("纯手表结果（3种子均值±标准差）")
    print(f"{'='*60}")
    print(f"  准确率:     {results[:,0].mean():.4f} ± {results[:,0].std():.4f}")
    print(f"  不参与F1:   {results[:,1].mean():.4f} ± {results[:,1].std():.4f}")
    print(f"  不参与召回: {results[:,2].mean():.4f} ± {results[:,2].std():.4f}")
    print(f"  不参与精确: {results[:,3].mean():.4f} ± {results[:,3].std():.4f}")
    print(f"  AUC:        {results[:,4].mean():.4f} ± {results[:,4].std():.4f}")

    print(f"\n对比:")
    print(f"{'配置':<20} {'准确率':>10} {'不参与F1':>12} {'AUC':>10}")
    print("-" * 55)
    print(f"{'纯视觉':<20} {'0.7721':>10} {'0.4979':>12} {'0.7630':>10}")
    print(f"{'视觉+手表':<20} {0.8892:>10.4f} {0.1619:>12.4f} {0.5425:>10.4f}")
    print(f"{'纯手表':<20} {results[:,0].mean():>10.4f} {results[:,1].mean():>12.4f} {results[:,4].mean():>10.4f}")
