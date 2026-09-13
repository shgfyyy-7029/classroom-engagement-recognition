"""
二分类GRU训练（C配置：2倍重采样）
输入：D:\DIPSER\sequences\ + C:\DIPSER\ 下的划分文件
输出：C:\DIPSER\gru_binary_final.pt
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

# ==================== 配置 ====================
SEQUENCES_DIR = r"D:\DIPSER\sequences"
SPLIT_DIR = r"C:\DIPSER"
MODEL_SAVE_PATH = r"C:\DIPSER\gru_binary_final.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 10
# =============================================


class GRUClassifier(nn.Module):
    def __init__(self, input_dim=1544, hidden_dim=128, num_layers=2,
                 num_classes=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        _, h = self.gru(x)
        out = self.fc(h[-1])
        return out


def load_split(split_file):
    """加载数据，标签2合并到标签1"""
    with open(os.path.join(SPLIT_DIR, split_file), 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    all_seqs = []
    all_labels = []

    for fname in files:
        path = os.path.join(SEQUENCES_DIR, fname)
        if os.path.exists(path):
            data = torch.load(path)
            labels = data['labels'].clone()
            labels[labels == 2] = 1
            all_seqs.append(data['sequences'])
            all_labels.append(labels)

    X = torch.cat(all_seqs, dim=0)
    y = torch.cat(all_labels, dim=0)
    return X, y


def find_best_threshold(probs, labels):
    """在验证集上找不参与F1最大的阈值"""
    from sklearn.metrics import f1_score
    best_thresh, best_f1 = 0.5, 0
    for thresh in np.arange(0.20, 0.81, 0.02):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, pos_label=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def train():
    print("=" * 60)
    print(f"二分类GRU训练 - C配置（2倍重采样） (设备: {DEVICE})")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    X_train, y_train = load_split('train_subjects.txt')
    X_val, y_val = load_split('val_subjects.txt')
    X_test, y_test = load_split('test_subjects.txt')

    print(f"训练集: {X_train.shape}, 标签分布: {dict(Counter(y_train.tolist()))}")
    print(f"验证集: {X_val.shape}, 标签分布: {dict(Counter(y_val.tolist()))}")
    print(f"测试集: {X_test.shape}, 标签分布: {dict(Counter(y_test.tolist()))}")

    # C配置：2倍重采样
    sample_weights = [2.0 if label == 0 else 1.0 for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=BATCH_SIZE, shuffle=False)

    # 模型
    model = GRUClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # 训练
    best_val_loss = float('inf')
    patience_counter = 0

    print("\n开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += y.size(0)
            train_correct += (outputs.argmax(1) == y).sum().item()

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                val_loss += criterion(outputs, y).item()
                val_total += y.size(0)
                val_correct += (outputs.argmax(1) == y).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"训练损失={avg_train_loss:.4f} 训练准确率={train_acc:.2f}% | "
              f"验证损失={avg_val_loss:.4f} 验证准确率={val_acc:.2f}%")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"  ✅ 保存最佳模型")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"早停触发 (Epoch {epoch+1})")
                break

    # 测试集评估
    print(f"\n{'='*60}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    # 验证集选阈值
    val_probs, val_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            p = torch.softmax(outputs, dim=1)[:, 1]
            val_probs.extend(p.cpu().tolist())
            val_labels.extend(y.cpu().tolist())

    best_thresh = find_best_threshold(np.array(val_probs), np.array(val_labels))
    print(f"验证集最优阈值: {best_thresh:.2f}")

    # 测试集用该阈值评估
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False)
    test_probs, test_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            p = torch.softmax(outputs, dim=1)[:, 1]
            test_probs.extend(p.cpu().tolist())
            test_labels.extend(y.cpu().tolist())

    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    preds = (test_probs >= best_thresh).astype(int)

    print(f"\n测试集分类报告（阈值={best_thresh:.2f}）:")
    print(classification_report(
        test_labels, preds,
        target_names=['不参与(0)', '参与(1)'],
        digits=4
    ))

    auc = roc_auc_score(test_labels, test_probs)
    print(f"测试集 AUC: {auc:.4f}")

    print("\n混淆矩阵:")
    cm = confusion_matrix(test_labels, preds)
    print("        预测0  预测1")
    for i, row in enumerate(cm):
        print(f"真实{i}    {row[0]:>5}  {row[1]:>5}")


if __name__ == '__main__':
    train()
