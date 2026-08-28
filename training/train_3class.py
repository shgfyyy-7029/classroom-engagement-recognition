"""
第三阶段：三分类GRU训练（优化版 V4）
- 特征标准化
- 仅使用类别权重（CrossEntropyLoss）
- Dropout 0.5
- 学习率 5e-5，权重衰减 1e-3
- 更温和的类别权重，避免过度牺牲标签1
输入：C:\DIPSER\sequences_split.pt
输出：C:\DIPSER\gru_3class_best_v4.pt
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import os

# ==================== 配置 ====================
DATA_PATH = r"C:\DIPSER\sequences_split.pt"
MODEL_SAVE_PATH = r"C:\DIPSER\gru_3class_best_v4.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-3
PATIENCE = 10
DROPOUT = 0.5
# =============================================


class GRUClassifier(nn.Module):
    def __init__(self, input_dim=1544, hidden_dim=128, num_layers=2,
                 num_classes=3, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        _, h = self.gru(x)
        out = self.fc(h[-1])
        return out


def train():
    print("=" * 60)
    print(f"第三阶段：三分类GRU训练 V4 (设备: {DEVICE})")
    print(f"Dropout={DROPOUT}, LR={LEARNING_RATE}, WD={WEIGHT_DECAY}")
    print("=" * 60)

    # 1. 加载数据
    data = torch.load(DATA_PATH)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    print(f"\n训练集: {X_train.shape}, 标签: {sorted(Counter(y_train.tolist()).items())}")
    print(f"验证集: {X_val.shape}, 标签: {sorted(Counter(y_val.tolist()).items())}")
    print(f"测试集: {X_test.shape}, 标签: {sorted(Counter(y_test.tolist()).items())}")

    # 2. 特征标准化
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    print("\n✅ 特征标准化完成")

    # 3. 类别权重（手动调整，更温和）
    # 原始计算: [1.672, 0.577, 1.497]
    # 手动调整：降低标签2的权重，适当提高标签1的权重
    class_weights = torch.tensor([1.5, 0.75, 1.2], dtype=torch.float).to(DEVICE)
    print(f"类别权重: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    # 4. DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 5. 模型 + 损失函数（CrossEntropyLoss，非Focal Loss）
    model = GRUClassifier(dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # 6. 训练
    best_val_loss = float('inf')
    patience_counter = 0

    print("\n开始训练...")
    for epoch in range(EPOCHS):
        # 训练
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
                loss = criterion(outputs, y)
                val_loss += loss.item()
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
            print(f"  ✅ 保存最佳模型 (验证损失={avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"早停触发 (Epoch {epoch+1})")
                break

    # 7. 测试集评估
    print(f"\n{'='*60}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存路径: {MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            all_preds.extend(outputs.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    print("\n测试集分类报告:")
    print(classification_report(
        all_labels, all_preds,
        target_names=['低参与度(0)', '中等参与度(1)', '高参与度(2)'],
        digits=4
    ))

    print("\n混淆矩阵:")
    cm = confusion_matrix(all_labels, all_preds)
    print("        预测0  预测1  预测2")
    for i, row in enumerate(cm):
        print(f"真实{i}    {row[0]:>5}  {row[1]:>5}  {row[2]:>5}")


if __name__ == '__main__':
    train()