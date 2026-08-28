"""
第三阶段（回归版）：GRU回归训练
输入：C:\DIPSER\sequences_regression\ 下的序列文件 + C:\DIPSER\ 下的划分文件
输出：C:\DIPSER\gru_regression_best.pt
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ==================== 配置 ====================
SEQUENCES_DIR = r"C:\DIPSER\sequences_regression"
SPLIT_DIR = r"C:\DIPSER"
MODEL_SAVE_PATH = r"C:\DIPSER\gru_regression_best.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 10
# =============================================


class GRURegressor(nn.Module):
    def __init__(self, input_dim=1544, hidden_dim=128, num_layers=2, dropout=0.3):
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
            nn.Linear(64, 1)
        )

    def forward(self, x):
        _, h = self.gru(x)
        out = self.fc(h[-1])
        return out.squeeze(-1)  # (batch,)


def load_split(split_file):
    """加载划分文件，返回合并后的特征和标签"""
    with open(os.path.join(SPLIT_DIR, split_file), 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    all_seqs = []
    all_labels = []

    for fname in files:
        path = os.path.join(SEQUENCES_DIR, fname)
        if os.path.exists(path):
            data = torch.load(path)
            all_seqs.append(data['sequences'])
            all_labels.append(data['labels'])

    X = torch.cat(all_seqs, dim=0)
    y = torch.cat(all_labels, dim=0)
    return X, y


def compute_metrics(preds, targets):
    """计算MAE, RMSE, one-off accuracy"""
    preds = np.array(preds)
    targets = np.array(targets)

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    preds_rounded = np.round(preds)
    one_off = np.mean(np.abs(preds_rounded - targets) <= 1.0)

    return mae, rmse, one_off


def train():
    print("=" * 60)
    print(f"第三阶段（回归版）：GRU回归训练 (设备: {DEVICE})")
    print("=" * 60)

    # 1. 加载数据
    print("\n加载数据...")
    X_train, y_train = load_split('train_subjects.txt')
    X_val, y_val = load_split('val_subjects.txt')
    X_test, y_test = load_split('test_subjects.txt')

    print(f"训练集: {X_train.shape}, 标签均值={y_train.mean():.3f}")
    print(f"验证集: {X_val.shape}, 标签均值={y_val.mean():.3f}")
    print(f"测试集: {X_test.shape}, 标签均值={y_test.mean():.3f}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # 2. 模型
    model = GRURegressor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # 3. 训练
    best_val_loss = float('inf')
    patience_counter = 0

    print("\n开始训练...")
    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().tolist())
            train_targets.extend(y.cpu().tolist())

        avg_train_loss = train_loss / len(train_loader)
        train_mae, _, _ = compute_metrics(train_preds, train_targets)

        # 验证
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                val_loss += criterion(outputs, y).item()
                val_preds.extend(outputs.cpu().tolist())
                val_targets.extend(y.cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_mae, _, _ = compute_metrics(val_preds, val_targets)

        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"训练损失={avg_train_loss:.4f} 训练MAE={train_mae:.4f} | "
              f"验证损失={avg_val_loss:.4f} 验证MAE={val_mae:.4f}")

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

    # 4. 测试集评估
    print(f"\n{'='*60}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            test_preds.extend(outputs.cpu().tolist())
            test_targets.extend(y.cpu().tolist())

    mae, rmse, one_off = compute_metrics(test_preds, test_targets)

    print(f"\n测试集评估结果:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  One-off accuracy (±1): {one_off:.4f} ({100*one_off:.1f}%)")


if __name__ == '__main__':
    train()