"""
数据集划分脚本
输入：D:\DIPSER\sequences\ 下每个 subject 的序列 .pt 文件
输出：C:\DIPSER\ 下的划分文件和合并后的序列数据
策略：按Subject划分，按Group分层，7:1.5:1.5
"""

import os
import torch
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# ==================== 配置 ====================
INPUT_DIR = r"D:\DIPSER\sequences"
OUTPUT_DIR = r"C:\DIPSER"
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# =============================================

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_subject_id(filename):
    """
    从文件名提取唯一标识
    group_01_experiment_01_subject_01.pt → unique_id='group_01_subject_01', group='group_01'
    """
    name = filename.replace('.pt', '')
    parts = name.split('_')
    group = f"{parts[0]}_{parts[1]}"          # group_01
    subj = f"{parts[4]}_{parts[5]}"           # subject_01
    unique_id = f"{group}_{subj}"             # group_01_subject_01
    return unique_id, group


def main():
    print("=" * 60)
    print("数据集划分脚本（按Group分层版）")
    print(f"策略: 按Subject划分, 按Group分层, 7:1.5:1.5")
    print(f"输入: {INPUT_DIR}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 60)

    # 1. 构建subject信息
    pt_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.pt')])
    print(f"\n找到 {len(pt_files)} 个序列文件")

    subject_info = {}
    for pt_file in pt_files:
        unique_id, group = extract_subject_id(pt_file)
        if unique_id not in subject_info:
            subject_info[unique_id] = group

    unique_ids = list(subject_info.keys())
    groups = [subject_info[uid] for uid in unique_ids]

    print(f"不重复Subject数: {len(unique_ids)}")
    for g in ['group_01', 'group_02', 'group_03']:
        count = groups.count(g)
        print(f"  {g}: {count} 个")

    # 2. 按Group分层划分
    train_ids, val_ids, test_ids = [], [], []

    for g in ['group_01', 'group_02', 'group_03']:
        g_ids = [uid for uid in unique_ids if subject_info[uid] == g]

        if len(g_ids) == 1:
            # 只有一个subject，放入训练集
            train_ids.extend(g_ids)
            continue

        # 第一次划分: train vs temp
        g_train, g_temp = train_test_split(
            g_ids,
            test_size=(1 - TRAIN_RATIO),
            random_state=RANDOM_SEED
        )

        # 第二次划分: val vs test
        val_ratio_adjusted = VAL_RATIO / (1 - TRAIN_RATIO)
        g_val, g_test = train_test_split(
            g_temp,
            test_size=(1 - val_ratio_adjusted),
            random_state=RANDOM_SEED
        )

        train_ids.extend(g_train)
        val_ids.extend(g_val)
        test_ids.extend(g_test)

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    print(f"\n训练集: {len(train_set)} 个subject")
    print(f"验证集: {len(val_set)} 个subject")
    print(f"测试集: {len(test_set)} 个subject")

    # 3. 构建序列数据
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    train_subject_files, val_subject_files, test_subject_files = [], [], []

    for pt_file in pt_files:
        unique_id, group = extract_subject_id(pt_file)
        data = torch.load(os.path.join(INPUT_DIR, pt_file))
        seqs = data['sequences']
        labels = data['labels']

        if unique_id in train_set:
            X_train.append(seqs)
            y_train.append(labels)
            train_subject_files.append(pt_file)
        elif unique_id in val_set:
            X_val.append(seqs)
            y_val.append(labels)
            val_subject_files.append(pt_file)
        elif unique_id in test_set:
            X_test.append(seqs)
            y_test.append(labels)
            test_subject_files.append(pt_file)

    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)
    X_val = torch.cat(X_val, dim=0)
    y_val = torch.cat(y_val, dim=0)
    X_test = torch.cat(X_test, dim=0)
    y_test = torch.cat(y_test, dim=0)

    # 4. 保存
    torch.save({
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }, os.path.join(OUTPUT_DIR, 'sequences_split.pt'))

    with open(os.path.join(OUTPUT_DIR, 'train_subjects.txt'), 'w') as f:
        f.write('\n'.join(sorted(train_subject_files)))
    with open(os.path.join(OUTPUT_DIR, 'val_subjects.txt'), 'w') as f:
        f.write('\n'.join(sorted(val_subject_files)))
    with open(os.path.join(OUTPUT_DIR, 'test_subjects.txt'), 'w') as f:
        f.write('\n'.join(sorted(test_subject_files)))

    # 5. 统计
    def print_split_stats(X, y, name):
        counts = Counter(y.tolist())
        total = len(y)
        print(f"\n{name}: {X.shape[0]} 个序列")
        for k in sorted(counts.keys()):
            print(f"  标签{k}: {counts[k]} ({100*counts[k]/total:.1f}%)")

    print("\n" + "=" * 60)
    print("划分结果统计")
    print("=" * 60)
    print_split_stats(X_train, y_train, "训练集")
    print_split_stats(X_val, y_val, "验证集")
    print_split_stats(X_test, y_test, "测试集")

    print("\n" + "=" * 60)
    print("各集合Group分布")
    print("=" * 60)
    for name, ids in [("训练集", train_set), ("验证集", val_set), ("测试集", test_set)]:
        group_counts = Counter()
        for uid in ids:
            group_counts[subject_info[uid]] += 1
        print(f"\n{name}:")
        for g in ['group_01', 'group_02', 'group_03']:
            print(f"  {g}: {group_counts.get(g, 0)} 个subject")

    print(f"\n✅ 数据已保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()