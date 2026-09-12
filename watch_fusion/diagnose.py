"""诊断：融合数据的关键指标"""
import os
import torch
import numpy as np
from collections import Counter

FUSED_DIR = r"D:\DIPSER\sequences_fused"
SPLIT_DIR = r"C:\DIPSER"
VIS_DIM = 1544
WATCH_DIM = 14
MIN_MATCH_RATE = 0.5


def diagnose_split(split_file, name):
    with open(os.path.join(SPLIT_DIR, split_file), 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"{name}: 共 {len(files)} 个subject文件")
    print(f"{'='*60}")

    total_seqs = 0
    total_labels = Counter()
    skipped_files = []
    match_rates = []

    all_vis_vals = []
    all_watch_vals = []

    for fname in files:
        path = os.path.join(FUSED_DIR, fname)
        if not os.path.exists(path):
            skipped_files.append((fname, "文件不存在"))
            continue
        data = torch.load(path)
        seqs = data['sequences']
        labels = data['labels']
        match_rate = data['watch_matched'].float().mean().item()

        if match_rate < MIN_MATCH_RATE:
            skipped_files.append((fname, f"匹配率{match_rate:.2%}"))
            continue

        match_rates.append(match_rate)
        total_seqs += len(labels)
        total_labels.update(labels.tolist())

        # 抽样统计数值范围（每个文件取前100个序列）
        sample = seqs[:min(100, len(seqs))]
        all_vis_vals.append(sample[:, :, :VIS_DIM].numpy())
        all_watch_vals.append(sample[:, :, VIS_DIM:].numpy())

    print(f"\n有效subject数: {len(match_rates)}")
    print(f"被剔除subject数: {len(skipped_files)}")
    if skipped_files:
        print(f"剔除原因:")
        for f, reason in skipped_files[:5]:
            print(f"  {f}: {reason}")
        if len(skipped_files) > 5:
            print(f"  ... 共{len(skipped_files)}个")

    print(f"\n总序列数: {total_seqs}")
    print(f"标签分布: 不参与{total_labels.get(0,0)}, 参与{total_labels.get(1,0)}")

    if match_rates:
        print(f"\n手表匹配率: 均值{np.mean(match_rates):.2%}, 最小{np.min(match_rates):.2%}, 最大{np.max(match_rates):.2%}")

    if all_vis_vals:
        vis = np.concatenate(all_vis_vals, axis=0)
        watch = np.concatenate(all_watch_vals, axis=0)
        print(f"\n视觉特征数值范围: [{vis.min():.4f}, {vis.max():.4f}], 均值{vis.mean():.4f}, 标准差{vis.std():.4f}")
        print(f"手表特征数值范围: [{watch.min():.4f}, {watch.max():.4f}], 均值{watch.mean():.4f}, 标准差{watch.std():.4f}")

        # 手表各维度统计
        print(f"\n手表14维各维度均值/标准差:")
        for i in range(WATCH_DIM):
            dim_vals = watch[:, :, i]
            print(f"  维度{i}: 均值{dim_vals.mean():.4f}, 标准差{dim_vals.std():.4f}, 范围[{dim_vals.min():.4f}, {dim_vals.max():.4f}]")


diagnose_split('train_subjects.txt', '训练集')
diagnose_split('val_subjects.txt', '验证集')
diagnose_split('test_subjects.txt', '测试集')
