"""
第二阶段（回归版）：序列构建
输入：C:\DIPSER\dipser_processed_regression\ 下每个 subject 的 .pt 文件
输出：C:\DIPSER\sequences_regression\ 下每个 subject 一个序列文件

窗口10帧，步长5帧，标签取窗口内均值，无标签一致性约束
"""

import os
import torch
from collections import Counter

# ==================== 配置 ====================
INPUT_DIR = r"C:\DIPSER\dipser_processed_regression"
OUTPUT_DIR = r"C:\DIPSER\sequences_regression"
SEQUENCE_LENGTH = 10
WINDOW_STRIDE = 5
# =============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_sequences(features, labels):
    """
    滑动窗口构建序列
    标签取窗口内10帧的平均值
    返回: (sequences, seq_labels) 或 (None, None)
    """
    n_frames = features.shape[0]
    if n_frames < SEQUENCE_LENGTH:
        return None, None

    seq_list = []
    label_list = []

    for start in range(0, n_frames - SEQUENCE_LENGTH + 1, WINDOW_STRIDE):
        end = start + SEQUENCE_LENGTH
        seq = features[start:end]                    # (10, 1544)
        window_labels = labels[start:end]            # (10,)
        avg_label = window_labels.mean().item()      # 标量

        seq_list.append(seq)
        label_list.append(avg_label)

    if len(seq_list) == 0:
        return None, None

    sequences = torch.stack(seq_list)               # (N_seq, 10, 1544)
    seq_labels = torch.tensor(label_list, dtype=torch.float32)  # (N_seq,)

    return sequences, seq_labels


def main():
    print("=" * 60)
    print("第二阶段（回归版）：序列构建")
    print(f"窗口: {SEQUENCE_LENGTH}帧, 步长: {WINDOW_STRIDE}帧")
    print(f"标签: 窗口内均值, 无一致性约束")
    print(f"输入: {INPUT_DIR}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 60)

    pt_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.pt')])
    print(f"\n找到 {len(pt_files)} 个 subject 文件")

    total_sequences = 0
    skipped = 0
    all_labels = []

    for pt_file in pt_files:
        filepath = os.path.join(INPUT_DIR, pt_file)
        data = torch.load(filepath)
        features = data['features']
        labels = data['labels']

        seqs, seq_labels = build_sequences(features, labels)

        if seqs is None:
            skipped += 1
            print(f"  ⚠️ {pt_file}: 帧数不足，跳过")
            continue

        output_path = os.path.join(OUTPUT_DIR, pt_file)
        torch.save({'sequences': seqs, 'labels': seq_labels}, output_path)

        total_sequences += seqs.shape[0]
        all_labels.extend(seq_labels.tolist())

    print(f"\n{'='*60}")
    print(f"处理完成:")
    print(f"  成功: {len(pt_files) - skipped} 个subject")
    print(f"  跳过: {skipped} 个subject")
    print(f"  总序列数: {total_sequences}")

    if all_labels:
        all_labels_tensor = torch.tensor(all_labels)
        print(f"  标签范围: [{all_labels_tensor.min():.2f}, {all_labels_tensor.max():.2f}]")
        print(f"  标签均值: {all_labels_tensor.mean():.3f}")
        print(f"  标签标准差: {all_labels_tensor.std():.3f}")

    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()