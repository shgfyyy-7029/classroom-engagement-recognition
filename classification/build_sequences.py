"""
第二阶段：序列构建 (仅 group_03)
输入：D:\DIPSER\dipser_processed\ 下的 .pt 文件
输出：D:\DIPSER\sequences\ 下每个 subject 一个序列 .pt 文件
"""

import os
import torch
from collections import Counter

# ==================== 配置 ====================
INPUT_DIR = r"D:\DIPSER\dipser_processed"
OUTPUT_DIR = r"D:\DIPSER\sequences"
SEQUENCE_LENGTH = 10
WINDOW_STRIDE = 5
# =============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_sequences(features, labels):
    """
    滑动窗口构建序列
    窗口内所有帧标签一致才保留
    """
    n_frames = features.shape[0]
    if n_frames < SEQUENCE_LENGTH:
        return None, None

    seq_list = []
    label_list = []

    for start in range(0, n_frames - SEQUENCE_LENGTH + 1, WINDOW_STRIDE):
        end = start + SEQUENCE_LENGTH
        window_labels = labels[start:end]

        if torch.all(window_labels == window_labels[0]):
            seq_list.append(features[start:end])
            label_list.append(window_labels[0].item())

    if len(seq_list) == 0:
        return None, None

    sequences = torch.stack(seq_list)
    seq_labels = torch.tensor(label_list, dtype=torch.long)

    return sequences, seq_labels


def main():
    print("=" * 60)
    print("第二阶段：序列构建 (仅 group_03)")
    print(f"窗口: {SEQUENCE_LENGTH}帧, 步长: {WINDOW_STRIDE}帧")
    print("=" * 60)

    # 只读取 group_03 的文件
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pt')]
    pt_files = sorted([f for f in all_files if f.startswith('group_03_')])

    print(f"找到 {len(pt_files)} 个 .pt 文件 (group_03)")

    total_sequences = 0
    total_labels = Counter()
    skipped = 0

    for pt_file in pt_files:
        filepath = os.path.join(INPUT_DIR, pt_file)
        data = torch.load(filepath)
        features = data['features']
        labels = data['labels']

        seqs, seq_labels = build_sequences(features, labels)

        if seqs is None:
            skipped += 1
            continue

        output_path = os.path.join(OUTPUT_DIR, pt_file)
        torch.save({'sequences': seqs, 'labels': seq_labels}, output_path)

        counts = Counter(seq_labels.tolist())
        total_sequences += seqs.shape[0]
        total_labels.update(counts)

    print(f"\n{'='*60}")
    print(f"处理完成:")
    print(f"  成功: {len(pt_files) - skipped} 个subject")
    print(f"  跳过: {skipped} 个subject（有效序列不足）")
    print(f"  总序列数: {total_sequences}")
    print(f"  标签分布: 0={total_labels.get(0,0)}, 1={total_labels.get(1,0)}, 2={total_labels.get(2,0)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()