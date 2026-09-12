"""
第三步：融合视觉序列和手表特征（0填充版）
输入：
  - D:\DIPSER\processed_with_time\
  - D:\DIPSER\watch_features\
输出：
  - D:\DIPSER\sequences_fused\
找不到手表数据时用全0填充，不用前一个值
"""
import os
import torch
import numpy as np
from collections import Counter

# ==================== 配置 ====================
VIS_DIR = r"D:\DIPSER\processed_with_time"
WATCH_DIR = r"D:\DIPSER\watch_features"
OUTPUT_DIR = r"D:\DIPSER\sequences_fused"
WINDOW = 10
STRIDE = 5
# =============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_second_key(timestamp):
    """从 '10:40:43:047895' 提取 '10:40:43'"""
    parts = timestamp.split(':')
    if len(parts) == 4:
        return f"{parts[0]}:{parts[1]}:{parts[2]}"
    return None


def build_sequences_with_time(features, labels, timestamps, window, stride):
    """滑动窗口构建序列，窗口内标签一致才保留，返回序列+起始秒键"""
    n = features.shape[0]
    if n < window:
        return None, None, None

    seqs, labs, start_keys = [], [], []
    for start in range(0, n - window + 1, stride):
        end = start + window
        w_labels = labels[start:end]
        if torch.all(w_labels == w_labels[0]):
            seqs.append(features[start:end])
            labs.append(w_labels[0].item())
            start_keys.append(get_second_key(timestamps[start]))

    if len(seqs) == 0:
        return None, None, None
    return torch.stack(seqs), torch.tensor(labs, dtype=torch.long), start_keys


def match_watch_features(start_key, watch_features_dict):
    """
    根据 '10:40:43' 匹配手表特征
    找不到直接返回全0
    """
    if start_key in watch_features_dict:
        return watch_features_dict[start_key], 1
    return np.zeros(14, dtype=np.float32), 0


def process_subject(vis_path, watch_path, subject_id):
    if not os.path.exists(vis_path):
        return None

    vis_data = torch.load(vis_path)
    features = vis_data['features']
    labels = vis_data['labels']
    timestamps = vis_data['timestamps']

    # 构建视觉序列
    seqs, labs, start_keys = build_sequences_with_time(features, labels, timestamps, WINDOW, STRIDE)
    if seqs is None:
        print(f"  {subject_id}: 无有效序列，跳过")
        return None

    # 加载手表特征
    watch_features_dict = {}
    if os.path.exists(watch_path):
        watch_data = torch.load(watch_path)
        wf = watch_data['watch_features'].numpy()
        wts = watch_data['timestamps']
        for i, ts in enumerate(wts):
            key = get_second_key(ts)
            if key and key not in watch_features_dict:
                watch_features_dict[key] = wf[i]

    # 匹配手表特征（找不到用0）
    watch_vecs = []
    matched_flags = []

    for key in start_keys:
        if key is None:
            watch_vecs.append(np.zeros(14, dtype=np.float32))
            matched_flags.append(0)
            continue
        vec, flag = match_watch_features(key, watch_features_dict)
        watch_vecs.append(vec)
        matched_flags.append(flag)

    watch_vecs = torch.tensor(np.array(watch_vecs), dtype=torch.float32)

    # 广播到10帧并与视觉拼接
    N_seq = seqs.shape[0]
    watch_broadcast = watch_vecs.unsqueeze(1).repeat(1, WINDOW, 1)
    fused = torch.cat([seqs, watch_broadcast], dim=2)

    matched_flags = torch.tensor(matched_flags, dtype=torch.long)

    save_path = os.path.join(OUTPUT_DIR, f'{subject_id}.pt')
    torch.save({
        'sequences': fused,
        'labels': labs,
        'watch_matched': matched_flags
    }, save_path)

    match_rate = matched_flags.float().mean().item()
    counts = Counter(labs.tolist())
    print(f"  {subject_id}: {N_seq}序列, 不参与{counts.get(0,0)}, 参与{counts.get(1,0)}, "
          f"手表匹配率{match_rate:.2%}")

    return fused.shape[2]


def main():
    print("=" * 60)
    print("第三步：融合视觉序列和手表特征（0填充版）")
    print(f"窗口{WINDOW}帧, 步长{STRIDE}帧")
    print("=" * 60)

    vis_files = sorted([f for f in os.listdir(VIS_DIR) if f.endswith('.pt')])
    print(f"\n找到 {len(vis_files)} 个视觉文件")

    total = 0
    dim_final = None

    for vis_file in vis_files:
        subject_id = vis_file.replace('.pt', '')
        vis_path = os.path.join(VIS_DIR, vis_file)
        watch_path = os.path.join(WATCH_DIR, vis_file)

        dim = process_subject(vis_path, watch_path, subject_id)
        if dim:
            total += 1
            if dim_final is None:
                dim_final = dim

    print(f"\n{'='*60}")
    print(f"处理完成: {total} 个subject")
    print(f"融合特征维度: {dim_final}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
