"""
第一阶段：从metadata/ JSON提取1544维特征 + 标签融合（增量处理版）
自动跳过已处理的subject，只处理缺失的
"""

import os
import json
import torch
import numpy as np
from collections import Counter

# ==================== 配置 ====================
DIPSER_ROOT = r"D:\Downloads\DIPSER"
OUTPUT_DIR = r"D:\DIPSER\dipser_processed"
MIN_FRAMES = 100
# =============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

ATTENTION_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}


def parse_timestamp(filename):
    name = filename.replace('.json', '')
    parts = name.split('_')
    if len(parts) == 4:
        return f"{parts[0]}:{parts[1]}:{parts[2]}:{parts[3]}"
    return None


def forward_fill(annotations):
    points = []
    for ann in annotations:
        if 'attention' in ann:
            dt = ann['datetime']
            raw = int(ann['attention'])
            if raw in [1, 2, 3, 4, 5]:
                points.append((dt, raw))
    if not points:
        return []
    points.sort(key=lambda x: x[0])
    return points


def get_label_for_frame(frame_dt, expert_points):
    last_label = None
    for dt, raw in expert_points:
        if dt <= frame_dt:
            last_label = raw
        else:
            break
    if last_label is not None:
        return ATTENTION_MAP.get(last_label)
    return None


def extract_features(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except:
        return None

    person = data.get('person')
    if person is None:
        return None
    face = person.get('face')
    if face is None:
        return None
    body = person.get('body')
    if body is None:
        return None

    features = []

    # 1. facemesh
    facemesh = face.get('facemesh') or []
    for i in range(478):
        if i < len(facemesh):
            pt = facemesh[i]
            features.extend([pt.get('x', 0.0), pt.get('y', 0.0), pt.get('z', 0.0)])
        else:
            features.extend([0.0, 0.0, 0.0])

    # 2. body_pose
    body_pose = body.get('body_pose') or []
    for i in range(33):
        if i < len(body_pose):
            pt = body_pose[i]
            features.extend([pt.get('x', 0.0), pt.get('y', 0.0), pt.get('z', 0.0)])
        else:
            features.extend([0.0, 0.0, 0.0])

    # 3. headpose
    headpose = face.get('headpose') or {}
    pose = headpose.get('pose') or {}
    features.extend([pose.get('pitch', 0.0), pose.get('yaw', 0.0), pose.get('roll', 0.0)])

    # 4. face_bbox
    fb = face.get('bounding_box') or []
    if isinstance(fb, list) and len(fb) >= 4:
        features.extend([float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3])])
    elif isinstance(fb, dict):
        features.extend([fb.get('x0', 0.0), fb.get('y0', 0.0), fb.get('x1', 0.0), fb.get('y1', 0.0)])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    # 5. body_bbox
    bb = body.get('bounding_box') or []
    if isinstance(bb, list) and len(bb) >= 4:
        features.extend([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])
    elif isinstance(bb, dict):
        features.extend([bb.get('x0', 0.0), bb.get('y0', 0.0), bb.get('x1', 0.0), bb.get('y1', 0.0)])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    return features


def process_subject(subject_path, subject_id):
    meta_dir = os.path.join(subject_path, 'metadata')
    label_dir = os.path.join(subject_path, 'labels')

    if not os.path.exists(meta_dir) or not os.path.exists(label_dir):
        return None

    json_files = [f for f in os.listdir(meta_dir) if f.endswith('.json')]
    if len(json_files) < MIN_FRAMES:
        print(f"  {subject_id}: 帧数不足 ({len(json_files)} < {MIN_FRAMES})，跳过")
        return None

    frame_map = {}
    for f in json_files:
        dt = parse_timestamp(f)
        if dt:
            frame_map[dt] = f

    sorted_dt = sorted(frame_map.keys())

    expert_points = []
    for i in range(1, 5):
        label_file = os.path.join(label_dir, f'labeler_0{i}.json')
        if not os.path.exists(label_file):
            print(f"  {subject_id}: 缺少 {label_file}")
            return None
        with open(label_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        points = forward_fill(annotations)
        if not points:
            print(f"  {subject_id}: labeler_0{i} 无有效标注点")
            return None
        expert_points.append(points)

    all_features = []
    all_labels = []
    tie_count = 0
    feature_dim = None

    for dt in sorted_dt:
        json_path = os.path.join(meta_dir, frame_map[dt])
        feat = extract_features(json_path)
        if feat is None:
            continue

        if feature_dim is None:
            feature_dim = len(feat)
        elif len(feat) != feature_dim:
            continue

        votes = []
        for points in expert_points:
            label = get_label_for_frame(dt, points)
            if label is not None:
                votes.append(label)

        if len(votes) < 4:
            continue

        counter = Counter(votes)
        most_common = counter.most_common()

        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            tie_count += 1
            continue

        final_label = most_common[0][0]
        all_features.append(feat)
        all_labels.append(final_label)

    if len(all_features) < MIN_FRAMES:
        print(f"  {subject_id}: 有效帧数不足 ({len(all_features)} < {MIN_FRAMES})，跳过")
        return None

    X = torch.tensor(all_features, dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.long)

    save_path = os.path.join(OUTPUT_DIR, f'{subject_id}.pt')
    torch.save({'features': X, 'labels': y}, save_path)

    label_counts = Counter(all_labels)
    print(f"  {subject_id}: {X.shape[0]}帧, "
          f"0={label_counts.get(0,0)}, 1={label_counts.get(1,0)}, 2={label_counts.get(2,0)}, "
          f"平票丢{tie_count}帧, 维度{X.shape[1]}")

    return X.shape[1]


def main():
    print("=" * 60)
    print("第一阶段：特征提取 + 标签融合（增量处理）")
    print("=" * 60)

    # 统计已有文件
    existing = set()
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith('.pt'):
                existing.add(f.replace('.pt', ''))

    print(f"已处理: {len(existing)} 个subject")

    total_subjects = 0
    skipped = 0
    feature_dim_final = None

    for group_dir in sorted(os.listdir(DIPSER_ROOT)):
        group_path = os.path.join(DIPSER_ROOT, group_dir)
        if not os.path.isdir(group_path):
            continue

        for exp_dir in sorted(os.listdir(group_path)):
            exp_path = os.path.join(group_path, exp_dir)
            if not os.path.isdir(exp_path):
                continue

            for sub_dir in sorted(os.listdir(exp_path)):
                sub_path = os.path.join(exp_path, sub_dir)
                if not os.path.isdir(sub_path):
                    continue

                subject_id = f"{group_dir}_{exp_dir}_{sub_dir}"

                if subject_id in existing:
                    skipped += 1
                    continue

                dim = process_subject(sub_path, subject_id)
                if dim:
                    total_subjects += 1
                    if feature_dim_final is None:
                        feature_dim_final = dim

    print(f"\n{'=' * 60}")
    print(f"新增处理: {total_subjects} 个subject")
    print(f"跳过已存在: {skipped} 个subject")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
