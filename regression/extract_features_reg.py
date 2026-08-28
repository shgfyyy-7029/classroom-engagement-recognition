"""
第一阶段（回归版·终版）：从metadata/ JSON提取1544维特征 + 回归标签（四位专家平均值）
自动跳过已处理的subject，仅处理新增/未完成的
"""

import os
import json
import torch
import numpy as np
from collections import Counter

# ==================== 配置 ====================
DIPSER_ROOT = r"C:\Downloads\DIPSER"
OUTPUT_DIR = r"C:\DIPSER\dipser_processed_regression"
MIN_FRAMES = 100
# =============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def get_regression_label(frame_dt, expert_points_list):
    values = []
    for points in expert_points_list:
        last_label = None
        for dt, raw in points:
            if dt <= frame_dt:
                last_label = raw
            else:
                break
        if last_label is not None:
            values.append(last_label)
    if len(values) == 0:
        return None
    return sum(values) / len(values)


def safe_float(val):
    """安全转换为float，非数字返回0.0"""
    if isinstance(val, (int, float)):
        return float(val)
    return 0.0


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

    # 1. facemesh: 478点 × 3维 = 1434维
    facemesh = face.get('facemesh') or []
    for i in range(478):
        if i < len(facemesh):
            pt = facemesh[i]
            features.extend([safe_float(pt.get('x', 0.0)), safe_float(pt.get('y', 0.0)), safe_float(pt.get('z', 0.0))])
        else:
            features.extend([0.0, 0.0, 0.0])

    # 2. body_pose: 33点 × 3维 = 99维
    body_pose = body.get('body_pose') or []
    for i in range(33):
        if i < len(body_pose):
            pt = body_pose[i]
            features.extend([safe_float(pt.get('x', 0.0)), safe_float(pt.get('y', 0.0)), safe_float(pt.get('z', 0.0))])
        else:
            features.extend([0.0, 0.0, 0.0])

    # 3. headpose: 3维
    headpose = face.get('headpose') or {}
    pose = headpose.get('pose') or {}
    for coord in ['pitch', 'yaw', 'roll']:
        features.append(safe_float(pose.get(coord, 0.0)))

    # 4. face_bbox: 4维
    fb = face.get('bounding_box') or []
    if isinstance(fb, list) and len(fb) >= 4:
        features.extend([safe_float(fb[0]), safe_float(fb[1]), safe_float(fb[2]), safe_float(fb[3])])
    elif isinstance(fb, dict):
        for key in ['x0', 'y0', 'x1', 'y1']:
            features.append(safe_float(fb.get(key, 0.0)))
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    # 5. body_bbox: 4维
    bb = body.get('bounding_box') or []
    if isinstance(bb, list) and len(bb) >= 4:
        features.extend([safe_float(bb[0]), safe_float(bb[1]), safe_float(bb[2]), safe_float(bb[3])])
    elif isinstance(bb, dict):
        for key in ['x0', 'y0', 'x1', 'y1']:
            features.append(safe_float(bb.get(key, 0.0)))
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
        print(f"  {subject_id}: JSON数量不足 ({len(json_files)} < {MIN_FRAMES})，跳过")
        return None

    frame_map = {}
    for f in json_files:
        dt = parse_timestamp(f)
        if dt:
            frame_map[dt] = f

    sorted_dt = sorted(frame_map.keys())

    expert_points_list = []
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
        expert_points_list.append(points)

    all_features = []
    all_labels = []

    for dt in sorted_dt:
        json_path = os.path.join(meta_dir, frame_map[dt])
        feat = extract_features(json_path)
        if feat is None:
            continue

        label = get_regression_label(dt, expert_points_list)
        if label is None:
            continue

        all_features.append(feat)
        all_labels.append(label)

    if len(all_features) < MIN_FRAMES:
        print(f"  {subject_id}: 有效帧数不足 ({len(all_features)} < {MIN_FRAMES})，跳过")
        return None

    X = torch.tensor(all_features, dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.float32)

    save_path = os.path.join(OUTPUT_DIR, f'{subject_id}.pt')
    torch.save({'features': X, 'labels': y}, save_path)

    print(f"  ✅ {subject_id}: {X.shape[0]}帧, 标签范围=[{y.min():.1f}, {y.max():.1f}], 均值={y.mean():.2f}")

    return X.shape[1]


def main():
    print("=" * 60)
    print("第一阶段（回归版·终版）：特征提取 + 回归标签（专家平均值）")
    print(f"特征: 1544维")
    print(f"标签: 四位专家平均值 (1.0-5.0)")
    print(f"输入: {DIPSER_ROOT}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 60)

    existing = set()
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith('.pt'):
                existing.add(f.replace('.pt', ''))
    print(f"已存在: {len(existing)} 个subject")

    total_new = 0
    total_skip = 0

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
                    total_skip += 1
                    continue

                dim = process_subject(sub_path, subject_id)
                if dim:
                    total_new += 1

    print(f"\n{'=' * 60}")
    print(f"处理完成: 新增 {total_new} 个, 跳过已存在 {total_skip} 个")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()