# Classroom Engagement Recognition with DIPSER

## 项目简介

本项目基于DIPSER数据集，探索一种隐私友好、低部署成本的的课堂学生参与度识别方法。系统比较了三种任务设定（二分类、三分类、回归）和七种特征组合，并对模型能力边界进行了严格评估。

## 数据集

本仓库**不包含数据集**。使用本项目需自行向DIPSER原作者申请数据集。

- **DIPSER**: 西班牙阿利坎特大学发布的真实课堂多模态数据集
- **数据申请地址**: https://www.scidb.cn/en/detail?dataSetId=7856c716c0cc4589a23ee4a23d8a0893
- **官方GitHub仓库**: https://github.com/luis-marquez/DIPSER-A-Dataset-for-In-Person-Student-Emotion-and-Engagement-Recognition-in-the-Wild
- 本项目使用官方结构化特征（MediaPipe facemesh + body_pose + headpose + bbox），共1544维
- 标签来自四位专家标注，经多数投票融合

### 引用

如果使用本仓库代码，请同时引用DIPSER原论文：

```bibtex
@misc{marquezcarpintero2025dipserdatasetinpersonstudent,
  title={DIPSER: A Dataset for In-Person Student Engagement Recognition in the Wild},
  author={Marquez-Carpintero, Luis and Suescun-Ferrandiz, Sergio and Lorenzo Álvarez, Carolina and Fernandez-Herrero, Jorge and Viejo, Diego and Roig-Vila, Rosabel and Cazorla, Miguel},
  year={2025},
  eprint={2502.20209},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2502.20209}
}

## 方法

- **特征提取**: 从metadata JSON提取1544维结构化特征
- **模型**: 2层GRU (hidden=128)，输入为10帧时序序列
- **数据划分**: 按Subject划分（同一学生不跨集合），按Group分层（7:1.5:1.5）
  - 训练集: 38个subject, 97784个序列
  - 验证集: 8个subject, 21159个序列
  - 测试集: 11个subject, 30412个序列
- **方法学**: 阈值在验证集上选择，测试集仅评估一次

## 主要实验结果

### 二分类（参与/不参与，全特征1544维）

**注意：测试集参与类占78.8%，全猜“参与”的trivial baseline准确率为78.83%。**

模型主要价值在不参与类识别能力：

| 指标 | 数值 | 说明 |
|------|------|------|
| **不参与F1** | **0.4979 ± 0.0113** | 核心指标 |
| 不参与PR-AUC | 0.4120 | 随机基线0.2117 |
| 不参与召回率 | 53.52% ± 4.07% | |
| 不参与精确率 | 46.76% ± 1.17% | |
| 准确率 | 77.21% ± 0.71% | 低于trivial baseline 78.83%，参考价值有限 |

### 三分类（低/中/高）

- 准确率: 52.5%

### 回归（预测1-5分）

- PCC: 0.21，模型倾向预测均值，实际不可用

## 消融实验

7种特征组合（3种子，验证集选阈值）。完整结果见 `ablation/results/ablation_results.md`。
*完整结果和种子详情见 results/binary_result.txt。消融实验中的 full 配置（0.4790±0.0189）与主结果的差异源于训练配置不同：主结果在更充分训练下取得，消融实验各配置统一使用相同的轻量训练设置以保证公平对比。*

**主要发现**:

1. **纯头部姿态（3维）整体准确率最高**（75.46%），但代价是不参与召回率仅43.91%
2. **头部+身体（102维）不参与召回率最高**（82.06%），但准确率降至63.42%
3. **面部网格对参与度识别贡献有限**：GRU容量对照实验（128/256/512）无显著提升
4. **特征选择取决于应用目标**：少冤枉学生选headpose_only，少漏走神选headpose_body

## 窗口长度对比

测试了5帧、10帧、20帧三种时序窗口（完整结果见 `results/window_comparison.md`）：

| 窗口 | 准确率 | 不参与F1 |
|------|--------|----------|
| 5帧 | 73.69% | 0.4847 |
| **10帧** | **77.21%** | **0.4979** |
| 20帧 | 79.36% | 0.4252 |

在当前标签融合和序列构建条件下，10帧窗口取得了最优的不参与F1和最好的稳定性。20帧虽然准确率最高，但不参与召回率仅40.13%，倾向于预测多数类。

## PCA分析

1544维特征中前5个主成分解释96.38%方差，但PCA降维后性能显著下降（PCA-10不参与F1 0.3703，全特征0.4979）。说明方差最大的方向主要反映面部共性结构，而非参与度判别信息。无监督降维不适用于此任务。详见 `results/pca_analysis.md`。

## 特征选择分析

对比了无监督降维（PCA）和有监督特征选择（互信息）。PCA降维后性能显著下降（PCA-10不参与F1 0.3703），而互信息选择的MI-50用50维达到全特征97%的性能（不参与F1 0.4786 vs 0.4979）。说明参与度信息存在于低方差维度中，需要用有监督方法才能找到。详见 `results/feature_selection.md`。

## 基线模型对比

为验证GRU的必要性，对比了多个基线模型（完整结果见 `results/baseline_comparison.md`）：

| 模型 | 准确率 | 不参与F1 |
|------|--------|----------|
| 多数类（全猜参与） | 78.83% | 0.0000 |
| LR | 69.75% | 0.3728 |
| MLP | 66.34% | 0.4084 |
| 帧级GRU（单帧） | 73.81% | 0.4632 |
| 序列GRU（10帧） | 77.21% | 0.4979 |

**结论**：GRU的时序建模带来稳定增益（比帧级GRU高0.035 F1）。但所有模型的准确率都低于多数类基线（78.83%），说明准确率在此任务上不可靠，应以不参与F1为主要指标。

## 模型评估

### 全特征二分类模型（无处理配置）

| 指标 | 数值 | 随机基线 |
|------|------|----------|
| 准确率 | 77.21% ± 0.71% | — |
| 不参与F1 | 0.4979 ± 0.0113 | — |
| AUC | 0.7630 | 0.50 |
| 参与类PR-AUC | 0.9214 | 0.7883 |
| 不参与类PR-AUC | 0.4120 | 0.2117 |

模型对参与类的排序能力很强，但对不参与类的排序能力有限。详细分析见 `results/auc_analysis.md`。

### 局限性

1. 不参与精确率约47%，模型只适合辅助教师初筛，不适合自动判定或警告
2. 模型对随机种子敏感，最优阈值波动大
3. 结构化特征信息量有限，三分类和回归均未取得可用结果

## 环境

- Python 3.9+
- PyTorch 2.x
- 单张笔记本GPU (CUDA)

依赖见 `requirements.txt`。
