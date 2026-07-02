# ResImpNet: A Multi-Modal Deep Learning Framework for Enzyme kcat Prediction  
**Sinusoidal Positional Encoding + Cross-Attention Importance + Mutation Suggestion**

---

## 🔍 Overview

ResImpNet 是一个用于预测酶催化常数（kcat）的多模态深度学习框架，整合了：

- **蛋白质序列嵌入（ProtT5）**
- **底物图嵌入（Graph-based substrate embeddings）**
- **蛋白质结构残基级嵌入（ residue embeddings）**
- **改进版 Scheme-B 架构（GFE + LFE + Cross-Attention）**
- **基于 cross-attention 的残基重要性（importance）提取**
- **基于重要性的突变建议（mutation suggestion）**

本仓库包含：

- `model.py` —— 模型结构（GFE/LFE、cross-attention、sinusoidal PE）
- `train.py` —— 训练、验证、测试、importance 提取、突变建议生成
- 特征提取脚本（可选）
- 训练好的模型（可选）
- 示例数据（推荐）

---

## 📁 Project Structure

---

## 🧬 Input Features

### 1. **Enzyme sequence embeddings（ProtT5）**
- 文件：`prott5_mean_embeddings.h5`
- 每个样本：`[1024]` 向量

### 2. **Substrate graph embeddings**
- 文件：`substrate_graph_embeddings.h5`
- 每个样本：`[128]` 向量

### 3. **Protein residue embeddings**
- 文件：`protein_residue_embeddings.h5`
- 每个样本：`[L, 128]`（L = 残基数）

---

## 🧠 Model Architecture

模型采用改进版 Scheme-B 框架，包括：

### ✔ **Sinusoidal positional encoding**  
替代原有 learned position embedding，使残基级 token 更稳定。

### ✔ **Cross-attention（substrate → residue tokens）**  
用于：
- 融合底物与蛋白质结构信息  
- 提取 per-residue importance（注意力权重）

### ✔ **GFE（Global Feature Extraction）**  
双向 cross-attention：  
- residue → sequence  
- sequence → residue  

### ✔ **LFE（Local Feature Extraction）**  
多尺度卷积（3/5/7 kernel）提取局部结构模式。

### ✔ **Fusion + Regression**  
融合 enzyme embedding + local feature → kcat 预测。

---

## 🚀 Training

运行：

```bash
python train.py \
    --data_csv DEKP-kcat_dataset_fixed.csv \
    --enzyme_h5 prott5_mean_embeddings.h5 \
    --substrate_h5 substrate_graph_embeddings.h5 \
    --residue_h5 protein_residue_embeddings.h5 \
    --out_dir SchemeB_importance_out_sinpos


