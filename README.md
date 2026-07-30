# ResImpNet: A Multi-Modal Deep Learning Framework for Enzyme kcat Prediction

**Sinusoidal Positional Encoding + Cross-Attention-Based Residue Importance **

---

## 🔍 Overview

ResImpNet is a multi-modal deep learning framework for predicting enzyme catalytic constants (**kcat or km**). The framework integrates protein sequence information, substrate molecular representations, and residue-level structural embeddings to improve predictive accuracy while providing interpretable residue importance scores for rational enzyme engineering.

The framework consists of:

- **Protein sequence embeddings (ProtT5)**
- **Substrate graph embeddings**
- **Protein residue-level structural embeddings**
- **Sinusoidal positional encoding**
- **Cross-attention-based residue importance extraction**
- **Global Feature Extraction (GFE)**
- **Local Feature Extraction (LFE)**
- **Importance-guided mutation suggestion**

This repository contains:

- `model.py` – ResImpNet model architecture
- `train.py` – Model training, validation, testing, residue importance extraction
- Feature extraction scripts
- Example input files

---

# 🧬 Input Features

## 1. Protein Sequence Embeddings (ProtT5)

**File**

```
prott5_mean_embeddings.h5
```

Each enzyme is represented as a global protein embedding.

Feature dimension

```
[1024]
```

---

## 2. Substrate Graph Embeddings

**File**

```
substrate_graph_embeddings.h5
```

Each substrate is encoded using a graph neural representation.

Feature dimension

```
[128]
```

---

## 3. Protein Residue-Level Structural Embeddings

**File**

```
protein_residue_embeddings.h5
```

Each residue is represented independently using structure-derived embeddings.

Feature dimension

```
[L,128]
```

where

- **L** = protein sequence length
- **128** = residue embedding dimension

---

# 🧠 Model Architecture

ResImpNet adopts the proposed **Scheme-B** architecture.

---

## 1. Sinusoidal Positional Encoding

Residue tokens are enhanced using sinusoidal positional encoding to preserve sequence-order information and improve residue representation learning.

---

## 2. Cross-Attention Module

### Substrate → Residue Tokens

Cross-attention models interactions between substrate representations and residue-level structural embeddings.

Functions include:

- Integrating substrate information into protein structural representations
- Identifying substrate-interacting residues
- Extracting residue importance scores from attention weights

---

## 3. Global Feature Extraction (GFE)

The GFE module applies bidirectional cross-attention between global sequence representations and residue representations.

Purpose:

- Capture long-range dependencies
- Learn global enzyme–substrate interaction patterns

---

## 4. Local Feature Extraction (LFE)

The LFE module employs multi-scale convolution layers.

Kernel sizes

```
3 / 5 / 7
```

Purpose:

- Capture local structural motifs
- Learn neighborhood-level residue patterns

---

## 5. Feature Fusion and Regression

The extracted representations are fused as follows:

```
Global enzyme representation
        +
Local residue features
        +
Substrate representation
        ↓
Regression Head
        ↓
Predicted log10(kcat)
```

The final output is the predicted enzyme catalytic constant.

---

# 💻 System Requirements

## Hardware

For prediction: Any machine running a Linux-based operating system is recommended.
For training: A Linux-based operating system on a GPU-enabled machine is recommended.

## Software

- Python 3.9
- PyTorch ≥ 1.12
- NumPy ≥ 1.21
- Pandas ≥ 1.3
- SciPy ≥ 1.7
- Scikit-learn ≥ 1.0
- h5py ≥ 3.6
- tqdm ≥ 4.64
- seaborn ≥ 0.11

The framework is compatible with Linux.

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/wenshuang399-web/ResImpNet.git

cd ResImpNet
```

### Install using Conda

```bash
conda env create -f environment.yml

conda activate ResImpNet
```

### Or install using pip

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

Example training command

```bash
python train.py \
    --data_csv kcat_dataset.csv \
    --enzyme_h5 prott5_mean_embeddings.h5 \
    --substrate_h5 substrate_graph_embeddings.h5 \
    --residue_h5 protein_residue_embeddings.h5 \
    --out_dir SchemeB_importance_out_sinpos
```

---

# ⏱ Installation Time

Typical installation on a standard desktop computer requires approximately **5–10 minutes**, depending on internet speed and package download time.

---

# ⏳ Expected Run Time for Demo

Running the demo on a single NVIDIA RTX 4090 GPU typically requires approximately **2–5 minutes**, depending on the size of the input dataset.

CPU execution is supported but may require significantly longer runtime.

---

# 📄 Expected Output

After training or inference, the output directory (`SchemeB_importance_out_sinpos/`) contains:

- Trained model checkpoints (`*.pth`)
- Training and validation logs
- Predicted log10(kcat) values
- Residue importance scores extracted from the cross-attention module

These outputs provide both predictive results and interpretable residue-level information for downstream enzyme engineering.

---

# 📜 License

This project is released under the MIT License.
