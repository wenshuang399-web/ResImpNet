# ResImpNet: A Multi-Modal Deep Learning Framework for Enzyme kcat Prediction

**Sinusoidal Positional Encoding + Cross-Attention-Based Residue Importance + Mutation Suggestion**

---

## 🔍 Overview

ResImpNet is a multi-modal deep learning framework designed for predicting enzyme catalytic constants (kcat or km). It integrates:

* **Protein sequence embeddings** (ProtT5)
* **Substrate graph embeddings**
* **Protein structure-based residue-level embeddings**
* **Scheme-B architecture** (GFE + LFE + Cross-Attention)
* **Cross-attention-based residue importance extraction**
* **Importance-guided mutation suggestion**

This repository contains:

* `model.py` — Model architecture, including GFE/LFE modules, cross-attention, and sinusoidal positional encoding
* `train.py` — Model training, validation, testing, residue importance extraction, and mutation suggestion generation
* Feature extraction scripts



---

# 🧬 Input Features

## 1. Enzyme Sequence Embeddings (ProtT5)

**File:**
`prott5_mean_embeddings.h5`

* Each enzyme is represented as a global sequence embedding.
* Feature dimension:

```
[1024]
```

---

## 2. Substrate Graph Embeddings

**File:**
`substrate_graph_embeddings.h5`

* The substrate molecular graph is encoded using a graph-based neural representation.
* Feature dimension:

```
[128]
```

---

## 3. Protein Residue-Level Structural Embeddings

**File:**
`protein_residue_embeddings.h5`

* Residue-level representations derived from protein structural information.
* Each residue is represented independently.

Feature dimension:

```
[L, 128]
```

where:

* **L** = protein sequence length
* **128** = residue embedding dimension

---

# 🧠 Model Architecture

ResImpNet adopts the Scheme-B architecture, consisting of several key modules.

---

## 1. Sinusoidal Positional Encoding

Sinusoidal positional encoding is introduced into residue-level tokens to provide spatial sequence-order information and improve the stability of residue representation learning.

---

## 2. Cross-Attention Module

### Substrate → Residue Tokens

Cross-attention is used to model the interaction between substrate features and residue-level structural representations.

Functions:

* Integrates substrate information with protein structural features
* Identifies substrate-interacting residues
* Extracts **per-residue importance scores** from attention weights

---

## 3. Global Feature Extraction (GFE)

The GFE module applies bidirectional cross-attention:

* Residue representation → Sequence representation
* Sequence representation → Residue representation

Purpose:

* Capture long-range dependencies
* Learn global enzyme-substrate interaction patterns

---

## 4. Local Feature Extraction (LFE)

The LFE module applies multi-scale convolution operations:

Kernel sizes:

```
3 / 5 / 7
```

Purpose:

* Capture local structural motifs
* Extract neighborhood-level residue patterns

---

## 5. Feature Fusion and Regression

The extracted features are combined:

```
Global enzyme representation
        +
Local residue-level features
        +
Substrate representation
        ↓
Regression Head
        ↓
Predicted log10(kcat)
```

The final model outputs the predicted enzyme catalytic constant (**kcat or km**).

---

# 🚀 Training

Example training command:

```bash
python train.py \
    --data_csv kcat_dataset.csv \
    --enzyme_h5 prott5_mean_embeddings.h5 \
    --substrate_h5 substrate_graph_embeddings.h5 \
    --residue_h5 protein_residue_embeddings.h5 \
    --out_dir SchemeB_importance_out_sinpos
```



This enables an AI-guided enzyme engineering strategy combining **kinetic prediction, structural interpretation, and rational mutation design**.
