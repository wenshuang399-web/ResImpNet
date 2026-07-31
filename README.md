# ResImpNet: A Multi-Modal Deep Learning Framework for Enzyme kcat Prediction

<img width="700" height="281" alt="image" src="https://github.com/user-attachments/assets/082c761b-fd1b-42e6-86f3-058e553877e2" />


---

## 🔍 Overview

ResImpNet is a multi-modal deep learning framework for predicting enzyme catalytic constants (**kcat or km**). The framework integrates protein sequence information, substrate molecular representations, and residue-level structural embeddings to improve predictive accuracy while providing interpretable residue importance scores for rational enzyme engineering.


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
