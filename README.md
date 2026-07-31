# ResImpNet: A Multi-Modal Deep Learning Framework for Enzyme kcat Prediction
---

## 🔍 Overview

Predicting enzyme kinetic parameters is fundamental for enzyme discovery and engineering. Here, we develop **ResImpNet**, a multi-modal deep learning framework for predicting enzyme catalytic parameters (**kcat and Km**) by integrating protein sequence information, substrate molecular representations, and protein structural information.
ResImpNet employs cross-attention-based feature interaction and multi-scale representation learning to capture complex enzyme–substrate relationships. In addition, the model provides interpretable residue importance scores through attention mechanisms, enabling identification of functionally important residues and facilitating rational mutation design. This framework establishes an integrated strategy combining kinetic prediction, structural interpretation, and AI-assisted enzyme engineering.


# 💻 System Requirements

## Hardware

For prediction: Any machine running a Linux-based operating system is recommended.
For training: A Linux-based operating system on a GPU-enabled machine is recommended.

## Create the ResImpNet environment
To run ResImpNet, you should create a conda environment that includes the following packages:

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
In addition, ResImpNet also relies on additional pre-trained models, including prot_t5_xl_uniref50 and GVP-GNN. These two models are used for extracting features from enzymes and structures, respectively. You need to place the weights for these two pre-trained models in the models directory.

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
