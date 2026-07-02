#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import re
import gc
import numpy as np
import pandas as pd
import os
import h5py
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

# ===================================
# 加载 ProtT5 模型
# ===================================
print("🔧 Loading ProtT5 model...")
tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"✅ Model loaded to {device}")

# ===================================
# 序列预处理
# ===================================
def preprocess_sequence(seq: str, max_len: int = 1000) -> str:
    if len(seq) > max_len:
        seq = seq[:500] + seq[-500:]
    seq = re.sub(r"[UZOB]", "X", seq)
    return ' '.join(seq)

# ===================================
# ProtT5 编码函数
# ===================================
def Seq_to_vec(seq_ids, sequences):
    features = []

    for i, seq_raw in enumerate(tqdm(sequences, desc="Encoding sequences")):
        if not isinstance(seq_raw, str) or len(seq_raw.strip()) == 0:
            print(f"Skipping empty sequence at index {i}")
            continue

        seq = preprocess_sequence(seq_raw)
        ids_tok = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding=True)

        input_ids = torch.tensor(ids_tok['input_ids']).to(device)
        attention_mask = torch.tensor(ids_tok['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        embedding = embedding.cpu().numpy()
        attn_len = (attention_mask[0] == 1).sum().item()
        seq_embed = embedding[0][:attn_len]   # (L, D)

        mean_vector = np.mean(seq_embed, axis=0)  # (D,)

        features.append({
            "id": seq_ids[i],
            "mean": mean_vector,
            "residue": seq_embed
        })

        torch.cuda.empty_cache()
        gc.collect()

    return features

# ===================================
# 读取 FASTA
# ===================================
def read_fasta(fasta_path):
    sequences = []
    seq_ids = []
    with open(fasta_path, 'r', encoding='utf-8') as f:
        seq_id = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id and seq_lines:
                    sequences.append("".join(seq_lines))
                    seq_ids.append(seq_id)
                seq_id = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if seq_id and seq_lines:
            sequences.append("".join(seq_lines))
            seq_ids.append(seq_id)
    return seq_ids, sequences

# ===================================
# 保存到两个 HDF5 文件
# ===================================
def save_embeddings(features, out_dir):
    mean_path = os.path.join(out_dir, "prott5_mean_embeddings2.h5")
    residue_path = os.path.join(out_dir, "prott5_residue_embeddings2.h5")

    with h5py.File(mean_path, "w") as h5_mean, h5py.File(residue_path, "w") as h5_res:
        for f in features:
            sid = f["id"]
            h5_mean.create_dataset(sid, data=f["mean"], compression="gzip")
            h5_res.create_dataset(sid, data=f["residue"], compression="gzip")

    print(f"✅ Saved mean embeddings → {mean_path}")
    print(f"✅ Saved residue embeddings → {residue_path}")

# ===================================
# 主流程
# ===================================
if __name__ == "__main__":
    fasta_file = "/home/bio.usr07/Desktop/pretrain/model/389.fasta"
    out_dir = "/home/bio.usr07/Desktop/pretrain/model/prott5_embeddings1"
    os.makedirs(out_dir, exist_ok=True)

    seq_ids, sequences = read_fasta(fasta_file)
    features = Seq_to_vec(seq_ids, sequences)
    save_embeddings(features, out_dir)

    print("\n✅✅✅ ProtT5 mean + residue embeddings saved into two separate HDF5 files!")

