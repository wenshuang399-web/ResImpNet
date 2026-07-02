#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import pandas as pd
import h5py
import numpy as np

from train_schemeB_with_sinpos import SchemeBModel, suggest_mutations

# 文件路径
enzyme_h5 = "prott5_mean_embeddings2.h5"
substrate_h5 = "substrate_graph_embeddings2.h5"
residue_h5 = "protein_residue_embeddings2.h5"
csv_file = "389.csv"
checkpoint_path = "model_round1_sinpos.pt"

# 读取 CSV（只有一条序列）
df = pd.read_csv(csv_file)
sequence = df.iloc[0]["Sequence"]

# 打开 HDF5 文件并取第一个键
enzyme_h5f = h5py.File(enzyme_h5, "r")
substrate_h5f = h5py.File(substrate_h5, "r")
residue_h5f = h5py.File(residue_h5, "r")

enzyme_key = list(enzyme_h5f.keys())[0]
substrate_key = list(substrate_h5f.keys())[0]
residue_key = list(residue_h5f.keys())[0]

enzyme_vec = torch.tensor(enzyme_h5f[enzyme_key][:], dtype=torch.float32).unsqueeze(0)
substrate_vec = torch.tensor(substrate_h5f[substrate_key][:], dtype=torch.float32).unsqueeze(0)
residue_tokens = torch.tensor(residue_h5f[residue_key][:], dtype=torch.float32).unsqueeze(0)
mask = torch.ones(1, residue_tokens.shape[1], dtype=torch.float32)

# 推理设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 推断维度
enzyme_dim = enzyme_vec.shape[1]
substrate_dim = substrate_vec.shape[1]
prost_dim = residue_tokens.shape[2]
prost_seq_len = residue_tokens.shape[1]

# 构建模型并加载 checkpoint
model = SchemeBModel(
    enzyme_dim=enzyme_dim,
    substrate_dim=substrate_dim,
    prost_dim=prost_dim,
    token_embed_dim=128,
    hidden_dim=128,
    num_heads=4,
    num_gfe=2,
    num_lfe=2,
    prost_seq_len=prost_seq_len
).to(device)

state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state, strict=False)
model.eval()

# 推理
with torch.no_grad():
    preds, importance = model(enzyme_vec.to(device),
                              substrate_vec.to(device),
                              residue_tokens.to(device),
                              mask.to(device),
                              return_attention=True)

pred_val = preds.item()
imp_vec = importance.squeeze(0).cpu().numpy()

print(f"Predicted value: {pred_val:.6f}")

# 保存 importance 对应的残基
rows = []
for i, (aa, score) in enumerate(zip(sequence, imp_vec), start=1):
    rows.append([i, aa, score, pred_val])

out_df = pd.DataFrame(rows, columns=["ResidueIndex","Residue","Importance","Predicted"])
out_df.to_csv("389prediction_importance.csv", index=False)
print("Saved prediction_importance.csv")

# 生成并保存 mutation suggestions
mut_sugg = suggest_mutations(imp_vec, sequence, top_n=30)
mut_df = pd.DataFrame(mut_sugg, columns=["ResidueIndex","Residue","Importance","Suggestion"])
mut_df.to_csv("389prediction_mutations.csv", index=False)
print("Saved prediction_mutations.csv")

# 打印前10条 mutation suggestions
print("\nTop mutation suggestions:")
for r in mut_sugg[:10]:
    print(f"Residue {r[0]} ({r[1]}), importance={r[2]:.3f}, suggestion={r[3]}")

