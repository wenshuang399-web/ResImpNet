#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from model import SchemeBModel   # ← 关键：从 model.py 导入模型

# ------------------------
# Dataset
# ------------------------
class KcatDatasetB(Dataset):
    def __init__(self, csv_file, enzyme_h5, substrate_h5, residue_h5):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.enzyme_h5 = h5py.File(enzyme_h5, "r")
        self.substrate_h5 = h5py.File(substrate_h5, "r")
        self.residue_h5 = h5py.File(residue_h5, "r")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        row_index = str(row["RowIndex"])
        uniprot_id = str(row["UniprotID"])
        label_value = row["Value"]

        enzyme_vec = torch.tensor(self.enzyme_h5[row_index][:], dtype=torch.float32)
        substrate_vec = torch.tensor(self.substrate_h5[row_index][:], dtype=torch.float32)

        pdb_id = uniprot_id + ".pdb"
        residue_tokens = torch.tensor(self.residue_h5[pdb_id][:], dtype=torch.float32)

        label = torch.tensor(label_value, dtype=torch.float32)

        return enzyme_vec, substrate_vec, residue_tokens, label, int(idx), uniprot_id

# ------------------------
# collate_fn
# ------------------------
def collate_fn(batch):
    enzyme_vecs, substrate_vecs, residue_tokens_list, labels, orig_indices, uniprot_ids = zip(*batch)

    enzyme_vecs = torch.stack(enzyme_vecs, dim=0)
    substrate_vecs = torch.stack(substrate_vecs, dim=0)
    labels = torch.stack(labels, dim=0)

    max_len = max([rt.shape[0] for rt in residue_tokens_list])
    hidden_dim = residue_tokens_list[0].shape[1]

    padded_tokens = torch.zeros(len(residue_tokens_list), max_len, hidden_dim)
    mask = torch.zeros(len(residue_tokens_list), max_len)

    for i, rt in enumerate(residue_tokens_list):
        L = rt.shape[0]
        padded_tokens[i, :L, :] = rt
        mask[i, :L] = 1.0

    return enzyme_vecs, substrate_vecs, padded_tokens, labels, mask, list(orig_indices), list(uniprot_ids)

# ------------------------
# Metrics
# ------------------------
def safe_mae(pred, target):
    return float(np.mean(np.abs(pred - target))) if pred.size else float('nan')

def safe_rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2))) if pred.size else float('nan')

def safe_r2(pred, target):
    if not target.size:
        return float('nan')
    denom = np.sum((target - np.mean(target)) ** 2)
    if denom == 0:
        return float('nan')
    return float(1 - np.sum((target - pred) ** 2) / denom)

def safe_pcc(pred, target):
    try:
        if not pred.size or np.std(pred) == 0 or np.std(target) == 0:
            return float('nan')
        from scipy.stats import pearsonr
        return float(pearsonr(pred, target)[0])
    except Exception:
        return float('nan')

# ------------------------
# Mutation suggestion
# ------------------------
def suggest_mutations(importance, sequence, top_n=30):
    L = len(sequence)
    if importance.shape[0] != L:
        if importance.shape[0] < L:
            tmp = np.zeros(L)
            tmp[:importance.shape[0]] = importance
            importance = tmp
        else:
            importance = importance[:L]

    ranked = np.argsort(-importance)[:top_n]
    suggestions = []
    for idx0 in ranked:
        aa = sequence[idx0]
        imp = float(importance[idx0])

        if aa in ['A', 'G', 'S']:
            sug = "尝试换成大体积疏水残基 (F, Y, W)"
        elif aa in ['D', 'E']:
            sug = "尝试换成正电荷残基 (K, R)"
        elif aa in ['K', 'R']:
            sug = "尝试换成负电荷残基 (D, E)"
        elif aa in ['P']:
            sug = "尝试换成非环状残基以增加柔性 (A, G)"
        else:
            sug = "尝试换成极性残基 (N, Q) 或疏水残基 (V, I)"

        suggestions.append((idx0 + 1, aa, imp, sug))

    return suggestions

# ------------------------
# Args
# ------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_csv', type=str, default="DEKP-kcat_dataset_fixed.csv")
    parser.add_argument('--enzyme_h5', type=str, default="prott5_mean_embeddings.h5")
    parser.add_argument('--substrate_h5', type=str, default="substrate_graph_embeddings.h5")
    parser.add_argument('--residue_h5', type=str, default="protein_residue_embeddings.h5")
    parser.add_argument('--out_dir', type=str, default="SchemeB_importance_out_sinpos")
    parser.add_argument('--num_gfe', type=int, default=2)
    parser.add_argument('--num_lfe', type=int, default=2)
    parser.add_argument('--token_embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    return parser.parse_args()

# ------------------------
# Training loop
# ------------------------
def train_random_split():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = KcatDatasetB(args.data_csv, args.enzyme_h5, args.substrate_h5, args.residue_h5)
    df = pd.read_csv(args.data_csv).reset_index(drop=True)

    out_dir = os.path.join(args.out_dir, f"lr_{args.lr:.0e}")
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, "results_sinpos.txt")
    all_pred_csv = os.path.join(out_dir, "all_predictions_sinpos.csv")

    with open(log_file, "w") as f:
        f.write("# Using sinusoidal positional encoding\n")

    all_predictions = []
    n = len(dataset)
    indices = np.arange(n)

    for round_idx in range(5):
        print(f"\n===== Round {round_idx+1} =====")

        train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=round_idx)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=round_idx)

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        # infer dims
        enzyme_dim = dataset.enzyme_h5[str(dataset.df.iloc[0]['RowIndex'])][:].shape[0]
        substrate_dim = dataset.substrate_h5[str(dataset.df.iloc[0]['RowIndex'])][:].shape[0]
        first_res_key = list(dataset.residue_h5.keys())[0]
        prost_dim = dataset.residue_h5[first_res_key].shape[1]

        model = SchemeBModel(
            enzyme_dim=enzyme_dim,
            substrate_dim=substrate_dim,
            prost_dim=prost_dim,
            token_embed_dim=args.token_embed_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_gfe=args.num_gfe,
            num_lfe=args.num_lfe
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience)
        loss_fn = nn.MSELoss()

        # ------------------------
        # Epoch loop
        # ------------------------
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            train_preds_list = []
            train_labels_list = []

            for enzyme_vec, substrate_vec, residue_tokens, labels, mask, orig_idxs, uids in train_loader:
                enzyme_vec = enzyme_vec.to(device)
                substrate_vec = substrate_vec.to(device)
                residue_tokens = residue_tokens.to(device)
                labels = labels.to(device)
                mask = mask.to(device)

                preds = model(enzyme_vec, substrate_vec, residue_tokens, mask)
                loss = loss_fn(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                train_preds_list.append(preds.detach().cpu().numpy())
                train_labels_list.append(labels.detach().cpu().numpy())

            train_loss = total_loss / len(train_loader)

            # validation
            model.eval()
            val_loss_acc = 0.0
            val_preds_list = []
            val_labels_list = []

            with torch.no_grad():
                for enzyme_vec, substrate_vec, residue_tokens, labels, mask, orig_idxs, uids in val_loader:
                    enzyme_vec = enzyme_vec.to(device)
                    substrate_vec = substrate_vec.to(device)
                    residue_tokens = residue_tokens.to(device)
                    labels = labels.to(device)
                    mask = mask.to(device)

                    preds = model(enzyme_vec, substrate_vec, residue_tokens, mask)
                    val_loss_acc += loss_fn(preds, labels).item()
                    val_preds_list.append(preds.cpu().numpy())
                    val_labels_list.append(labels.cpu().numpy())

            val_loss = val_loss_acc / len(val_loader)
            scheduler.step(val_loss)

            print(f"Round {round_idx+1} Epoch {epoch} | train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        # ------------------------
        # Save model
        # ------------------------
        save_path = os.path.join(out_dir, f"model_round{round_idx+1}_sinpos.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model: {save_path}")

        # ------------------------
        # Test + importance extraction
        # ------------------------
        model.eval()
        round_imp_dict = []

        with torch.no_grad():
            for enzyme_vec, substrate_vec, residue_tokens, labels, mask, orig_idxs, uids in test_loader:
                enzyme_vec = enzyme_vec.to(device)
                substrate_vec = substrate_vec.to(device)
                residue_tokens = residue_tokens.to(device)
                mask = mask.to(device)

                preds, imp = model(enzyme_vec, substrate_vec, residue_tokens, mask, return_attention=True)

                for i in range(imp.shape[0]):
                    round_imp_dict.append({
                        "orig_idx": orig_idxs[i],
                        "uniprot_id": uids[i],
                        "importance": imp[i].cpu().numpy(),
                        "pred": float(preds[i].cpu().item()),
                        "label": float(labels[i].cpu().item())
                    })

        np.savez(os.path.join(out_dir, f"importance_round{round_idx+1}_sinpos.npz"), data=round_imp_dict)

        # ------------------------
        # Mutation suggestions
        # ------------------------
        if "Sequence" in df.columns:
            for sample in round_imp_dict:
                seq = df.iloc[sample["orig_idx"]]["Sequence"]
                imp_vec = sample["importance"]
                uid = sample["uniprot_id"]
                orig_idx_val = sample["orig_idx"]

                mut_sugg = suggest_mutations(imp_vec, seq, top_n=30)
                sample_csv = os.path.join(out_dir, f"round{round_idx+1}_sample{orig_idx_val}_{uid}_mutation_suggestions.csv")
                pd.DataFrame(mut_sugg, columns=['ResidueIndex','Residue','Importance','Suggestion']).to_csv(sample_csv, index=False)

        # ------------------------
        # Save predictions
        # ------------------------
        for sample in round_imp_dict:
            rowindex_val = int(df.iloc[sample["orig_idx"]]["RowIndex"])
            ec_val = df.iloc[sample["orig_idx"]].get("ECNumber", "")
            all_predictions.append([
                round_idx+1,
                sample["orig_idx"],
                rowindex_val,
                ec_val,
                sample["pred"],
                sample["label"]
            ])

        pd.DataFrame(all_predictions, columns=['Run','OrigIndex','RowIndex','ECNumber','Predicted','Actual']).to_csv(all_pred_csv, index=False)


if __name__ == "__main__":
    train_random_split()

