#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_schemeB_with_importance_improved_sinpos.py

- 保留改进版的 GFE / LFE / cross-attention 架构
- 在 cross-attention（substrate -> residue tokens）处提取 attention weights
- forward 支持 return_attention=True，返回归一化的 per-residue importance
- 使用正弦/余弦位置编码（sinusoidal positional encoding），替换原有 learned prost_pos
- test 阶段保存 importance（.npz）并为每个测试样本生成 mutation suggestions CSV
- 不对 5 轮结果做跨轮平均（每轮单独保存 importance）
"""

import os
import time
import argparse
import math
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# ------------------------
# Dataset 类（方案 B）
# ------------------------
class KcatDatasetB(Dataset):
    def __init__(self, csv_file, enzyme_h5, substrate_h5, residue_h5):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        # open HDF5 files (read-only)
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

        # Basic existence checks (raise informative error if missing)
        if row_index not in self.enzyme_h5:
            raise KeyError(f"RowIndex {row_index} not found in enzyme_h5")
        if row_index not in self.substrate_h5:
            raise KeyError(f"RowIndex {row_index} not found in substrate_h5")

        enzyme_vec = torch.tensor(self.enzyme_h5[row_index][:], dtype=torch.float32)
        substrate_vec = torch.tensor(self.substrate_h5[row_index][:], dtype=torch.float32)

        pdb_id = uniprot_id + ".pdb"
        if pdb_id not in self.residue_h5:
            raise KeyError(f"{pdb_id} not found in residue_h5")
        residue_tokens = torch.tensor(self.residue_h5[pdb_id][:], dtype=torch.float32)

        label = torch.tensor(label_value, dtype=torch.float32)

        # return also original index and UniprotID for alignment and downstream analysis
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

    padded_tokens = torch.zeros(len(residue_tokens_list), max_len, hidden_dim, dtype=torch.float32)
    mask = torch.zeros(len(residue_tokens_list), max_len, dtype=torch.float32)

    for i, rt in enumerate(residue_tokens_list):
        length = rt.shape[0]
        padded_tokens[i, :length, :] = rt
        mask[i, :length] = 1.0

    return enzyme_vecs, substrate_vecs, padded_tokens, labels, mask, list(orig_indices), list(uniprot_ids)

# ------------------------
# Metrics (safe)
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
# Enhanced GFEBlock and LFEBlock (from your improved script)
# ------------------------
class GFEBlock(nn.Module):
    def __init__(self, seq_input_dim, token_embed_dim, num_heads):
        super().__init__()
        self.cross_token_to_seq = nn.MultiheadAttention(embed_dim=token_embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_seq_to_token = nn.MultiheadAttention(embed_dim=token_embed_dim, num_heads=num_heads, batch_first=True)

        if seq_input_dim != token_embed_dim:
            self.proj_seq = nn.Sequential(
                nn.Linear(seq_input_dim, token_embed_dim),
                nn.LayerNorm(token_embed_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            self.proj_seq = nn.Identity()

        self.norm1 = nn.LayerNorm(token_embed_dim)
        self.norm2 = nn.LayerNorm(token_embed_dim)

    def forward(self, seq_embed, token_embed):
        # seq_embed: (B, D) ; token_embed: (B, L, D)
        seq_proj = self.proj_seq(seq_embed)  # (B, D)
        seq_proj_exp = seq_proj.unsqueeze(1).expand(-1, token_embed.size(1), -1)  # (B, L, D)

        # token as query, seq as key/value -> update token
        token_updated, _ = self.cross_token_to_seq(token_embed, seq_proj_exp, seq_proj_exp)
        token_embed = self.norm1(token_embed + token_updated)

        # seq as query, token as key/value -> update seq
        seq_query = seq_proj.unsqueeze(1)  # (B,1,D)
        seq_updated, _ = self.cross_seq_to_token(seq_query, token_embed, token_embed)
        seq_embed = self.norm2(seq_query + seq_updated).squeeze(1)  # (B,D)

        return seq_embed, token_embed

class LFEBlock(nn.Module):
    def __init__(self, token_embed_dim, hidden_dim):
        super().__init__()
        self.conv3 = nn.Conv1d(token_embed_dim, hidden_dim, 3, padding=1)
        self.conv5 = nn.Conv1d(token_embed_dim, hidden_dim, 5, padding=2)
        self.conv7 = nn.Conv1d(token_embed_dim, hidden_dim, 7, padding=3)

        self.ln_pooled = nn.LayerNorm(hidden_dim * 3)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, token_embed_dim),
            nn.LayerNorm(token_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, token_embed):
        # token_embed: [B, L, D]
        x = token_embed.permute(0, 2, 1)  # [B, D, L]
        c3 = F.gelu(self.conv3(x))
        c5 = F.gelu(self.conv5(x))
        c7 = F.gelu(self.conv7(x))

        pooled = torch.cat([
            F.adaptive_avg_pool1d(c3, 1).squeeze(-1),
            F.adaptive_avg_pool1d(c5, 1).squeeze(-1),
            F.adaptive_avg_pool1d(c7, 1).squeeze(-1)
        ], dim=-1)  # [B, hidden_dim*3]

        pooled = self.ln_pooled(pooled)
        return self.fc(pooled)  # [B, token_embed_dim]

# ------------------------
# SchemeBModel (improved) with sinusoidal positional encoding
# ------------------------
class SchemeBModel(nn.Module):
    def __init__(self, enzyme_dim=1024, substrate_dim=128, prost_dim=1024,
                 token_embed_dim=128, hidden_dim=128, num_heads=4,
                 num_gfe=2, num_lfe=2, prost_seq_len=1):
        super().__init__()

        self.token_embed_dim = token_embed_dim
        self.hidden_dim = hidden_dim
        self.prost_seq_len = prost_seq_len  # kept for compatibility but not used for learned pos

        self.fc_enzyme = nn.Sequential(
            nn.Linear(enzyme_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.fc_substrate = nn.Sequential(
            nn.Linear(substrate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.substrate_to_token = nn.Sequential(
            nn.Linear(hidden_dim, token_embed_dim),
            nn.LayerNorm(token_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # If prost provided as single vector, project to token_embed_dim
        self.prost_fc_single = nn.Sequential(
            nn.Linear(prost_dim, token_embed_dim),
            nn.LayerNorm(token_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # substrate <-> prost cross-attention (we will extract attn weights here)
        self.sub_prost_cross = nn.MultiheadAttention(embed_dim=token_embed_dim, num_heads=num_heads, batch_first=True)

        # NOTE: removed learned prost_pos parameter; use sinusoidal positional encoding instead

        # GFE and LFE stacks
        self.gfe_blocks = nn.ModuleList([
            GFEBlock(seq_input_dim=token_embed_dim, token_embed_dim=token_embed_dim, num_heads=num_heads)
            for _ in range(num_gfe)
        ])
        self.lfe_blocks = nn.ModuleList([
            LFEBlock(token_embed_dim, hidden_dim) for _ in range(num_lfe)
        ])

        # fusion and output
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim + token_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def sinusoidal_positional_encoding(self, seq_len, dim, device):
        """
        Return tensor of shape (1, seq_len, dim) with sinusoidal positional encodings.
        """
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / dim))  # (dim/2,)
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, dim)

    def forward(self, enzyme_vec, substrate_vec, residue_tokens, mask, return_attention=False):
        """
        enzyme_vec: (B, enzyme_dim)
        substrate_vec: (B, substrate_dim)
        residue_tokens: (B, L, D_token)  -- D_token should equal token_embed_dim or prost_dim projected
        mask: (B, L) float mask with 1 for valid tokens, 0 for padding
        return_attention: if True, also return per-residue importance (B, L)
        """
        device = enzyme_vec.device
        enzyme_embed = self.fc_enzyme(enzyme_vec)           # (B, hidden_dim)
        substrate_embed = self.fc_substrate(substrate_vec)  # (B, hidden_dim)

        # prepare prost token embeddings from residue_tokens
        token_embed = residue_tokens  # assume residue_tokens already in token_embed_dim; if not, user should project upstream

        # add sinusoidal positional encoding dynamically
        prost_L = token_embed.size(1)
        pos = self.sinusoidal_positional_encoding(prost_L, self.token_embed_dim, device)  # (1, L, D)
        token_embed = token_embed + pos  # broadcasting over batch

        # substrate -> token space
        sub_for_attn = self.substrate_to_token(substrate_embed)  # (B, token_embed_dim)
        sub_q = sub_for_attn.unsqueeze(1)  # (B,1,D)

        # key_padding_mask expects True for positions that should be ignored
        key_padding_mask = (mask == 0)  # (B, L) bool

        # cross-attend: substrate query, prost keys/values
        # request weights so we can compute importance
        sub_updated, attn_weights = self.sub_prost_cross(sub_q, token_embed, token_embed, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False)
        seq_embed = sub_updated.squeeze(1)  # (B, D)
        # token_embed remains (B, L, D)

        # GFE blocks (seq <-> token bidirectional)
        for gfe in self.gfe_blocks:
            seq_embed, token_embed = gfe(seq_embed, token_embed)

        # LFE using prost token embeddings (no 3Di tokens)
        if token_embed.size(1) > 0:
            local_feat = sum([lfe(token_embed) for lfe in self.lfe_blocks]) / len(self.lfe_blocks)
        else:
            local_feat = seq_embed

        fusion_input = torch.cat([enzyme_embed, local_feat], dim=-1)
        output = self.fusion_fc(fusion_input).squeeze(1)

        if return_attention:
            # attn_weights shape: (B, num_heads, query_len=1, seq_len=L)
            # squeeze query dim -> (B, num_heads, L)
            attn_weights = attn_weights.squeeze(2)
            # average across heads -> (B, L)
            importance = attn_weights.mean(dim=1)
            # mask padding positions
            importance = importance * mask
            # per-sample min-max normalization to [0,1] (avoid divide by zero)
            minv = importance.min(dim=1, keepdim=True)[0]
            maxv = importance.max(dim=1, keepdim=True)[0]
            importance = (importance - minv) / (maxv - minv + 1e-8)
            return output, importance

        return output

# ------------------------
# Mutation suggestion (simple rule-based)
# ------------------------
def suggest_mutations(importance, sequence, top_n=30):
    """
    importance: 1D numpy array length L
    sequence: string of length L (1-letter amino acids)
    returns list of tuples: (ResidueIndex (1-based), Residue, Importance, Suggestion)
    """
    L = len(sequence)
    # ensure importance length matches sequence length (if not, truncate/pad)
    if importance.shape[0] != L:
        # if importance shorter, pad with zeros; if longer, truncate
        if importance.shape[0] < L:
            imp = np.zeros(L, dtype=float)
            imp[:importance.shape[0]] = importance
            importance = imp
        else:
            importance = importance[:L]

    ranked = np.argsort(-importance)[:top_n]
    suggestions = []
    for idx0 in ranked:
        aa = sequence[idx0]
        imp = float(importance[idx0])
        # simple biochemical heuristics for suggestion (can be extended)
        if aa in ['A', 'G', 'S']:
            suggestion = "尝试换成大体积疏水残基 (F, Y, W)"
        elif aa in ['D', 'E']:
            suggestion = "尝试换成正电荷残基 (K, R)"
        elif aa in ['K', 'R']:
            suggestion = "尝试换成负电荷残基 (D, E)"
        elif aa in ['P']:
            suggestion = "尝试换成非环状残基以增加柔性 (A, G)"
        else:
            suggestion = "尝试换成极性残基 (N, Q) 或疏水残基 (V, I) 视位置而定"
        # convert to 1-based residue index for user-friendly output
        suggestions.append((int(idx0 + 1), aa, imp, suggestion))
    return suggestions

# ------------------------
# Argument parsing
# ------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train improved SchemeBModel with importance output and mutation suggestions")
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
# Training loop with importance extraction and mutation suggestions
# ------------------------
def train_random_split():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset = KcatDatasetB(args.data_csv, args.enzyme_h5, args.substrate_h5, args.residue_h5)
    df = pd.read_csv(args.data_csv).reset_index(drop=True)

    # output dir
    out_dir = os.path.join(args.out_dir, f"lr_{args.lr:.0e}")
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, "results_sinpos.txt")
    per_run_csv = os.path.join(out_dir, "per_run_metrics_sinpos.csv")
    all_pred_csv = os.path.join(out_dir, "all_predictions_sinpos.csv")

    with open(log_file, "w") as f:
        f.write("# Using sinusoidal positional encoding (方案: sinpos)\n")
        f.write(f"# Params: lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}, weight_decay={args.weight_decay}\n")
        f.write("Round\tEpoch\tTime(sec)\tLoss_train\tMAE_train\tRMSE_train\tR2_train\t"
                "loss_val\tMAE_val\tRMSE_val\tR2_val\tMAE_test\tRMSE_test\tR2_test\tPCC_test\tLr\tSchedulerInfo\n")

    all_metrics = []
    all_predictions = []

    n = len(dataset)
    indices = np.arange(n)

    for round_idx in range(5):
        print(f"\n===== Round {round_idx+1} =====")

        train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=round_idx, shuffle=True)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=round_idx, shuffle=True)

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        # infer dims from dataset HDF5 entries
        enzyme_dim = dataset.enzyme_h5[str(dataset.df.iloc[0]['RowIndex'])][:].shape[0]
        substrate_dim = dataset.substrate_h5[str(dataset.df.iloc[0]['RowIndex'])][:].shape[0]
        # prost_dim and prost_seq_len from first residue entry
        first_res_key = list(dataset.residue_h5.keys())[0]
        prost_seq_len = dataset.residue_h5[first_res_key].shape[0]
        prost_dim = dataset.residue_h5[first_res_key].shape[1]

        model = SchemeBModel(
            enzyme_dim=enzyme_dim,
            substrate_dim=substrate_dim,
            prost_dim=prost_dim,
            token_embed_dim=args.token_embed_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_gfe=args.num_gfe,
            num_lfe=args.num_lfe,
            prost_seq_len=prost_seq_len
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=False)
        loss_fn = nn.MSELoss()

        history = {
            'train_loss': [], 'val_loss': [], 'test_mae': [],
            'train_mae': [], 'val_mae': [], 'train_rmse': [], 'val_rmse': [], 'test_rmse': []
        }

        test_sample_origidxs = []
        test_ecs = []
        test_preds = np.array([])
        test_labels = np.array([])

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            total_loss = 0.0
            train_preds_list = []
            train_labels_list = []

            for enzyme_vec, substrate_vec, residue_tokens, labels, mask, orig_idxs, ec_nums in train_loader:
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

            train_loss = total_loss / max(1, len(train_loader))
            train_preds = np.concatenate(train_preds_list) if train_preds_list else np.array([])
            train_labels = np.concatenate(train_labels_list) if train_labels_list else np.array([])

            mae_train = safe_mae(train_preds, train_labels) if train_preds.size else float('nan')
            rmse_train = safe_rmse(train_preds, train_labels) if train_preds.size else float('nan')
            r2_train = safe_r2(train_preds, train_labels) if train_preds.size else float('nan')

            # validation
            model.eval()
            val_preds_list = []
            val_labels_list = []
            val_loss_acc = 0.0
            with torch.no_grad():
                for enzyme_vec, substrate_vec, residue_tokens, labels, mask, orig_idxs, ec_nums in val_loader:
                    enzyme_vec = enzyme_vec.to(device)
                    substrate_vec = substrate_vec.to(device)
                    residue_tokens = residue_tokens.to(device)
                    labels = labels.to(device)
                    mask = mask.to(device)

                    preds = model(enzyme_vec, substrate_vec, residue_tokens, mask)
                    val_loss_acc += loss_fn(preds, labels).item()
                    val_preds_list.append(preds.cpu().numpy())
                    val_labels_list.append(labels.cpu().numpy())

            val_loss = val_loss_acc / max(1, len(val_loader))
            val_preds = np.concatenate(val_preds_list) if val_preds_list else np.array([])
            val_labels = np.concatenate(val_labels_list) if val_labels_list else np.array([])

            mae_val = safe_mae(val_preds, val_labels) if val_preds.size else float('nan')
            rmse_val = safe_rmse(val_preds, val_labels) if val_preds.size else float('nan')
            r2_val = safe_r2(val_preds, val_labels) if val_preds.size else float('nan')

            # test evaluation (collect predictions)
            test_preds_list = []
            test_labels_list = []
            test_sample_origidxs = []
            test_ecs = []
            with torch.no_grad():
                for enzyme_vec, substrate_vec, residue_tokens, labels, mask, orig_idxs, ec_nums in test_loader:
                    enzyme_vec = enzyme_vec.to(device)
                    substrate_vec = substrate_vec.to(device)
                    residue_tokens = residue_tokens.to(device)
                    labels = labels.to(device)
                    mask = mask.to(device)

                    preds = model(enzyme_vec, substrate_vec, residue_tokens, mask)
                    test_preds_list.append(preds.cpu().numpy())
                    test_labels_list.append(labels.cpu().numpy())
                    test_sample_origidxs.extend(orig_idxs)
                    test_ecs.extend(ec_nums)

            test_preds = np.concatenate(test_preds_list) if test_preds_list else np.array([])
            test_labels = np.concatenate(test_labels_list) if test_labels_list else np.array([])

            mae_test = safe_mae(test_preds, test_labels) if test_preds.size else float('nan')
            rmse_test = safe_rmse(test_preds, test_labels) if test_preds.size else float('nan')
            r2_test = safe_r2(test_preds, test_labels) if test_labels.size else float('nan')
            pcc_test = safe_pcc(test_preds, test_labels) if test_preds.size else float('nan')

            lr_now = optimizer.param_groups[0]['lr']
            elapsed = time.time() - epoch_start

            # scheduler step (ReduceLROnPlateau uses val_loss)
            scheduler_info_line = ""
            if scheduler is not None:
                scheduler.step(val_loss)
                scheduler_info_line = f"ReduceLROnPlateau_step(val_loss={val_loss:.6f})"

            # log
            with open(log_file, "a") as f:
                f.write(f"{round_idx+1}\t{epoch}\t{elapsed:.2f}\t{train_loss:.6f}\t{mae_train:.6f}\t{rmse_train:.6f}\t{r2_train:.6f}\t"
                        f"{val_loss:.6f}\t{mae_val:.6f}\t{rmse_val:.6f}\t{r2_val:.6f}\t"
                        f"{mae_test:.6f}\t{rmse_test:.6f}\t{r2_test:.6f}\t{pcc_test:.6f}\t{lr_now:.8f}\t{scheduler_info_line}\n")

            print(f"Round {round_idx+1} Epoch {epoch} | train_loss={train_loss:.6f} val_loss={val_loss:.6f} test_mae={mae_test:.6f} lr={lr_now:.8f}")

            # update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(mae_train)
            history['val_mae'].append(mae_val)
            history['test_mae'].append(mae_test)
            history['train_rmse'].append(rmse_train)
            history['val_rmse'].append(rmse_val)
            history['test_rmse'].append(rmse_test)

        # end epoch loop: save model for this round
        save_path = os.path.join(out_dir, f"model_round{round_idx+1}_sinpos.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model (sinusoidal pos encoding): {save_path}")

 # ===== test + importance extraction =====
        model.eval()
        round_imp_dict = []
        round_origidxs = []   # 定义列表
        round_uniprot = []

        with torch.no_grad():
            for enzyme_vec, substrate_vec, residue_tokens, labels, mask, orig_idxs, uids in test_loader:
                enzyme_vec = enzyme_vec.to(device)
                substrate_vec = substrate_vec.to(device)
                residue_tokens = residue_tokens.to(device)
                mask = mask.to(device)

                preds, imp = model(enzyme_vec, substrate_vec, residue_tokens, mask, return_attention=True)

        # 遍历 batch 内每个样本，单独保存 importance
                for i in range(imp.shape[0]):
                    round_imp_dict.append({
                        "orig_idx": orig_idxs[i],
                        "uniprot_id": uids[i],
                        "importance": imp[i].cpu().numpy(),   # 每个样本的 importance 向量
                        "pred": float(preds[i].cpu().item()),
                        "label": float(labels[i].cpu().item())
                    })
                    round_origidxs.append(orig_idxs[i])   # 保存索引
                    round_uniprot.append(uids[i])         # 保存 UniprotID
# 保存为 npz 文件（字典列表）
        imp_save_path = os.path.join(out_dir, f"importance_round{round_idx+1}_sinpos.npz")
        np.savez(imp_save_path, data=round_imp_dict)
        print(f"Saved importance (sinusoidal pos encoding): {imp_save_path}")

        # save per-sample predictions for this round (for record)
        for i, sample in enumerate(round_imp_dict):
            rowindex_val = int(df.iloc[sample["orig_idx"]]["RowIndex"]) if "RowIndex" in df.columns else sample["orig_idx"]
            ec_val = df.iloc[sample["orig_idx"]].get("ECNumber", "") if "ECNumber" in df.columns else ""
            pred_val = sample["pred"]
            actual_val = sample["label"]
            all_predictions.append([round_idx+1, i, rowindex_val, ec_val, pred_val, actual_val])

        pd.DataFrame(all_predictions, columns=['Run','SampleIdx','RowIndex','ECNumber','Predicted','Actual']).to_csv(all_pred_csv, index=False)

        # generate mutation suggestions CSVs for each test sample that has sequence info
        # require that dataset.df contains a "Sequence" column with 1-letter codes
        if "Sequence" in df.columns and len(round_imp_dict) > 0:
            suggestions_rows = []
            for sample in round_imp_dict:
                seq = df.iloc[sample["orig_idx"]]["Sequence"]
                imp_vec = sample["importance"]
                # ensure sequence length matches importance length; if not, handle gracefully
                if len(seq) == 0:
                    continue
                mut_sugg = suggest_mutations(imp_vec, seq, top_n=30)
                # write per-sample CSV and also aggregate
                uid = sample["uniprot_id"]
                orig_idx_val = sample["orig_idx"]
                sample_prefix = os.path.join(out_dir, f"round{round_idx+1}_sample{orig_idx_val}_{uid}_sinpos")
                sample_csv = sample_prefix + "_mutation_suggestions.csv"
                pd.DataFrame(mut_sugg, columns=['ResidueIndex','Residue','Importance','Suggestion']).to_csv(sample_csv, index=False)
                # aggregate rows for summary CSV
                for r in mut_sugg:
                    suggestions_rows.append([round_idx+1, orig_idx_val, uid, r[0], r[1], r[2], r[3]])
            # save aggregated suggestions for this round
            if suggestions_rows:
                sugg_df = pd.DataFrame(suggestions_rows, columns=['Run','OrigIndex','UniprotID','ResidueIndex','Residue','Importance','Suggestion'])
                sugg_df.to_csv(os.path.join(out_dir, f"mutation_suggestions_round{round_idx+1}_sinpos.csv"), index=False)
                print(f"Saved mutation suggestions for round {round_idx+1} (per-sample and aggregated).")
            else:
                print("No mutation suggestions generated (no sequences or empty importance).")
        else:
            print("Sequence column not found in CSV or no importance data; skipping mutation suggestion generation.")

        # collect per-run metrics (use final test metrics)
        all_metrics.append([round_idx+1, float(r2_test) if not np.isnan(r2_test) else None,
                            float(rmse_test) if not np.isnan(rmse_test) else None,
                            float(mae_test) if not np.isnan(mae_test) else None,
                            float(pcc_test) if not np.isnan(pcc_test) else None])
        pd.DataFrame(all_metrics, columns=['Run','R2','RMSE','MAE','PCC']).to_csv(per_run_csv, index=False)

    print("Training complete. All rounds finished.")

if __name__ == "__main__":
    train_random_split()

