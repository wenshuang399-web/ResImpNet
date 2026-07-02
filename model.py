import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# GFE Block
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
        seq_proj = self.proj_seq(seq_embed)
        seq_proj_exp = seq_proj.unsqueeze(1).expand(-1, token_embed.size(1), -1)

        token_updated, _ = self.cross_token_to_seq(token_embed, seq_proj_exp, seq_proj_exp)
        token_embed = self.norm1(token_embed + token_updated)

        seq_query = seq_proj.unsqueeze(1)
        seq_updated, _ = self.cross_seq_to_token(seq_query, token_embed, token_embed)
        seq_embed = self.norm2(seq_query + seq_updated).squeeze(1)

        return seq_embed, token_embed


# ------------------------
# LFE Block
# ------------------------
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
        x = token_embed.permute(0, 2, 1)
        c3 = F.gelu(self.conv3(x))
        c5 = F.gelu(self.conv5(x))
        c7 = F.gelu(self.conv7(x))

        pooled = torch.cat([
            F.adaptive_avg_pool1d(c3, 1).squeeze(-1),
            F.adaptive_avg_pool1d(c5, 1).squeeze(-1),
            F.adaptive_avg_pool1d(c7, 1).squeeze(-1)
        ], dim=-1)

        pooled = self.ln_pooled(pooled)
        return self.fc(pooled)


# ------------------------
# SchemeBModel
# ------------------------
class SchemeBModel(nn.Module):
    def __init__(self, enzyme_dim=1024, substrate_dim=128, prost_dim=1024,
                 token_embed_dim=128, hidden_dim=128, num_heads=4,
                 num_gfe=2, num_lfe=2):
        super().__init__()

        self.token_embed_dim = token_embed_dim
        self.hidden_dim = hidden_dim

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

        self.prost_fc_single = nn.Sequential(
            nn.Linear(prost_dim, token_embed_dim),
            nn.LayerNorm(token_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.sub_prost_cross = nn.MultiheadAttention(embed_dim=token_embed_dim, num_heads=num_heads, batch_first=True)

        self.gfe_blocks = nn.ModuleList([
            GFEBlock(seq_input_dim=token_embed_dim, token_embed_dim=token_embed_dim, num_heads=num_heads)
            for _ in range(num_gfe)
        ])
        self.lfe_blocks = nn.ModuleList([
            LFEBlock(token_embed_dim, hidden_dim) for _ in range(num_lfe)
        ])

        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim + token_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def sinusoidal_positional_encoding(self, seq_len, dim, device):
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, enzyme_vec, substrate_vec, residue_tokens, mask, return_attention=False):
        device = enzyme_vec.device

        enzyme_embed = self.fc_enzyme(enzyme_vec)
        substrate_embed = self.fc_substrate(substrate_vec)

        token_embed = residue_tokens
        L = token_embed.size(1)
        pos = self.sinusoidal_positional_encoding(L, self.token_embed_dim, device)
        token_embed = token_embed + pos

        sub_for_attn = self.substrate_to_token(substrate_embed)
        sub_q = sub_for_attn.unsqueeze(1)

        key_padding_mask = (mask == 0)

        sub_updated, attn_weights = self.sub_prost_cross(
            sub_q, token_embed, token_embed,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        seq_embed = sub_updated.squeeze(1)

        for gfe in self.gfe_blocks:
            seq_embed, token_embed = gfe(seq_embed, token_embed)

        local_feat = sum([lfe(token_embed) for lfe in self.lfe_blocks]) / len(self.lfe_blocks)

        fusion_input = torch.cat([enzyme_embed, local_feat], dim=-1)
        output = self.fusion_fc(fusion_input).squeeze(1)

        if return_attention:
            attn_weights = attn_weights.squeeze(2).mean(dim=1)
            importance = attn_weights * mask
            minv = importance.min(dim=1, keepdim=True)[0]
            maxv = importance.max(dim=1, keepdim=True)[0]
            importance = (importance - minv) / (maxv - minv + 1e-8)
            return output, importance

        return output

