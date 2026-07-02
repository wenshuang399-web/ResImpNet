# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
import h5py

# Step 1: 定义 GIN 编码器
class GINEncoder(nn.Module):
    def __init__(self, in_dim=9, hidden=64, out_dim=128, num_layers=5):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            self.convs.append(GINConv(mlp))
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
        g_emb = global_add_pool(h, batch)   # 图级池化
        g_emb = self.proj(g_emb)
        return g_emb, h   # 返回图级和节点级嵌入

# Step 2: 加载 graphs.pt
graphs = torch.load("graphs2.pt")
print(f"共加载 {len(graphs)} 个分子图")

# 给每个 Data 加上行号 idx
for i, data in enumerate(graphs):
    data.idx = i

# Step 3: 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GINEncoder(in_dim=graphs[0].x.size(1), hidden=64, out_dim=128).to(device)
model.eval()

# Step 4: DataLoader
loader = DataLoader(graphs, batch_size=64, shuffle=False)

# Step 5: 分开保存到两个 HDF5 文件
with h5py.File("substrate_graph_embeddings.h5", "w") as g_h5, \
     h5py.File("substrate_node_embeddings.h5", "w") as n_h5:

    for batch in loader:
        batch = batch.to(device)
        g_emb, node_emb = model(batch.x, batch.edge_index, batch.batch)

        for b_idx in range(g_emb.size(0)):
            sid = str(batch.idx[b_idx].item())  # 用行号作为唯一 key

            # 图级 embedding
            if sid in g_h5:
                del g_h5[sid]
            g_h5.create_dataset(sid, data=g_emb[b_idx].detach().cpu().numpy(), compression="gzip")

            # 节点级 embedding
            if sid in n_h5:
                del n_h5[sid]
            node_mask = (batch.batch == b_idx)
            n_h5.create_dataset(sid, data=node_emb[node_mask].detach().cpu().numpy(), compression="gzip")

print("✅ 已分别保存到 substrate_graph_embeddings.h5 和 substrate_node_embeddings.h5")

