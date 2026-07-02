import torch
import h5py
import json
import gvp.data
from gvp.models import MQAModel
from torch_geometric.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 JSON 文件
with open("protein_structures2.json") as f:
    data_list = json.load(f)

dataset = gvp.data.ProteinGraphDataset(data_list)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# 初始化模型
model = MQAModel(
    node_in_dim=(6,3), node_h_dim=(128,16),
    edge_in_dim=(32,1), edge_h_dim=(32,1),
    seq_in=True
).to(device)
model.eval()

# 分别打开两个文件：一个保存残基级，一个保存蛋白质级
with h5py.File("protein_residue_embeddings1.h5", "w") as h5f_residue, \
     h5py.File("protein_protein_embeddings1.h5", "w") as h5f_protein:

    for batch in loader:
        batch = batch.to(device)
        nodes = (batch.node_s, batch.node_v)
        edges = (batch.edge_s, batch.edge_v)

        # 调用模型，解包三个返回值
        residue_emb, protein_emb, score = model(
            nodes, batch.edge_index, edges,
            seq=batch.seq, batch=batch.batch
        )

        # 遍历 batch 内的蛋白
        for i, name in enumerate(batch.name):
            dataset_name = name  # ✅ 用唯一的 PDB ID

            # 如果已经存在，先删除再写入（避免重复报错）
            if dataset_name in h5f_residue:
                del h5f_residue[dataset_name]
            if dataset_name in h5f_protein:
                del h5f_protein[dataset_name]

            # 残基级 embedding (seq_len, hidden_dim)
            res_emb_np = residue_emb[batch.batch == i].detach().cpu().numpy()
            dset_res = h5f_residue.create_dataset(dataset_name, data=res_emb_np)
            dset_res.attrs["seq_len"] = res_emb_np.shape[0]

            # 蛋白质级 embedding (hidden_dim,)
            prot_emb_np = protein_emb[i].detach().cpu().numpy()
            dset_prot = h5f_protein.create_dataset(dataset_name, data=prot_emb_np)
            dset_prot.attrs["seq_len"] = res_emb_np.shape[0]

            # 打印信息
            print(f"{dataset_name} residue_emb shape:", res_emb_np.shape)
            print(f"{dataset_name} protein_emb shape:", prot_emb_np.shape)
            print(f"{dataset_name} score:", score[i].item())

print("✅ 残基级 embedding 已保存到 protein_residue_embeddings1.h5")
print("✅ 蛋白质级 embedding 已保存到 protein_protein_embeddings1.h5")

