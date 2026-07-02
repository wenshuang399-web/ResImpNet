# coding=utf-8
import pandas as pd
import torch
from torch_geometric.utils.smiles import from_smiles
from torch_geometric.data import Data

# Step 1: 读取 CSV 文件
csv_path = "389.csv"
df = pd.read_csv(csv_path)

# Step 2: 转换 SMILES 为 PyG Data 对象
graphs = []
for i, row in df.iterrows():
    smiles = row["Smiles"]
    try:
        data: Data = from_smiles(smiles)

        # 如果有标签列，比如 kcat，可以加到 data.y
        if "kcat" in df.columns:
            data.y = torch.tensor([row["kcat"]], dtype=torch.float)

        graphs.append(data)

        # 打印第一个分子图的结构
        if i == 0:
            print("第一个分子图对象:")
            print(data)
            print("节点特征矩阵 x:", data.x.shape)
            print("边索引 edge_index:", data.edge_index.shape)
            print("边特征 edge_attr:", data.edge_attr.shape)

    except Exception as e:
        print(f"跳过非法 SMILES: {smiles}, 错误: {e}")

print(f"成功转换 {len(graphs)} 个分子为 PyG 图对象")

# Step 3: 保存到磁盘
torch.save(graphs, "graphs2.pt")
print("所有分子图已保存到 graphs2.pt")

# Step 4: 加载示例
loaded_graphs = torch.load("graphs2.pt")
print(f"加载成功，共 {len(loaded_graphs)} 个分子图")
print("示例:", loaded_graphs[0])

