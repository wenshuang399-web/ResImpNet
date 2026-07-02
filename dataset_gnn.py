import json
import gvp.data
from collections import defaultdict

def merge_chains(chains, pdb_id):
    """
    输入: 同一个 PDB 文件的多个链条信息 (list of dict)
    输出: 合并后的单个条目 (dict)，包含所有链的序列和坐标
    """
    merged_seq = "".join([c["seq"] for c in chains])
    merged_coords = []
    for c in chains:
        merged_coords.extend(c["coords"])

    return {
        "name": pdb_id,   # ✅ 用 PDB ID，保证唯一
        "seq": merged_seq,
        "coords": merged_coords,
        "chains": [c["name"] for c in chains]  # 记录来源链
    }

# 读取原始 JSON
with open("protein_structures2.json") as f:
    structures = json.load(f)

# 按 PDB 文件分组
merged_structures = defaultdict(list)
for s in structures:
    pdb_id = s["name"].split(".")[0]   # 去掉 .pdb 后缀
    merged_structures[pdb_id].append(s)

# 合并链
final_structures = []
for pdb_id, chains in merged_structures.items():
    merged_entry = merge_chains(chains, pdb_id)
    final_structures.append(merged_entry)

# 构造数据集
dataset = gvp.data.ProteinGraphDataset(final_structures)

print("蛋白质图数量:", len(dataset))  # 应该等于 PDB 文件数 (1936)
print("第一个图:", dataset[0])

