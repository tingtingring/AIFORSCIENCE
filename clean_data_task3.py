import pandas as pd
import os

# 读取桌面上的 merged_data.csv 文件
file_path = '../dataset/train/merged_data_task1.csv'
data = pd.read_csv(file_path)

# 提取 f.eid 列的数据
f_eid_data = data['f.eid'].values

# ECG 数据目录路径
ecg_folder = '../dataset/train/ecg_data_without_norm' # 替换为实际 ECG 文件夹路径

# 创建一个空的列表，用于存放存在的行索引
valid_indices = []

# 遍历每个 f.eid
for eid in f_eid_data:
    # 构建 ECG 文件名
    ecg_file = os.path.join(ecg_folder, f"{eid}_20205_2_0.csv")

    # 检查文件是否存在
    if os.path.exists(ecg_file):
        valid_indices.append(data[data['f.eid'] == eid].index[0])  # 保存有效行的索引

# 根据有效索引筛选数据
filtered_data = data.loc[valid_indices]

# 保存筛选后的数据到新文件
filtered_data.to_csv('../dataset/train/data_task2_without_norm.csv', index=False)  # 修改保存路径

# 打印筛选后的数据
print(filtered_data)