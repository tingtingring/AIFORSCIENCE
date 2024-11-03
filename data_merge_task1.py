import pandas as pd

# 定义文件路径
base_file_path = '../dataset/test/test_filled_baseline_characteristics.csv'
label_file_path = '../dataset/test/ground_true.csv'
life_style_file_path = '../dataset/test/processed_life_style.csv'
output_file_path = '../dataset/test/merged_data_task1.csv'

try:
    # 读取CSV文件
    base_df = pd.read_csv(base_file_path)
    label_df = pd.read_csv(label_file_path)
    life_style_df = pd.read_csv(life_style_file_path)
    print(base_df.head)
    print(label_df.head)
    print(life_style_df.head)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)
except pd.errors.EmptyDataError:
    print("Error: One of the CSV files is empty.")
    exit(1)
except pd.errors.ParserError:
    print("Error: There was a problem parsing one of the CSV files.")
    exit(1)

common_key = 'f.eid'

# 合并数据
try:
    merged_df = pd.merge(label_df, base_df, on=common_key)
    final_df = pd.merge(merged_df, life_style_df, on=common_key)
except KeyError as e:
    print(f"Error: Common key '{common_key}' not found in one of the dataframes: {e}")
    exit(1)

# # 提取前2400个和后800个样本
# first_2400 = final_df.iloc[:2400]
# last_800 = final_df.iloc[-800:]
#
# # 打乱前2400个样本并取出800个
# shuffled_first_2400 = first_2400.sample(frac=1).reset_index(drop=True)  # 打乱
# selected_from_first = shuffled_first_2400.iloc[:800]
# print(selected_from_first.head())
# # 将打乱后的800个与后800个合并
# combined = pd.concat([selected_from_first, last_800]).sample(frac=1).reset_index(drop=True)  # 最后打乱

# # 显示合并后的数据
# print("\nFinal Combined Data:")
# print(combined.head())

# 保存合并后的数据到新的CSV文件（可选）
# combined.to_csv(output_file_path, index=False)
final_df.to_csv(output_file_path, index=False)