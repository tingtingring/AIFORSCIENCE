import pandas as pd
from sklearn.ensemble import RandomForestRegressor

load_path='./data/CG2405/train/filled_Baseline_characteristics.xlsx'

# 读取Excel数据
df = pd.read_excel(load_path)

# 使用第二列和第三列作为输入特征
X = df[['f.4079.0.0', 'f.4080.0.0']]  # 第二列和第三列作为输入
Y = df['f.21001.0.0']  # 第一列作为输出（需要补全的列）

# 找到第一列没有缺失值的行（用于训练）
non_missing_rows = df['f.21001.0.0'].notnull()

# 训练数据
X_train = X[non_missing_rows]  # 训练集中的输入特征
Y_train = Y[non_missing_rows]  # 训练集中的输出特征

# 构建随机森林回归模型（非线性回归）
model = RandomForestRegressor()
model.fit(X_train, Y_train)

# 找到第一列有缺失值的行（需要预测的部分）
missing_rows = df['f.21001.0.0'].isnull()
X_missing = X[missing_rows]

# 使用模型预测缺失的第一列的值
Y_pred = model.predict(X_missing)

df.loc[missing_rows, 'f.21001.0.0'] = Y_pred

save_path='./data/CG2405/train/train_filled_Baseline_characteristics.csv'

# 保存填补后的数据
df.to_csv(save_path, index=False)

print(f"缺失值填补完成，文件已保存到{save_path}")