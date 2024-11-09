import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

# load_path='./data/CG2405/train/Baseline_characteristics.xlsx'
load_path='./data/CG2405/test/Baseline_characteristics.xlsx'

# 读取Excel数据
df = pd.read_excel(load_path)

X = df[['f.21001.0.0']]  # 第一列作为输入特征
Y = df[['f.4079.0.0', 'f.4080.0.0']]  # 第二列和第三列作为输出特征

# 找到第二列和第三列都没有缺失值的行（用于训练）
non_missing_rows = df[['f.4079.0.0', 'f.4080.0.0']].notnull().all(axis=1)

# 训练数据
X_train = X[non_missing_rows]  # 训练集中的输入特征
Y_train = Y[non_missing_rows]  # 训练集中的输出特征

# 构建随机森林的多输出回归模型（非线性回归）
model = MultiOutputRegressor(RandomForestRegressor())
model.fit(X_train, Y_train)

# 找到第二列和第三列同时缺失的行（需要预测的部分）
missing_rows = df[['f.4079.0.0', 'f.4080.0.0']].isnull().all(axis=1)
X_missing = X[missing_rows]

# 使用模型预测缺失的值
Y_pred = model.predict(X_missing)

# 将预测值四舍五入为整数
Y_pred_int = np.round(Y_pred).astype(int)

# 将四舍五入后的整数值填入原始数据
df.loc[missing_rows, ['f.4079.0.0', 'f.4080.0.0']] = Y_pred_int


# save_path='./data/CG2405/train/filled_Baseline_characteristics.csv'
save_path='./data/CG2405/test/filled_Baseline_characteristics.csv'


# 保存填补后的数据
df.to_csv(save_path, index=False)

print(f"缺失值填补完成，文件已保存到{save_path}")
