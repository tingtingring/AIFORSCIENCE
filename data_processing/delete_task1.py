import pandas as pd
import numpy as np

df = pd.read_csv('../dataset/test/life_style.csv')

print(df.shape[1])
# 假设你已经有一个 DataFrame df
# 列名列表
columns_to_drop = ['f.100760.0.0', 'f.3731.0.0', 'f.3466.0.0', 'f.3456.0.0', 'f.3436.0.0', 'f.104400.0.0','f.100024.0.0','f.100025.0.0','f.100009.0.0','f.100017.0.0','f.100015.0.0','f.100011.0.0','f.100005.0.0']

# 删除列
df = df.drop(columns=columns_to_drop)

# 根据分布随机插值填充缺失值
def random_interpolate_fill(df):
    for column in df.columns:
        if df[column].isnull().any():
            # 获取不缺失值的样本
            valid_values = df[column].dropna()
            # 计算均值和标准差
            mean = valid_values.mean()
            std = valid_values.std()
            # 生成与缺失值数量相同的随机数
            num_missing = df[column].isnull().sum()
            random_values = np.random.normal(mean, std, num_missing)
            # 填充缺失值
            df.loc[df[column].isnull(), column] = random_values
    return df

# 填充缺失值
df = random_interpolate_fill(df)

# 将负值替换为0
df[df < 0] = 0
print(df.shape[1])

df.to_csv('../dataset/test/processed_life_style.csv', index=False)

# 如果你希望在原 DataFrame 上修改，可以使用 inplace=True
# df.drop(columns=columns_to_drop, inplace=True)
