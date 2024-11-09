import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 创建示例 DataFrame

df = pd.read_csv("../dataset/train/processed_life_style.csv")

df_without_first_column = df.iloc[:, 1:]  # 选择除了第一列的所有列

# 计算最大值和最小值
max_values = df_without_first_column.max()
min_values = df_without_first_column.min()


print("最大值：")
print(max_values)

print("\n最小值：")
print(min_values)