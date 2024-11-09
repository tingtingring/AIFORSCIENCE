import pandas as pd

import pandas as pd

# 读取 CSV 文件
data = pd.read_csv('../dataset/test/data_task2.csv')

# 填充 'date' 列中的空值
data['date'].fillna('0000-00-00', inplace=True)

# 输出处理后的数据（可选）
data.to_csv('../dataset/test/data_task2.csv', index=False)

