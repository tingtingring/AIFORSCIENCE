import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import matplotlib.font_manager as fm
from scipy.stats import fisher_exact

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 绘制条形图，针对浮点数列
def analyze_float_column1(middle_path):
    """
    读取合并后的数据集文件，分析浮点数列的缺失情况与T2D状态之间的关系，并绘制统计图。
    参数：
    middle_path (str): 'train' 或 'test'，用于读取相应的数据集。
    """
    # 1. 读取数据集
    data_df = pd.read_csv("./dataset/" + middle_path + "/merged_origin_data_task1.csv")

    # 2. 查找所有浮点数类型的列
    float_columns = data_df.select_dtypes(include=['float']).columns

    # 3. 创建一个新表格，统计每个个体的浮点数列是否缺失
    missing_data = data_df[float_columns].isna().astype(int)

    # 4. 将"T2D"列加到missing_data中，表示是否患病
    missing_data['T2D'] = data_df['T2D']

    # 5. 计算每个浮点数特征的缺失比例，按T2D分组
    missing_by_group = missing_data.groupby('T2D').mean()

    # 6. 可视化：绘制堆叠条形图，显示每个浮点数特征的缺失比例
    plt.figure(figsize=(12, 6))
    missing_by_group.T.plot(kind='bar', stacked=True, color=['blue', 'orange'])
    if middle_path == 'train':
        plt.title(f'按照是否患T2D分组的缺失值比例(训练集)')
    elif middle_path == 'test':
        plt.title(f'按照是否患T2D分组的缺失值比例(测试集)')
    plt.ylabel('缺失值比例')
    plt.xticks(rotation=45)
    plt.legend(['健康', '患病'], title='是否患T2D')
    plt.tight_layout()
    plt.show()

    # 7. 整合所有浮点数列的缺失情况，形成一个综合的缺失列
    missing_data['any_missing'] = missing_data[float_columns].any(axis=1).astype(int)

    # 8. 对综合的缺失列与T2D进行卡方检验
    contingency_table = pd.crosstab(missing_data['any_missing'], missing_data['T2D'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print("\n浮点数类型列值的缺失与是否患T2D的关系：")
    print(f"卡方统计量: {chi2}, p-value: {p}")
    if p < 0.05:
        print("缺失值与T2D显著相关。")
    else:
        print("缺失值与T2D不显著相关。")



    # # 7. 卡方检验：判断缺失值与T2D之间的关系
    # print("\n卡方检验结果：")
    # for col in float_columns:
    #     contingency_table = pd.crosstab(missing_data[col], missing_data['T2D'])
    #     chi2, p, dof, expected = chi2_contingency(contingency_table)
    #     print(f"特征: {col}, p-value: {p}")
    #     if p < 0.05:
    #         print(f"特征 {col} 的缺失值与T2D状态显著相关。")
    #     else:
    #         print(f"特征 {col} 的缺失值与T2D状态无显著相关。")



# 绘制矩阵，针对浮点数列
def analyze_float_column2(middle_path):
    # 读取数据
    data_df = pd.read_csv("./dataset/" + middle_path + "/" + "merged_origin_data_task1.csv")

    # 只选取数据类型为浮点数的列
    float_columns = data_df.select_dtypes(include=['float']).columns.tolist()


    # 初始化四个统计值
    T2D_and_missing = 0  # 患病且缺失
    T2D_and_not_missing = 0  # 患病且不缺失
    non_T2D_and_missing = 0  # 不患病且缺失
    non_T2D_and_not_missing = 0  # 不患病且不缺失

    # 遍历每一行，根据浮点数列是否缺失和T2D值进行统计
    for index, row in data_df.iterrows():
        T2D_status = row['T2D']  # 获取T2D状态
        float_cols_missing = row[float_columns].isna().sum() > 0  # 判断是否有浮点数列缺失

        if T2D_status == 1:  # 患T2D
            if float_cols_missing:
                T2D_and_missing += 1
            else:
                T2D_and_not_missing += 1
        elif T2D_status == 0:  # 不患T2D
            if float_cols_missing:
                non_T2D_and_missing += 1
            else:
                non_T2D_and_not_missing += 1

    # 构建2x2矩阵
    contingency_table = [[T2D_and_missing, T2D_and_not_missing],
                         [non_T2D_and_missing, non_T2D_and_not_missing]]

    # 使用Fisher精确检验
    _, p_value = fisher_exact(contingency_table)

    # 输出检验结果
    if middle_path == 'train':
        print("训练集Fisher精确检验结果:")
    elif middle_path == 'test':
        print("测试集Fisher精确检验结果:")
    print(f"p值: {p_value}")


    # 绘制可视化图表
    fig, ax = plt.subplots()
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)

    if middle_path == 'train':
        ax.set_title('是否患T2D与浮点数列是否缺失的Fisher精确检验矩阵(训练集)')
    elif middle_path == 'test':
        ax.set_title('是否患T2D与浮点数列是否缺失的Fisher精确检验矩阵(测试集)')
    ax.set_xlabel('是否缺失')
    ax.set_ylabel('是否患T2D')
    ax.set_xticklabels(['缺失', '不缺失'])
    ax.set_yticklabels(['患T2D', '不患T2D'])

    # plt.text(0.5, 1.5, f"p值: {p_value:.2e}", ha='center', va='center_baseline', fontsize=12)

    plt.tight_layout()
    plt.show()

    if p_value < 0.05:
        print("浮点数列是否缺失与是否患T2D显著相关。")
    else:
        print("浮点数列是否缺失与是否患T2D不显著相关。")


# 绘制矩阵，针对所选列
def analyze_selected_column(middle_path):
    # 读取数据
    data_df = pd.read_csv(f"./dataset/{middle_path}/merged_origin_data_task1.csv")

    # 选择特定的浮点数列
    selected_columns = ['f.100760.0.0', 'f.3436.0.0', 'f.104400.0.0']

    # 检查所选列是否在数据集中存在
    missing_columns = [col for col in selected_columns if col not in data_df.columns]

    if missing_columns:
        print(f"警告: 以下列未在数据中找到: {missing_columns}")
        return  # 如果缺少列，退出函数

    # 初始化四个统计值
    T2D_and_missing = 0  # 患病且缺失
    T2D_and_not_missing = 0  # 患病且不缺失
    non_T2D_and_missing = 0  # 不患病且缺失
    non_T2D_and_not_missing = 0  # 不患病且不缺失

    # 统计缺失值情况
    for index, row in data_df.iterrows():
        T2D_status = row['T2D']  # 获取T2D状态
        float_cols_missing = row[selected_columns].isna().any()  # 判断是否有选定列缺失

        if T2D_status == 1:  # 患T2D
            if float_cols_missing:  # 缺失
                T2D_and_missing += 1
            else:  # 不缺失
                T2D_and_not_missing += 1
        elif T2D_status == 0:  # 不患T2D
            if float_cols_missing:  # 缺失
                non_T2D_and_missing += 1
            else:  # 不缺失
                non_T2D_and_not_missing += 1

    # 构建2x2矩阵
    contingency_table = [
        [T2D_and_missing, T2D_and_not_missing],
        [non_T2D_and_missing, non_T2D_and_not_missing]
    ]

    # 使用Fisher精确检验
    _, p_value = fisher_exact(contingency_table)

    # 输出检验结果
    if middle_path == 'train':
        print("训练集Fisher精确检验结果:")
    elif middle_path == 'test':
        print("测试集Fisher精确检验结果:")
    print(f"p值: {p_value}")

    # 绘制可视化图表
    fig, ax = plt.subplots()
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    if middle_path == 'train':
        ax.set_title('是否患T2D与所选列是否缺失的Fisher精确检验矩阵(训练集)')
    elif middle_path == 'test':
        ax.set_title('是否患T2D与所选列是否缺失的Fisher精确检验矩阵(测试集)')
    ax.set_xlabel('是否缺失')
    ax.set_ylabel('是否患T2D')
    ax.set_xticklabels(['缺失', '不缺失'])
    ax.set_yticklabels(['患T2D', '不患T2D'])

    # 在图中显示p值
    # plt.text(0.5, 1.5, f"p值: {p_value:.2e}", ha='center', va='center', fontsize=12)
    plt.tight_layout()
    plt.show()

    if p_value < 0.05:
        print("所选列是否缺失与是否患T2D显著相关。")
    else:
        print("所选列是否缺失与是否患T2D不显著相关。")


if __name__ == '__main__':
    analyze_float_column1('train')
    analyze_float_column1('test')
    # analyze_float_column2("train")
    # analyze_float_column2("test")
    # analyze_selected_column("train")
    # analyze_selected_column("test")