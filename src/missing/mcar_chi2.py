import pandas as pd
from scipy.stats import chi2_contingency


# ==========================================
# 2. 核心功能：缺失指示变量检验函数
# ==========================================
def missing_chi2_test(df, target_col, features_to_test):
    """
    对指定特征进行缺失指示变量法检验(卡方检验)

    参数:
    df: pandas.DataFrame, 包含特征和因变量的数据集
    target_col: str, 因变量的列名 (例如 'Target')
    features_to_test: list, 需要检验缺失机制的特征列表
    target_labels: dict, 因变量标签映射，为了输出好看 (例如 {0: '上感', 1: '肺炎'})
    """
    results = []

    for feature in features_to_test:
        # 如果该特征根本没有缺失值，跳过
        if df[feature].isnull().sum() == 0:
            continue

        # 创建缺失指示变量：1表示缺失，0表示未缺失
        missing_indicator = df[feature].isnull().astype(int)

        # 构建交叉表 (Crosstab)
        contingency_table = pd.crosstab(missing_indicator, df[target_col])

        # 如果列数不足2列（比如某个类别的样本全缺失或全没缺失），无法做卡方检验
        if contingency_table.shape[0] < 2:
            p_value = "N/A"
            chi2 = "N/A"
            conclusion = "无法检验(分布单一)"
        else:
            # 进行卡方检验
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            chi2 = round(chi2, 4)
            p_value = round(p_value, 4)

            # 判断结论
            if p_value < 0.05:
                conclusion = "MAR/MNAR (与Target显著相关)"
            else:
                conclusion = "MCAR (与Target无显著相关)"

        # 统计每个Target类别下的缺失率 (为了方便写论文表格)
        row_data = {"变量名称": feature}
        row_data["卡方统计量"] = chi2
        row_data["P-value"] = p_value
        row_data["缺失机制结论"] = conclusion
        results.append(row_data)

    # 转换为DataFrame并返回，方便展示
    results_df = pd.DataFrame(results)
    return results_df


def multicenter_mcar_chi2_main(df, center_col, target_col, features, output_dir):
    centers = df[center_col].unique()
    for center in centers:
        df_center = df[df[center_col] == center]
        results_df = missing_chi2_test(df_center, target_col, features)
        results_df["中心"] = center
        results_df.to_excel(f"{output_dir}/mcar_chi2_test_{center}.xlsx", index=False)
