import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer

from src.config import config

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]

# 解决负号显示问题
matplotlib.rcParams["axes.unicode_minus"] = False


def process_mnar(df, site_col, mnar_vars):
    """
    处理MNAR缺失：
    - 为每个变量添加缺失指示器（原始是否缺失）
    - 对指定中心的缺失值填充：连续变量填充-999，类别变量填充'missing'

    参数：
    df : DataFrame
    site_col : str, 中心标识列名
    mnar_vars : dict, 格式 {中心名称: [变量名列表]}
    """
    for center, var_list in mnar_vars.items():
        # 定位该中心的样本
        mask_center = df[site_col] == center
        for var in var_list:
            if var not in df.columns:
                print(f"警告：变量 {var} 不在数据中，跳过")
                continue
            # 创建缺失指示器（基于原始缺失）
            indicator_name = f"{var}_missing"
            df[indicator_name] = (
                df[var].isna().astype(int)
            )  #             cnt = df[indicator_name].value_counts()
            # 只对该中心的缺失样本进行填充
            missing_mask = mask_center & df[var].isna()
            if missing_mask.any():
                # 判断变量类型：数值型（int, float）-> 连续，否则类别

                if df[var].nunique() < 10:
                    df[var] = df[var].apply(lambda x: str(x) if pd.notna(x) else x)
                    # 添加"missing"类别到现有的类别中
                    existing_categories = df[var].dropna().unique()
                    if "missing" not in existing_categories:
                        categories_with_missing = list(existing_categories) + [
                            "missing"
                        ]
                    else:
                        categories_with_missing = existing_categories
                    df[var] = pd.Categorical(
                        df[var], categories=categories_with_missing
                    )
                    df.loc[missing_mask, var] = "missing"

                else:
                    df[var] = pd.to_numeric(df[var], errors="coerce")
                    df.loc[missing_mask, var] = -999

    return df


def process_mar(df, site_col, mar_vars, all_centers_key="all_centers"):
    """
    处理MAR缺失：使用多重插补（IterativeImputer）
    - 对不同中心分别进行插补（考虑到各中心变量集合可能不同）
    - all_centers中的变量会加入到每个中心的插补中

    参数：
    df : DataFrame (已处理完MNAR)
    site_col : str, 中心标识列名
    mar_vars : dict, 格式 {中心名称: [变量名列表]}，包含'all_centers'键
    all_centers_key : str, 表示所有中心共同MAR变量的键名
    """
    # 提取所有中心名称（mar_vars中的键，除去all_centers_key）
    centers = [c for c in mar_vars.keys() if c != all_centers_key]
    all_center_vars = mar_vars.get(all_centers_key, [])

    for center in centers:
        center_vars = mar_vars[center] + all_center_vars
        # 过滤出实际存在于数据中的变量
        exist_vars = [v for v in center_vars if v in df.columns]
        if not exist_vars:
            continue

        # 定位该中心的样本
        mask_center = df[site_col] == center
        if mask_center.sum() == 0:
            print(f"警告：中心 {center} 无样本，跳过")
            continue

        # 提取该中心子集，仅包含需要插补的变量以及其他可能用于预测的变量
        # 其他变量：除插补变量外的所有数值型变量（MNAR已填充，可作为预测因子）
        other_numeric_cols = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in exist_vars
        ]
        # 构建插补所用的列：插补变量 + 其他数值型变量
        impute_cols = exist_vars + other_numeric_cols
        sub_df = df.loc[mask_center, impute_cols].copy()

        # 确保插补变量列至少有一个非缺失值，否则无法插补
        valid_vars = []
        for var in exist_vars:
            if sub_df[var].notna().any():
                valid_vars.append(var)
            else:
                print(f"警告：变量 {var} 在中心 {center} 中全部缺失，跳过插补")
        if not valid_vars:
            continue

        # 执行多重插补（IterativeImputer）
        imputer = IterativeImputer(max_iter=10, random_state=42)
        # 注意：IterativeImputer要求输入为数值，非数值列需要提前处理
        # 此处假设所有用于插补的列都是数值型（MAR变量为连续变量，其他预测变量也是数值型）
        # 如果存在类别型预测变量，需先进行编码，本示例简化处理
        try:
            imputed_array = imputer.fit_transform(sub_df)
            imputed_df = pd.DataFrame(
                imputed_array, columns=sub_df.columns, index=sub_df.index
            )
            # 将插补后的值写回原df（仅替换MAR变量）
            for var in valid_vars:
                df.loc[mask_center, var] = imputed_df[var]
        except Exception as e:
            print(f"中心 {center} 多重插补失败：{e}")

    return df


def impute_main(df, mnar_jsonf, mar_jsonf, output_dir, output_file):
    mnar_vars = json.load(open(mnar_jsonf))
    mar_vars = json.load(open(mar_jsonf))
    # 缺失填补
    # 处理MNAR缺失
    df_processed = process_mnar(df, config.center, mnar_vars)
    cnt = df_processed["血_B型柯萨奇病毒抗体.IgM_missing"].value_counts()

    # 处理MAR缺失
    df_final = process_mar(df_processed, config.center, mar_vars)

    print("处理完成。")
    print(df_final.head())
    df_final.to_excel(f"{output_dir}/{output_file}", index=False)


# ----------------- 使用示例 -----------------
if __name__ == "__main__":
    # 模拟数据（包含各中心及变量）
    np.random.seed(42)
    n_samples = 100
    sites = ["控江医院"] * 30 + ["控江社区"] * 30 + ["延吉社区"] * 40
    df = pd.DataFrame(
        {
            "site": sites,
            "血B型柯萨奇病毒抗体.IgM": np.random.choice(
                [1, 2, np.nan], size=100, p=[0.4, 0.4, 0.2]
            ),
            "血肺炎支原体抗体.IgM": np.random.choice(
                ["阴性", "阳性", np.nan], size=100, p=[0.45, 0.45, 0.1]
            ),
            "血白细胞计数": np.random.normal(6, 1.5, 100),
            "血C反应蛋白": np.random.gamma(2, 2, 100),
            "血淀粉样蛋白A": np.random.gamma(3, 1.5, 100),
            "血血小板分布宽度": np.random.normal(12, 2, 100),
        }
    )
    # 人为引入缺失（模拟MAR）
    df.loc[df["site"] == "延吉社区", "血白细胞计数"] = np.nan
    df.loc[df["site"] == "控江医院", ["血C反应蛋白", "血淀粉样蛋白A"]] = np.nan
    df.loc[:, "血血小板分布宽度"] = np.where(
        np.random.rand(100) < 0.2, np.nan, df["血血小板分布宽度"]
    )

    # 定义MNAR和MAR字典
    mnar_vars = {
        "控江医院": ["血B型柯萨奇病毒抗体.IgM", "血肺炎支原体抗体.IgM"],
        "控江社区": ["血B型柯萨奇病毒抗体.IgM", "血肺炎支原体抗体.IgM"],
    }
    mar_vars = {
        "延吉社区": ["血白细胞计数"],
        "控江医院": ["血C反应蛋白", "血淀粉样蛋白A"],
        "all_centers": ["血血小板分布宽度"],
    }

    # 处理MNAR缺失
    df_processed = process_mnar(df, "site", mnar_vars)
    # 处理MAR缺失
    df_final = process_mar(df_processed, "site", mar_vars)

    print("处理完成。")
    print(df_final.head())
