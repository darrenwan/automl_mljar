# data_loader.py

import pandas as pd

from ..config import config


def remove_invalid_rows(df, out_dir):
    # 每行的缺失率
    missing_per_row = df.isnull().sum(axis=1)
    missing_per_row_pct = (df.isnull().sum(axis=1) / df.shape[1]) * 100
    row_missing_summary = pd.DataFrame(
        {
            "Row Index": df[config.pid],
            "Missing Features Count": missing_per_row,
            "Missing Features Percentage": missing_per_row_pct,
        }
    ).sort_values("Missing Features Percentage", ascending=False)

    row_missing_summary.to_excel(f"{out_dir}/row_missing_summary.xlsx", index=False)

    df["missing_per_row_pct"] = missing_per_row_pct
    df = df[df["missing_per_row_pct"] < 50]  # 删除缺失率大于50%的样本
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=["missing_per_row_pct"], inplace=True)

    # 删除重复样本
    print("\nChecking for duplicate samples...\n")
    df.sort_values(
        by=[config.center, config.pid, config.age],
        ascending=[True, True, False],
        inplace=True,
    )
    df_dup = df[df.duplicated([config.pid], keep=False)]
    print(f"Total duplicate samples: {df_dup.shape[0]}")
    print(df_dup[config.pid].unique())
    df_dup.to_excel(f"{out_dir}/duplicate_samples.xlsx", index=False)
    df = df.drop_duplicates(subset=[config.pid], keep="first")

    return df


def remove_invalid_features(df, output_dir, missing_rate_threshold=0.5):
    """
    删除不符合条件的特征：
    1. 在任意一个中心中，特征值全为缺失值
    2. 在任意一个中心中，特征值只有唯一值（方差为0）
    """
    print("\nChecking features center-wise...\n")

    bad_features = []

    report_rows = []

    centers = df[config.center].unique()
    cols = df.columns.drop([config.center, config.pid, config.label])

    for c in centers:
        df_c = df[df[config.center] == c]

        for col in cols:
            sub = df_c[col]

            # -------- 单值 --------
            nunique = sub.dropna().nunique()
            single_value = nunique <= 1

            missing_rate = sub.isna().mean()

            report_rows.append(
                {
                    "feature": col,
                    "center": c,
                    "missing_rate": missing_rate,
                    "n_unique": nunique,
                    "single_value": single_value,
                }
            )

            if missing_rate >= missing_rate_threshold or (
                single_value and missing_rate == 0.0
            ):
                # 缺失率大于70%或单值且缺失率为0的特征
                bad_features.append(col)

    report_df = pd.DataFrame(report_rows)
    report_df.to_excel(
        f"{output_dir}/features_isna_report_{missing_rate_threshold}.xlsx", index=False
    )

    print("Total bad features:", len(bad_features))
    print(bad_features)

    df_clean = df.drop(columns=bad_features)

    return df_clean


def load_data(mid_file):
    # 1. 读取数据
    df = pd.read_excel(mid_file)
    print("--- 原始数据信息 ---")
    print(df.info())

    return df


def zh2en():
    df = pd.read_excel(config.DATA_ZH_EN_FILE)
    zh2en_map = dict(zip(df["中文"], df["英文"]))
    return zh2en_map


def row_col_rm(df, out_dir):
    """
    删除缺失率大于50%的样本和特征
    """
    # 1. 读取数据
    df = remove_invalid_features(df, out_dir, missing_rate_threshold=1.0)
    df = remove_invalid_rows(df, out_dir)
    df = remove_invalid_features(df, out_dir, missing_rate_threshold=0.7)

    if config.DATA_ZH_EN_FILE:
        zh2en_map = zh2en()
        cols = [zh2en_map.get(i, i) for i in df.columns]
        df.columns = cols

    cat_cols = []
    exclude_cols = [config.pid]  # 排除医院ID列、pid

    for col in df.columns:
        if col in exclude_cols:
            continue  # 跳过医院ID列、pid

        # 如果独立值数量<=10 且不是数值型中的连续大数，则认定为分类变量

        if df[col].nunique() < 10:
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else x)
            # df[col] = df[col].astype("category")
            cat_cols.append(col)

        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\n--- 识别到的分类特征 ---")
    print(cat_cols)

    # 自动推断连续变量名 (排除目标、医院ID和分类变量)
    numerical_features = [
        col for col in df.columns if col not in exclude_cols + cat_cols
    ]

    return df, cat_cols, numerical_features
