# run_pipeline.py

import os

from matplotlib import pyplot as plt

from src.config import config
from src.missing.center_miss import run_multicenter_missing_analysis
from src.missing.data_loader import load_data, row_col_rm
from src.missing.mcar_chi2 import multicenter_mcar_chi2_main
from src.missing.missing_analysis import (
    compute_missing_summary,
    evidently_wasserstein_psi,
    norm_analysis,
    run_mcar_test,
    tableone_analysis,
)
from src.missing.visualization import (
    plot_missing_patterns_msno,
    plot_missing_umap,
    sns_feat_miss_overview,
    sns_feat_miss_overview_row_cols,
    sns_multicenter_missing_cluster,
    sns_multicenter_missing_top,
)

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]

# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False


def main():
    """
    MCAR + independence OK
    ⇒ MCAR

    Observed predictors exist
    ⇒ MAR

    Strong center effect + unexplained
    ⇒ MNAR suspected
    90%缺失变量 很可能 MNAR
    :return:
    """
    out_dir = config.DATA_MID_DIR + "/missing_analysis"
    os.makedirs(out_dir, exist_ok=True)
    out_distri_dir = config.DATA_MID_DIR + "/distribution_analysis"
    os.makedirs(out_distri_dir, exist_ok=True)
    print("Loading data start...")
    df = load_data(config.DATA_WIDE_FILE)

    # sns_multicenter_missing_top(df, out_dir, center_col=config.center, top_n=10, min_missing=0.05, suffix='raw')
    # sns_multicenter_missing_cluster(df, out_dir, center_col=config.center, suffix='raw')

    # sns_feat_miss_overview(df, out_dir, suffix='raw')
    # sns_feat_miss_overview_row_cols(df, out_dir, suffix='raw')

    df, cat_cols, num_cols = row_col_rm(df, out_dir)
    df.to_excel(config.DATA_WIDE_FILE.replace(".xlsx", "_clean.xlsx"), index=False)
    df_xmiss = df[df.isnull().sum(axis=1) == 0.0]
    df_xmiss.to_excel(
        config.DATA_WIDE_FILE.replace(".xlsx", "_clean_no_miss.xlsx"), index=False
    )

    sns_multicenter_missing_top(
        df,
        out_dir,
        center_col=config.center,
        top_n=10,
        min_missing=0.05,
        suffix="clean",
    )
    sns_multicenter_missing_cluster(
        df, out_dir, center_col=config.center, suffix="clean"
    )
    sns_feat_miss_overview(df, out_dir, suffix="clean")
    sns_feat_miss_overview_row_cols(df, out_dir, suffix="clean")
    #
    # print("Missing summary start...")
    miss_small, miss_big = compute_missing_summary(
        df, features=cat_cols + num_cols, output_dir=out_dir
    )
    print("miss_big...", " ".join(miss_big))
    #
    if config.pid in df.columns:
        df.drop(columns=config.pid, inplace=True)

    # center-independent missingness stat
    center_dep_vars, non_center_dep_vars = run_multicenter_missing_analysis(
        df, miss_big, output_dir=out_dir
    )
    print("center_dep_vars: ", center_dep_vars)
    print("non_center_dep_vars: ", non_center_dep_vars)

    # missing patterns：MNAR/MAR/MCAR
    # MNAR：缺失值与目标变量相关
    # MAR：缺失值与目标变量无关，但与其他变量相关
    # MCAR：缺失值与目标变量无关，且与其他变量无关
    multicenter_mcar_chi2_main(
        df,
        center_col=config.center,
        target_col=config.label,
        features=miss_big,
        output_dir=out_dir,
    )
    category_rate = len(cat_cols) / (len(num_cols) + len(cat_cols))
    if category_rate < 0.4:
        mcar = run_mcar_test(
            df, output_dir=out_dir
        )  # 类别变量>40%不适合，建议使用chi2检验
        print("mcar: ", mcar)
    plot_missing_patterns_msno(
        df, center_dep_vars, non_center_dep_vars, output_dir=out_dir
    )

    plot_missing_umap(df, features=cat_cols + num_cols, output_dir=out_dir)

    # 多中心分布分析：SMD analysis
    norm_features = norm_analysis(df, num_cols, output_dir=out_distri_dir)
    tableone_analysis(df, cat_cols, num_cols, norm_features, output_dir=out_distri_dir)
    evidently_wasserstein_psi(
        df, config.center, cat_cols, num_cols, output_dir=out_distri_dir
    )


if __name__ == "__main__":
    main()
