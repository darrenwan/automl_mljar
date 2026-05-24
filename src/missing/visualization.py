# visualization.py
import pandas as pd
import umap

from ..config import config

import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno


def sns_multicenter_missing_top(
        df, out_dir,
        center_col="CENTER",
        top_n=30,
        min_missing=0.05,
        suffix='raw'
):
    # 1️⃣ missing rate
    missing_df = (
        df
        .groupby(center_col)
        .apply(lambda x: x.isna().mean())
        .T
    )

    # 2️⃣ 计算跨中心差异（max - min）
    missing_df["diff"] = missing_df.max(axis=1) - missing_df.min(axis=1)

    # 3️⃣ 过滤低缺失变量
    missing_df = missing_df[missing_df.max(axis=1) > min_missing]

    # 4️⃣ 排序（差异最大）
    missing_df = missing_df.sort_values("diff", ascending=False)

    # 5️⃣ 取 Top N
    missing_rates = missing_df.drop(columns=["diff"]).head(top_n)

    # 6️⃣ 画图

    plt.figure(figsize=(12, 10))
    # 直接使用前面算好的宽表 missing_rates
    sns.heatmap(
        missing_rates,
        annot=True,  # 显示数值
        fmt=".2f",  # 保留一位小数
        cmap="YlOrRd",  # 颜色越红代表缺失率越高
        cbar_kws={'label': 'Missing Rate (%)'}
    )
    plt.title(f"Top {top_n} Center-wise Missing Differences ({suffix})", pad=15, fontdict={'fontsize': 16})
    plt.xticks(rotation=45)

    plt.xlabel(center_col)
    plt.ylabel("Features")

    plt.tight_layout()
    plt.savefig(
        f"{out_dir}/sns_multicenter_missing_top_{suffix}.svg", dpi=300
    )
    plt.show()


def sns_multicenter_missing_cluster(df, out_dir, center_col, suffix='raw'):
    missing_rates = (
        df.groupby(center_col)
        .apply(lambda x: x.isna().mean())
        .T
    )
    cols_ncount = missing_rates.shape[1]
    rows_count = missing_rates.shape[0]
    plt.figure(figsize=(12, 10))
    sns.clustermap(
        missing_rates,
        cmap="YlOrRd",
        annot=False
    )

    plt.suptitle(f"Clustered Missing Pattern Across {center_col}")

    plt.tight_layout()
    plt.savefig(
        f"{out_dir}/sns_multicenter_missing_cluster_{suffix}.svg", dpi=300
    )
    plt.show()


def sns_feat_miss_overview(df, out_dir, suffix='raw'):
    # 计算每个医院每个特征的缺失率
    missing_rates = df.set_index(config.center).isna().groupby(config.center).mean() * 100
    cols_ncount = missing_rates.shape[1]
    rows_count = missing_rates.shape[0]
    plt.figure(figsize=(cols_ncount * 0.8, rows_count * 2))
    # 直接使用前面算好的宽表 missing_rates
    sns.heatmap(
        missing_rates,
        annot=True,  # 显示数值
        fmt=".2f",  # 保留一位小数
        cmap="YlOrRd",  # 颜色越红代表缺失率越高
        cbar_kws={'label': 'Missing Rate (%)'}
    )
    plt.title(f'Missing Data Heatmap by {config.center} ({suffix})', pad=15, fontdict={'fontsize': 16})
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        f"{out_dir}/sns_feat_miss_overview_{suffix}.svg", dpi=300
    )
    plt.show()


def sns_feat_miss_overview_row_cols(df, out_dir, suffix='raw'):
    # ==========================================
    # 2. 计算特征列和样本行的缺失率
    # ==========================================
    # 特征列缺失率 (按列计算均值，转换为百分比，并降序排列)
    col_missing_rate = (df.isna().mean(axis=0) * 100).sort_values(ascending=False)

    # 样本行缺失率 (按行计算均值，转换为百分比)
    row_missing_rate = df.isna().mean(axis=1) * 100

    # ==========================================
    # 3. 使用 Seaborn 进行左右组合绘图
    # ==========================================
    # sns.set_theme() 会重置 Matplotlib 的全局设置。
    # sns.set_theme(style="whitegrid", font='SimHei')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1行2列的画布

    # ---------- 图1：特征列缺失率 (柱状图) ----------
    ax1 = sns.barplot(
        x=col_missing_rate.values,
        y=col_missing_rate.index,
        ax=axes[0],
        palette="flare"  # 使用渐变色系
    )
    axes[0].set_title(f'Missing Rate by Features (Columns) ({suffix})', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Missing Rate (%)', fontsize=12)
    axes[0].set_ylabel('Features', fontsize=12)
    # 修改点：延长一点 X 轴的范围，防止右侧的百分比标签被切掉
    axes[0].set_xlim(0, col_missing_rate.max() * 1.15)

    # 在柱子上添加具体数值标签
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10)

    # ---------- 图2：样本行缺失率分布 (直方图) ----------
    sns.histplot(
        row_missing_rate,
        bins=15,  # 分成15个区间
        kde=True,  # 显示核密度估计曲线
        color="teal",
        ax=axes[1]
    )
    axes[1].set_title(f'Distribution of Missing Rates Across Samples (Rows) ({suffix})', fontsize=14, fontweight='bold',
                      pad=10)
    axes[1].set_ylabel('Number of Samples', fontsize=12)
    axes[1].set_xlabel('Missing Rate per Sample (%)', fontsize=12)

    # 调整整体布局，防止标签被截断
    plt.tight_layout()
    plt.savefig(
        f"{out_dir}/sns_feat_miss_overview_row_cols_{suffix}.svg", dpi=300
    )
    plt.show()


def plot_missing_patterns_msno(df, center_dep_vars, non_center_dep_vars, output_dir):
    CENTER_COL = config.center
    fontsize = 10

    def plot(df, fig_type, output_dir):
        if fig_type == "matrix":
            msno.matrix(
                df,
                sparkline=False,
                fontsize=fontsize,
            )
        elif fig_type == "heatmap":
            msno.heatmap(
                df,
                fontsize=fontsize,
            )
        elif fig_type == "dendrogram":
            msno.dendrogram(
                df,
                fontsize=fontsize,
            )

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/msno_{fig_type}.svg",
            dpi=300
        )

        plt.show()

    def plot_by_center(df, fig_type, output_dir):
        centers = sorted(df[CENTER_COL].unique())

        n_center = len(centers)

        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_center,
            figsize=(6 * n_center, 8)
        )

        # 如果只有1个center
        if n_center == 1:
            axes = [axes]

        for i, center in enumerate(centers):
            df_center = df[
                df[CENTER_COL] == center
                ].drop(columns=[CENTER_COL])

            if fig_type == "matrix":
                msno.matrix(
                    df_center,
                    ax=axes[i],
                    sparkline=False,
                    fontsize=fontsize,
                )
            elif fig_type == "heatmap":
                msno.heatmap(
                    df_center,
                    ax=axes[i],
                    fontsize=fontsize,
                )
            elif fig_type == "dendrogram":
                msno.dendrogram(
                    df_center,
                    ax=axes[i],
                    fontsize=fontsize,
                )

            axes[i].set_title(
                f"Center {center}\n"
                f"N={len(df_center)}"
            )

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/msno_by_center_{fig_type}.svg",
            dpi=300
        )

        plt.show()

    if center_dep_vars:
        df2 = df[center_dep_vars + [CENTER_COL]]
        plot_by_center(df2, "matrix", output_dir)
        if len(center_dep_vars) > 1:
            plot_by_center(df2, "heatmap", output_dir)
            plot_by_center(df2, "dendrogram", output_dir)
    if non_center_dep_vars:
        df3 = df[non_center_dep_vars]
        plot(df3, "matrix", output_dir)
        if len(non_center_dep_vars) > 1:
            plot(df3, "heatmap", output_dir)
            plot(df3, "dendrogram", output_dir)


def plot_missing_corr(df, features, out_dir):
    missing_matrix = (
        df[features]
        .isna()
        .astype(int)
    )

    corr = missing_matrix.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr)
    plt.title("Missing Correlation")
    plt.tight_layout()

    plt.savefig(
        f"{out_dir}/missing_corr.png"
    )


def plot_missing_umap(df, features, output_dir):
    missing_matrix = (
        df[features]
        .isna()
        .astype(int)
    )

    embedding = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        random_state=42
    ).fit_transform(missing_matrix)

    # 将类别变量转换为数值编码用于颜色映射
    categories = df[config.center]
    unique_categories = categories.unique()
    category_to_num = {cat: i for i, cat in enumerate(unique_categories)}
    color_vals = [category_to_num[cat] for cat in categories]

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=color_vals,
        cmap='tab10',  # 使用适合分类的颜色映射
        alpha=0.7
    )

    # 添加颜色条并设置标签
    plt.colorbar(ticks=range(len(unique_categories)), label=config.center)
    plt.title("Missingness UMAP")
    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/missing_umap.svg", dpi=300
    )


def feat_distribution(df, output_dir, suffix="raw"):
    if config.pid in df.columns:
        data = df.drop(columns=config.pid)
    else:
        data = df

    # 转换数据格式
    data_long = data.melt(
        id_vars=[config.center],
        var_name='feature',
        value_name='value'
    )

    # 总体分布直方图
    g = sns.FacetGrid(
        data_long,
        col='feature',
        # hue=config.center,
        col_wrap=5,
        sharex=False,  # FacetGrid 直接支持 sharex
        sharey=False,
        height=4,
        aspect=1.2
    )
    g.map(sns.histplot, 'value', stat='count')
    g.add_legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feat_distribution_{suffix}.svg", dpi=300)
    plt.show()

    # 基于多中心的分布直方图
    g = sns.FacetGrid(
        data_long,
        col='feature',
        hue=config.center,
        col_wrap=5,
        sharex=False,  # FacetGrid 直接支持 sharex
        sharey=False,
        height=4,
        aspect=1.2
    )
    g.map(sns.histplot, 'value', stat='count')
    g.add_legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feat_distribution_by_center_{suffix}.svg", dpi=300)
    plt.show()
