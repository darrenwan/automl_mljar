# missing_analysis.py
import json
from collections import defaultdict

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.config import config


def compute_missing_summary(df, features, output_dir):
    # 每行的缺失率
    missing_per_row = df.isnull().sum(axis=1)
    missing_per_row_pct = (df.isnull().sum(axis=1) / (df.shape[1] - 3)) * 100
    row_missing_summary = pd.DataFrame({
        'Row Index': df[config.pid],
        'Missing Features Count': missing_per_row,
        'Missing Features Percentage': missing_per_row_pct
    }).sort_values('Missing Features Percentage', ascending=False)

    results = {}

    # Overall missing
    overall = df[features].isna().mean()

    results["overall"] = overall

    # By center
    by_center = (
        df.groupby(config.center)[features]
        .apply(lambda x: x.isna().mean())
    )

    results["by_center"] = by_center

    # Center × label
    cross = (
        df.groupby(
            [config.center, config.label]
        )[features]
        .apply(lambda x: x.isna().mean())
    )

    miss_big = [col for col in cross.columns if cross[col].max() > 0.05]
    miss_small = [col for col in cross.columns if cross[col].max() <= 0.05 and cross[col].max() > 0.0]

    results["by_center_label"] = cross
    results["miss_big_small"] = pd.DataFrame({"miss_big": [miss_big], "miss_small": [miss_small]})

    label_counts = df.groupby(config.center)[config.label].value_counts()

    with pd.ExcelWriter(f"{output_dir}/missing_summary.xlsx") as writer:
        row_missing_summary.to_excel(writer, sheet_name="row_missing_summary", index=False)
        results["overall"].to_excel(writer, sheet_name="overall")
        results["by_center"].to_excel(writer, sheet_name="by_center")
        results["by_center_label"].to_excel(writer, sheet_name="by_center_label")
        results["miss_big_small"].to_excel(writer, sheet_name="miss_big_small_by_center_label")
        label_counts.to_excel(writer, sheet_name="label_counts")

    return miss_small, miss_big


# 完整的工作流
def norm_analysis(df, num_cols, output_dir):
    """
    完整的正态性分析流程
    样本量建议：

    n < 50：使用Shapiro-Wilk

    n > 50：使用K-S检验或Anderson检验

    快速决策树
        需要最可靠结果 → Shapiro-Wilk（n<5000）

        怀疑重尾或异常值 → Anderson-Darling

        想了解偏离原因（偏态/峰态） → D'Agostino-Pearson

        避免使用 → KS检验（除非不估计参数且分布已知）
        最佳实践：以Shapiro-Wilk为主，Anderson-Darling为辅，报告两个结果。    """
    from scipy import stats

    results = []
    norm_features = []

    for col in num_cols:
        df2 = df[col]
        df2.dropna(inplace=True)

        # 1. Shapiro-Wilk检验
        shapiro_stat, shapiro_p = stats.shapiro(df2)

        # 2. D'Agostino-Pearson检验
        dp_stat, dp_p = stats.normaltest(df2)
        # 3. Anderson-Darling检验（更稳健）
        anderson_stat, anderson_p = stats.anderson(df2, dist='norm', method='interpolate')

        # 4. KS检验（用样本均值和标准差作为理论参数）
        ks_stat, ks_p = stats.kstest(df2, 'norm', args=(df2.mean(), df2.std()))

        # 判断是否符合正态分布（以P>0.05为标准）
        shapiro_normal = '是' if shapiro_p > 0.05 else '否'
        dp_normal = '是' if dp_p > 0.05 else '否'
        ks_normal = '是' if ks_p > 0.05 else '否'
        anderson_normal = '是' if anderson_p > 0.05 else '否'

        norm_cnt = 0
        if shapiro_normal == '是':
            norm_cnt += 1
        # if dp_normal == '是':
        #     norm_cnt += 1
        # if ks_normal == '是':
        #     norm_cnt += 1
        if anderson_normal == '是':
            norm_cnt += 1

        if norm_cnt >= 1:
            norm_features.append(col)

        results.append({
            'sample_size': df2.shape[0],
            '变量': col,
            'Shapiro统计量': round(shapiro_stat, 4),
            'Shapiro_P值': round(shapiro_p, 4),
            'Shapiro_符合正态': shapiro_normal,
            'DP统计量': round(dp_stat, 4),
            'DP_P值': round(dp_p, 4),
            'DP_符合正态': dp_normal,
            'KS统计量': round(ks_stat, 4),
            'KS_P值': round(ks_p, 4),
            'KS_符合正态': ks_normal,
            'Anderson统计量': anderson_stat,
            'Anderson_P值': round(anderson_p, 4),
            'Anderson_符合正态': anderson_normal,
            '满足正态检验的个数-shapiro-anderson': norm_cnt
        })

    res = pd.DataFrame(results)
    res.to_excel(f"{output_dir}/normal_test.xlsx", index=False)
    return norm_features


def tableone_analysis(df, cat_cols, num_cols, norm_features, output_dir):
    from tableone import TableOne
    nonnormal = list(set(num_cols) - set(norm_features))
    rename = {'death': 'mortality'}

    mytable = TableOne(
        df, columns=cat_cols + num_cols, categorical=cat_cols, nonnormal=nonnormal,
        continuous=num_cols, groupby=config.center, missing=True, pval=True, smd=True,
        # pval_adjust='bonferroni'
    )
    print(mytable.tabulate(tablefmt="fancy_grid"))
    mytable.to_excel(f"{output_dir}/tableone.xlsx")


def run_mcar_test(df, output_dir):
    """
    p > 0.05 ⇒ MCAR plausible
    p < 0.05 ⇒ 非 MCAR
    :param df:
    :return:
    """

    from missdat import mcar_test

    df_test = df.copy()
    if config.pid in df_test.columns:
        df_test.drop(columns=config.pid, inplace=True)
    # if config.label in df_test.columns:
    #     df_test.drop(columns=config.label, inplace=True)
    # 处理分类数据类型，将其转换为适合MCAR测试的格式
    # df_test = pd.get_dummies(df_test, columns=config.non_order_cols, drop_first=True,
    #                          dtype=int)  # drop_first=False 保留全部k类
    print(df_test.info())
    # print(df_test.columns)
    mcar_dict = {}
    for hos in df_test[config.center].unique():
        df_hos = df_test.loc[df_test[config.center] == hos]
        df_hos.drop(columns=config.center, inplace=True)

        for col in df_hos.columns:
            print(f'col: {col}')
            if df_hos[col].nunique() < 10:
                # 对于对象类型，尝试转换为数值类型，否则用标签编码
                try:
                    df_hos[col] = df_hos[col].apply(lambda x: int(float(x)) if pd.notna(x) else x)

                    # df_test[col] = pd.to_numeric(df_test[col])

                except ValueError:
                    # 使用标签编码
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    mask = df_hos[col].notna()
                    df_hos.loc[mask, col] = le.fit_transform(df_hos.loc[mask, col])
                    df_hos[col] = df_hos[col].apply(lambda x: int(x) if pd.notna(x) else x)

        print('mcar_test info: ', df_hos.info())
        result = mcar_test(df_hos)
        p = result.loc['p', 'MCAR Test Values']
        print(f'mcar_test p: {p}')
        if p > 0.05:
            mcar_dict[hos] = True
        else:
            mcar_dict[hos] = False
        result.to_excel(f"{output_dir}/mcar_test_{hos}.xlsx", index=True)
    return mcar_dict


def evidently_wasserstein_psi(df, col, cat_cols, num_cols, output_dir):
    # 获取数据中所有的分组类别（例如：'对照组', '实验组'）

    # 2. 统计该分组下各医院的样本量
    hos_cnt = df[col].value_counts()
    cols = [col] + cat_cols + num_cols
    cols = list(set(cols))
    df = df[cols]
    if config.center in cat_cols:
        cat_cols.remove(config.center)

    if len(hos_cnt) >= 2:

        # 3. 选定该分组下样本最多的医院作为基线
        hos_cnt_max = hos_cnt.index[0]
        reference_data = df[df[col] == hos_cnt_max]

        # 获取需要测试的其他医院
        current_hospitals = hos_cnt.index.tolist()[1:]

        # 4. 循环对比该分组下的其他医院
        for hosp in current_hospitals:
            print(f"\n正在生成报告: 基线医院({hos_cnt_max}) vs 测试医院({hosp})...")

            current_data = df[df[col] == hosp]

            local_schema = DataDefinition(
                numerical_columns=num_cols,
                categorical_columns=cat_cols,
            )

            # 构建 Evidently Dataset
            eval_data_1 = Dataset.from_pandas(reference_data, data_definition=local_schema)
            eval_data_2 = Dataset.from_pandas(current_data, data_definition=local_schema)

            # 运行报告

            report = Report(metrics=[
                DataDriftPreset(cat_method="psi", num_method="wasserstein", cat_threshold=0.2, num_threshold=0.1)])

            my_eval = report.run(reference_data=eval_data_1, current_data=eval_data_2)
            # print(my_eval.json())

            # 导出为 HTML 可视化报告
            html_file = f"{output_dir}/DataDriftPreset_{hos_cnt_max}_vs_{hosp}.html"
            my_eval.save_html(html_file)
            print(f"✅ 报告已成功保存至: {html_file}")
