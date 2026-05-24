import json
import os

import pandas as pd
from matplotlib import pyplot as plt

from src.config import config
from src.missing.impute_mice import median_mode_impute, mice_impute
from src.missing.missing_analysis import evidently_wasserstein_psi
from src.missing.visualization import feat_distribution

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False


class DataImputer:
    """
    数据插补器类，用于处理缺失值插补任务
    """

    def __init__(
        self,
        input_file: str,
        out_distri_dir="/distribution_analysis",
        output_file="imputed.xlsx",
    ):
        self.input_file = input_file
        self.output_dir = config.DATA_FINAL_DIR
        self.out_distri_dir = config.DATA_MID_DIR + out_distri_dir
        self.output_file = output_file
        self.id_cols = [config.pid, config.center, config.label]
        self.model_miss_encoder_dir = config.MODEL_MISS_ENCODER_DIR
        self.model_mice_dir = config.MODEL_MICE_DIR

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.out_distri_dir, exist_ok=True)

        # 初始化数据
        self.cat_cols, self.cat_str_cols, self.num_cols, self.df = (
            self._load_and_classify_data()
        )

        # 分析缺失值
        (
            self.miss_small_cat,
            self.miss_small_num,
            self.miss_big_cat,
            self.miss_big_num,
        ) = self._analyze_missing_values()

    def _load_and_classify_data(
        self,
    ) -> tuple[list[str], list[str], list[str], pd.DataFrame]:
        """
        加载数据并按类型分类列
        """
        df = pd.read_excel(self.input_file)

        cat_cols = []
        num_cols = []
        cat_str_cols = []

        for col in df.columns:
            if col in self.id_cols:
                continue
            if df[col].nunique() < 10:
                cat_cols.append(col)
                if df[col].dtype == "object":
                    cat_str_cols.append(col)
            else:
                num_cols.append(col)

        return cat_cols, cat_str_cols, num_cols, df

    def _analyze_missing_values(
        self,
    ) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
        """
        分析缺失值分布
        """
        # Center × label
        cross = self.df.groupby([config.center, config.label])[
            self.cat_cols + self.num_cols
        ].apply(lambda x: x.isna().mean())
        print(cross.index)

        miss_big = [col for col in cross.columns if cross[col].max() > 0.05]
        miss_small = [
            col
            for col in cross.columns
            if cross[col].max() <= 0.05 and cross[col].max() > 0.0
        ]
        miss_small_cat = [col for col in miss_small if col in self.cat_cols]
        miss_small_num = [col for col in miss_small if col in self.num_cols]
        miss_big_cat = [col for col in miss_big if col in self.cat_cols]
        miss_big_num = [col for col in miss_big if col in self.num_cols]
        cols_dict = {
            "miss_small_cat": miss_small_cat,
            "miss_small_num": miss_small_num,
            "miss_big_cat": miss_big_cat,
            "miss_big_num": miss_big_num,
        }
        with open(
            self.out_distri_dir + "/" + "missing_cols.json", "w", encoding="utf-8"
        ) as f:
            json.dump(cols_dict, f, ensure_ascii=False, indent=4)

        return miss_small_cat, miss_small_num, miss_big_cat, miss_big_num

    def process_small_missing_values(self):
        """
        处理小规模缺失值
        """
        self.df = median_mode_impute(self.df, self.miss_small_cat, self.miss_small_num)

    def process_by_hospital(self):
        """
        按医院中心分别处理大规模缺失值
        """
        mice_imputed = []
        hos_list = self.df[config.center].unique()
        print("hos_list: ", hos_list)

        for hos in hos_list:
            print(f"hos: {hos}")
            df_hos = self.df[self.df[config.center] == hos].copy()
            df_hos.reset_index(drop=True, inplace=True)

            isna = df_hos.isna().mean()
            if isna.max() > 0.05:
                data_imputed = mice_impute(
                    df_hos,
                    cat_str_cols=self.cat_str_cols,
                    miss_big_cat=self.miss_big_cat,
                    num_cols=self.num_cols,
                    id_cols=self.id_cols,
                    model_miss_encoder_dir=self.model_miss_encoder_dir,
                    model_mice_dir=self.model_mice_dir,
                )
                mice_imputed.append(data_imputed)
            else:
                mice_imputed.append(df_hos)

        self.df = pd.concat(mice_imputed, ignore_index=True)

    def save_results(self):
        """
        保存结果到文件
        """
        self.df.to_excel(f"{self.output_dir}/{self.output_file}", index=False)

        # 打印插补后的缺失情况
        cnt = self.df[self.miss_big_cat[0]].value_counts(dropna=False)
        print(f"插补后{self.miss_big_cat[0]}缺失情况：", cnt)

    def visualize_results(self, suffix="clean"):
        """
        可视化插补结果
        """
        feat_distribution(self.df, output_dir=self.out_distri_dir, suffix=suffix)

    def run(self):
        """
        运行完整的插补流程
        """
        self.visualize_results(suffix="clean")

        print("开始处理小规模缺失值...")
        self.process_small_missing_values()

        print("开始按医院中心处理大规模缺失值...")
        self.process_by_hospital()

        print("保存结果...")
        self.save_results()

        print("生成可视化结果...")
        self.visualize_results(suffix="imputed")

        print("插补完成！")


def impute_compare(
    impute_before_file, impute_after_file, miss_json_file, out_distri_dir
):
    with open(miss_json_file, encoding="utf-8") as f:
        cols_dict = json.load(f)
    miss_small_cat = cols_dict["miss_small_cat"]
    miss_small_num = cols_dict["miss_small_num"]
    miss_big_cat = cols_dict["miss_big_cat"]
    miss_big_num = cols_dict["miss_big_num"]
    cols = [config.pid] + miss_small_cat + miss_big_cat + miss_small_num + miss_big_num

    impute_before = pd.read_excel(impute_before_file)
    impute_before = impute_before[cols]
    impute_before["impute"] = "beforeImpute"

    impute_after = pd.read_excel(impute_after_file)
    impute_after = impute_after[cols]
    impute_after["impute"] = "afterImpute"

    df_cmp = pd.concat([impute_before, impute_after], ignore_index=True)
    evidently_wasserstein_psi(
        df_cmp,
        col="impute",
        cat_cols=miss_small_cat + miss_big_cat,
        num_cols=miss_small_num + miss_big_num,
        output_dir=out_distri_dir,
    )


def main():
    """
    主函数
    """

    mid_input_file = config.DATA_WIDE_FILE.replace(".xlsx", "_clean.xlsx")
    suffix = "控江医院"
    mid_input_file = mid_input_file.replace(
        ".xlsx", f"_clean_{suffix}_pid_dropped.xlsx"
    )

    imputer = DataImputer(
        mid_input_file,
        out_distri_dir=f"/distribution_analysis_{suffix}",
        output_file=f"imputed_{suffix}.xlsx",
    )
    imputer.run()

    impute_before_file = mid_input_file
    impute_after_file = config.DATA_FINAL_DIR + f"/imputed_{suffix}.xlsx"
    out_distri_dir = config.DATA_MID_DIR + f"/distribution_analysis_{suffix}"
    miss_json_file = out_distri_dir + "/missing_cols.json"

    impute_compare(
        impute_before_file, impute_after_file, miss_json_file, out_distri_dir
    )


if __name__ == "__main__":
    main()
