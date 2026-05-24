import os

import joblib
import pandas as pd

from src.config import config
from src.model.feat_model import (
    model_predict,
    model_predict_out2xlsx,
    train_explain_custom,
)


def load_data(input_file: str, pos_label="pneumonia"):
    """
    加载数据并按类型分类列
    """
    model_encoder_dir = config.MODEL_ENCODER_DIR
    df = pd.read_excel(input_file)
    df[config.label] = (
        df[config.label].str.strip().map({"上感": "URTI", "肺炎": pos_label})
    )

    # # 找出object类型的列
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    object_cols = [
        col
        for col in object_cols
        if col not in [config.pid, config.center, config.label]
    ]

    # 对每个object类型的列进行标签编码
    from sklearn.preprocessing import LabelEncoder

    for col in object_cols:
        col_encoder_file = model_encoder_dir + f"/cat_encoder_{col}.joblib"

        if not os.path.exists(col_encoder_file):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            joblib.dump(le, col_encoder_file)
        else:
            le = joblib.load(col_encoder_file)
            df[col] = le.transform(df[col])

    print(df.info())
    return df


def run_main_t1p2(
    df: pd.DataFrame, mode="Perform", pos_label="pneumonia", sampling_strategy=0.5
):
    hos_rate = df[center_col].value_counts()
    hos_train = hos_rate.index[0]
    df_train = df[df[center_col] == hos_train]
    df_train.drop(columns=[pid_col], inplace=True)
    out_dir = config.DATA_FINAL_DIR + f"/results_{mode}-{hos_train}_{sampling_strategy}"

    train_explain_custom(
        df_train,
        target_col,
        center_col,
        mode,
        results_path=out_dir,
        suffix=hos_train,
        pos_label=pos_label,
        sampling_strategy=sampling_strategy,
    )
    model_predict_out2xlsx(
        df,
        target_col,
        other_cols=[center_col, pid_col],
        results_path=out_dir,
        suffix=f"predict_all_{mode}-{hos_train}train",
    )

    for hos in hos_rate.index[1:]:
        df_hos = df[df[center_col] == hos]
        model_predict(
            df_hos,
            target_col,
            other_cols=[center_col, pid_col],
            results_path=out_dir,
            suffix=hos,
            pos_label=pos_label,
        )


def run_main_t123(
    df: pd.DataFrame, mode="Perform", pos_label="pneumonia", sampling_strategy=0.5
):
    hos_train = "3个医院"
    df_train = df.copy()
    df_train.drop(columns=[pid_col], inplace=True)
    out_dir = config.DATA_FINAL_DIR + f"/results_{mode}-{hos_train}_{sampling_strategy}"

    train_explain_custom(
        df_train,
        target_col,
        center_col,
        mode,
        results_path=out_dir,
        suffix=hos_train,
        pos_label=pos_label,
        sampling_strategy=sampling_strategy,
    )

    model_predict_out2xlsx(
        df,
        target_col,
        other_cols=[center_col, pid_col],
        results_path=out_dir,
        suffix=f"predict_all_{mode}-{hos_train}train",
    )


def run_main_t12p3(df: pd.DataFrame, mode="Perform", pos_label="pneumonia"):
    hos_rate = df[center_col].value_counts()
    hos_train_0 = hos_rate.index[0]
    hos_train_1 = hos_rate.index[1]
    hos_train_2 = hos_rate.index[2]

    # 样本量排名第一、第二的用于训练
    hos_train = f"{hos_train_0}_{hos_train_1}"

    df_train = df[df[center_col].isin([hos_train_0, hos_train_1])]
    df_train.drop(columns=[pid_col], inplace=True)

    train_explain_custom(
        df_train,
        target_col,
        center_col,
        mode,
        results_path=config.DATA_FINAL_DIR + f"/results_{mode}-{hos_train}",
        suffix=hos_train,
        pos_label=pos_label,
    )

    df_hos = df[df[center_col] == hos_train_2]
    model_predict(
        df_hos,
        target_col,
        other_cols=[center_col, pid_col],
        results_path=config.DATA_FINAL_DIR + f"/results_{mode}-{hos_train}",
        suffix=hos_train_2,
        pos_label=pos_label,
    )

    # 样本量排名第一、第三的用于训练
    hos_train = f"{hos_train_0}_{hos_train_2}"
    df_train = df[df[center_col].isin([hos_train_0, hos_train_2])]
    df_train.drop(columns=[pid_col], inplace=True)

    train_explain_custom(
        df_train,
        target_col,
        center_col,
        mode,
        results_path=config.DATA_FINAL_DIR + f"/results_{mode}-{hos_train}",
        suffix=hos_train,
        pos_label=pos_label,
    )

    df_hos = df[df[center_col] == hos_train_1]
    model_predict(
        df_hos,
        target_col,
        other_cols=[center_col, pid_col],
        results_path=config.DATA_FINAL_DIR + f"/results_{mode}-{hos_train}",
        suffix=hos_train_1,
        pos_label=pos_label,
    )


if __name__ == "__main__":
    # D:\anaconda\envs\py312_mljar\Lib\site-packages\supervised\utils\shap.py

    """
    # 在代码的 PlotSHAP 类的 is_available 静态方法中（大约在代码的第 38-41 行）：
    @staticmethod
    def is_available(algorithm, X_train, y_train, ml_task):
        if not shap_pacakge_available:
            return False
        # https://github.com/mljar/mljar-supervised/issues/112 disable for NN
        # https://github.com/mljar/mljar-supervised/issues/114 disable for CatBoost
        if algorithm.algorithm_short_name in ["Baseline", "Neural Network", "CatBoost"]:
            return False

    # PlotSHAP.compute() line200-219
    with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    explainer = PlotSHAP.get_explainer(algorithm, X_train)
    X_vald, y_vald = PlotSHAP.get_sample(X_validation, y_validation)
    shap_values = explainer.shap_values(X_vald)
    if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        if ml_task == MULTICLASS_CLASSIFICATION:
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        else:
            shap_values = shap_values[:, :, -1]

    # fix problem with 1 or 2 dimensions for binary classification
    expected_value = explainer.expected_value
    if ml_task == BINARY_CLASSIFICATION:
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        if isinstance(expected_value, list) and len(expected_value) > 1:
            expected_value = expected_value[1]
        elif isinstance(expected_value, np.ndarray) and expected_value.size > 1:
            expected_value = expected_value[1]

    """
    modes = ["Explain", "Perform", "Compete", "Optuna"]
    suffixes = ["控江医院", "3个医院"]
    sampling_strategies = [0.2, 0.5, 1.0]
    for sampling_strategy in sampling_strategies[-1:]:
        for suffix in suffixes[:]:
            impute_after_file = config.DATA_FINAL_DIR + f"/imputed_{suffix}.xlsx"
            mode = modes[1]

            pos_label = "pneumonia"
            center_col = config.center
            target_col = config.label
            pid_col = config.pid
            df = load_data(impute_after_file, pos_label)
            if suffix == "控江医院":
                run_main_t1p2(
                    df,
                    mode=mode,
                    pos_label=pos_label,
                    sampling_strategy=sampling_strategy,
                )
            else:
                run_main_t123(
                    df,
                    mode=mode,
                    pos_label=pos_label,
                    sampling_strategy=sampling_strategy,
                )

        # run_main_t12p3(df, mode=mode, pos_label=pos_label)
