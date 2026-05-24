import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight
from supervised.automl import AutoML

from src.config import config


def auc_cm(
    automl,
    X_test,
    y_test,
    plot_curves=True,
    output_dir=None,
    suffix=None,
    pos_label=None,
):
    # 外层评估
    y_pred_proba = automl.predict_proba(X_test)[:, 1]
    y_pred = automl.predict(X_test)

    # 计算评估指标

    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    # 计算PR曲线相关指标
    precision, recall, _ = precision_recall_curve(
        y_test, y_pred_proba, pos_label=pos_label
    )
    avg_precision = average_precision_score(y_test, y_pred_proba, pos_label=pos_label)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=pos_label)

    results = {
        "auc": round(auc, 4),
        "avg_precision": round(avg_precision, 4),
        "confusion_matrix": cm.tolist(),  # 转换为列表以便JSON序列化
    }

    if plot_curves:
        # 设置seaborn样式
        sns.set_style("whitegrid")
        plt.style.use("default")

        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 使用seaborn绘制ROC曲线
        roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        sns.lineplot(
            data=roc_data,
            x="FPR",
            y="TPR",
            ax=axes[0],
            linewidth=2,
            label=f"ROC curve (AUC = {auc:.4f})",
        )
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("Receiver Operating Characteristic (ROC) Curve")
        axes[0].legend(loc="lower right")

        # 使用seaborn绘制PR曲线
        pr_data = pd.DataFrame({"Recall": recall, "Precision": precision})
        sns.lineplot(
            data=pr_data,
            x="Recall",
            y="Precision",
            ax=axes[1],
            linewidth=2,
            label=f"PR curve (AP = {avg_precision:.4f})",
        )
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall (PR) Curve")
        axes[1].legend(loc="lower left")

        plt.tight_layout()

        # 保存图片
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_filename = os.path.join(output_dir, f"roc_pr_curves_{suffix}.svg")
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            print(f"ROC and PR curves saved to: {plot_filename}")
        else:
            plt.show()

        plt.close()  # 关闭图形以释放内存

    print(f"results: \n{results}")
    return results


def train_explain_custom(
    df: pd.DataFrame,
    target_col,
    center_col,
    mode,
    results_path,
    suffix="",
    pos_label="pneumonia",
    sampling_strategy=0.5,
):
    from imblearn.under_sampling import RandomUnderSampler

    # 假设你的上感是 0 (多数类)，肺炎是 1 (少数类)
    # sampling_strategy = 0.5 的意思是：少数类 / 多数类 = 1/2，即 上感:肺炎 = 2:1
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    for hos in df[center_col].unique():
        df_hos = df[df[center_col] == hos]

        X_hos = df_hos.drop([target_col, center_col], axis=1)
        y_hos = df_hos[target_col]
        # ============================================================
        # 第三部分：数据划分（必须分层！）
        # ============================================================

        X_hos_train, X_hos_test, y_hos_train, y_hos_test = train_test_split(
            X_hos,
            y_hos,
            test_size=0.2,
            random_state=42,
            stratify=y_hos,  # ⭐ 关键：保持正负样本比例
        )
        # 1. 物理下采样 (注意：绝对只能对 X_train 和 y_train 进行，绝不能碰测试集！)
        X_train_resampled, y_train_resampled = rus.fit_resample(
            X_hos_train, y_hos_train
        )

        print(
            f"原始训练集形状: {X_hos_train.shape}, 比例: {y_hos_train.value_counts().to_dict()}"
        )
        print(
            f"下采样后形状: {X_train_resampled.shape}, 比例: {y_train_resampled.value_counts().to_dict()}"
        )

        X_train_list.append(X_train_resampled)
        y_train_list.append(y_train_resampled)
        X_test_list.append(X_hos_test)
        y_test_list.append(y_hos_test)

    X_train = pd.concat(X_train_list, axis=0, ignore_index=True)
    y_train = pd.concat(y_train_list, axis=0, ignore_index=True)
    X_test = pd.concat(X_test_list, axis=0, ignore_index=True)
    y_test = pd.concat(y_test_list, axis=0, ignore_index=True)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # 配置 MLJAR AutoML（内层交叉验证自动调参）
    automl = AutoML(
        # 算法配置
        results_path=results_path,
        mode=mode,
        golden_features=False,
        # features_selection=True,
        kmeans_features=False,
        model_time_limit=3600,
        algorithms=[
            "Linear",
            "Random Forest",
            "Extra Trees",
            "Xgboost",
            "CatBoost",
            # "Nearest Neighbors",
            # "Neural Network"  # MLP
        ],
        # validation_strategy={
        #     "validation_type": "kfold",
        #     "k_folds": 5,
        #     "shuffle": True,
        #     "stratify": True,  # 交叉验证也要分层
        #     "random_seed": 42
        # },
        # eval_metric="logloss", #"auc",
        train_ensemble=False,
        stack_models=False,
        explain_level=2,
        # 其他
        random_state=42,
        n_jobs=-1,  # 并行计算
    )

    # 训练模型（自动完成内层调参）
    automl.fit(X_train, y_train, sample_weight=sample_weights)
    print("Class distribution:")

    print("Sample weights summary:")
    print(pd.Series(sample_weights).describe())

    suffix_train = suffix + "_train"

    scores = auc_cm(
        automl,
        X_train,
        y_train,
        plot_curves=True,
        output_dir=results_path,
        suffix=suffix_train,
        pos_label=pos_label,
    )
    with open(results_path + f"/scores_{suffix_train}.json", "w") as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)

    suffix_test = suffix + "_test_inner"
    scores = auc_cm(
        automl,
        X_test,
        y_test,
        plot_curves=True,
        output_dir=results_path,
        suffix=suffix_test,
        pos_label=pos_label,
    )
    with open(results_path + f"/scores_{suffix_test}.json", "w") as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)


def model_predict(
    test_df: pd.DataFrame,
    target_col: str,
    other_cols: list[str],
    results_path: str,
    suffix: str,
    pos_label="pneumonia",
):
    from supervised import AutoML

    # 假设这是你今天新获取到的测试数据，需要做预测
    X_new = test_df.drop([target_col] + other_cols, axis=1)
    y_new = test_df[target_col]

    # ==========================================
    # ⭐ 如何加载保存的模型：
    # 只需要实例化 AutoML，并把 results_path 指向之前的文件夹即可！
    # ==========================================
    loaded_automl = AutoML(results_path=results_path)

    # ==========================================
    # ⭐ 直接预测！
    # MLJAR 会自动读取文件夹里的配置文件，
    # 自动找到当初胜出的“最佳模型”（或者 Ensemble 模型），
    # 自动套用之前的数据预处理，然后输出结果。
    # ==========================================
    suffix_test = suffix + "_test_outer"
    scores = auc_cm(
        loaded_automl,
        X_new,
        y_new,
        plot_curves=True,
        output_dir=results_path,
        suffix=suffix_test,
        pos_label=pos_label,
    )
    with open(results_path + f"/scores_{suffix_test}.json", "w") as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)


def feats_select(dropped_feats_file):
    with open(dropped_feats_file, encoding="utf-8") as file:
        feat_drops_json = json.load(file)
        feat_drops_json = [i for i in feat_drops_json if i not in ["random_feature"]]
    return feat_drops_json


def model_predict_out2xlsx(
    test_df: pd.DataFrame,
    target_col: str,
    other_cols: list[str],
    results_path: str,
    suffix: str,
):
    # 假设这是你今天新获取到的测试数据，需要做预测
    X_new = test_df.drop([target_col] + other_cols, axis=1)

    # ==========================================
    # ⭐ 如何加载保存的模型：
    # 只需要实例化 AutoML，并把 results_path 指向之前的文件夹即可！
    # ==========================================
    loaded_automl = AutoML(results_path=results_path)
    y_pred = loaded_automl.predict(X_new)
    y_pred_proba = loaded_automl.predict_proba(X_new)[:, 1]

    test_df["pred"] = y_pred
    test_df["pred_proba"] = y_pred_proba

    test_df["compare"] = test_df[target_col] == test_df["pred"]
    test_df = test_df[test_df["compare"] == False]

    file = f"{results_path}/drop_features.json"
    feat_drops_json = feats_select(file)
    test_df = test_df.drop(columns=feat_drops_json, axis=1)
    test_df.sort_values(
        by=[config.center, "pred_proba"], ascending=[True, False], inplace=True
    )

    outf = results_path + f"/{suffix}.xlsx"
    test_df.to_excel(outf, index=False)
