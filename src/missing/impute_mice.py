import json
import os

import joblib
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def median_mode_impute(df, small_cat_cols, small_num_cols):
    for c in small_cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])
    for c in small_num_cols:
        df[c] = df[c].fillna(df[c].median())
    return df


def mice_impute(data, cat_str_cols, miss_big_cat, num_cols, id_cols, model_miss_encoder_dir, model_mice_dir):
    """
    为什么用 LabelEncoder 而不是 OneHotEncoder？
    OneHotEncoder 会将多分类变量拆分为多个二元变量，增加特征数量，且插补时需同时处理多个哑变量，更复杂。
    LabelEncoder 简单高效，且树模型（如 RandomForest）不会被数值顺序误导（树模型分裂时只看类别差异，不假设顺序）。
    """

    label_encoders = {}
    data_encoded = data.drop(columns=id_cols)
    data_encoded_cols = data_encoded.columns.tolist()

    for col in cat_str_cols:
        # 仅对非缺失值编码，缺失值保持NaN
        non_missing = data_encoded[col].dropna()
        col_encoder_file = model_miss_encoder_dir + f'/cat_str_encoder_{col}.joblib'
        if not os.path.exists(col_encoder_file):
            le = LabelEncoder()
            data_encoded.loc[non_missing.index, col] = le.fit_transform(non_missing)
            # 保存编码器
            joblib.dump(le, col_encoder_file)
        else:
            le = joblib.load(col_encoder_file)
            data_encoded.loc[non_missing.index, col] = le.fit_transform(non_missing)

        label_encoders[col] = le

    mice_file = model_mice_dir + '/mice.joblib'
    if not os.path.exists(mice_file):
        rf_estimator = RandomForestRegressor(
            n_estimators=50,  # 树的数量，50-100即可，太大跑得慢
            max_depth=10,  # 限制深度防过拟合
            random_state=42,
            n_jobs=-1  # 调用所有CPU核心加速
        )
        imputer = IterativeImputer(
            estimator=rf_estimator,
            max_iter=50,  # 迭代次数（一般10-20次就收敛了）
            tol=1e-3,
            min_value=0,  # 【极其重要】检验指标（如白细胞、CRP）不可能为负数！
            random_state=42,
            verbose=0  # 打印插补进度
        )

        # 对所有变量进行插补（注意：插补结果是numpy数组）
        data_imputed_array = imputer.fit_transform(data_encoded)
        joblib.dump(imputer, mice_file)
    else:
        imputer = joblib.load(mice_file)
        data_imputed_array = imputer.transform(data_encoded)

    # 4. 转换回DataFrame并逆编码分类变量
    data_imputed = pd.DataFrame(data_imputed_array, columns=data_encoded_cols)

    # 分类变量逆编码（四舍五入到整数，因为分类标签是离散的）
    for col in cat_str_cols:
        data_imputed[col] = np.round(data_imputed[col]).astype(int)  # 确保是整数
        data_imputed[col] = label_encoders[col].inverse_transform(data_imputed[col])
    for col in miss_big_cat:
        if col not in cat_str_cols:
            data_imputed[col] = np.round(data_imputed[col]).astype(int)  # 确保是整数

    # 连续变量保持原样
    data_imputed[num_cols] = data_imputed[num_cols].astype(float)

    # print("\n插补后缺失情况：\n", data_imputed.isnull().sum())
    cnt = data_imputed[miss_big_cat[0]].value_counts(dropna=False)
    print(f"插补后{miss_big_cat[0]}缺失情况：{cnt}")
    data_ids = data[id_cols]
    data_imputed_ids = pd.concat([data_ids, data_imputed], axis=1)

    return data_imputed_ids
