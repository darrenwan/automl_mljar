import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import chi2_contingency

from src.config import config

############################################
# 配置
############################################

CENTER_COL = config.center
LABEL_COL = config.label

# 可选协变量（推荐加入）
ADJUST_VARS = [config.gender, config.age]  # 若不存在会自动忽略

ALPHA = 0.05


# ========== 7. 综合判断逻辑 ==========
def judge_center_effect(
    data,
    missing_col="missing",
    center_col="center",
    covariates=["age", "gender", "group"],
):
    """
    综合判断缺失是否与中心效应相关

    返回:
        - conclusion: 结论字符串
        - evidence: 各项检验证据
    """
    evidence = {}

    # 1. 描述性统计
    center_rates = data.groupby(center_col)[missing_col].mean()
    evidence["rate_range"] = f"{center_rates.min():.2%} - {center_rates.max():.2%}"
    evidence["max_rate_center"] = center_rates.idxmax()

    # 2. 卡方检验
    contingency = pd.crosstab(data[center_col], data[missing_col])
    chi2, p_chi2, _, _ = chi2_contingency(contingency)
    evidence["chi2_p"] = round(p_chi2, 8)

    # 3. 多因素logistic回归
    hos_ref = data[center_col].value_counts().idxmax()
    formula = (
        f'{missing_col} ~ C({center_col}, Treatment(reference="{hos_ref}")) + '
        + " + ".join(covariates)
    )
    try:
        model = smf.logit(formula, data=data).fit(method="bfgs", maxiter=100, disp=0)
    except Exception as e:
        print(f"Warning: Could not fit full model due to convergence issues: {e}")
        # 如果收敛失败，尝试其他方法
        try:
            model = smf.logit(formula, data=data).fit(method="nm", maxiter=500, disp=0)
        except:
            # 如果仍然失败，返回默认结果
            evidence["lr_p"] = np.nan
            evidence["convergence_error"] = True
            # 仅基于卡方检验做判断
            if p_chi2 < 0.05:
                conclusion = (
                    "存在显著的中心效应（基于卡方检验），但logistic回归模型拟合失败"
                )
            else:
                conclusion = (
                    "未发现显著的中心效应（基于卡方检验），但logistic回归模型拟合失败"
                )
            return conclusion, evidence

    center_effects = model.summary2().tables[1]
    center = center_effects[center_effects.index.str.contains(config.center)]
    center_dict = dict(zip(center.index, center["P>|z|"]))
    evidence["center_p"] = center_dict.values()
    print("center_effects: ", center_effects)

    # 似然比检验
    formula_no_center = f"{missing_col} ~ " + " + ".join(covariates)
    try:
        model_no_center = smf.logit(formula_no_center, data=data).fit(
            method="bfgs", maxiter=100, disp=0
        )
    except Exception as e:
        print(f"Warning: Could not fit reduced model due to convergence issues: {e}")
        try:
            model_no_center = smf.logit(formula_no_center, data=data).fit(
                method="nm", maxiter=500, disp=0
            )
        except:
            # 如果拟合失败，使用卡方检验结果
            evidence["lr_p"] = np.nan
            if p_chi2 < 0.05:
                conclusion = (
                    "存在显著的中心效应（基于卡方检验），但logistic回归模型拟合失败"
                )
            else:
                conclusion = (
                    "未发现显著的中心效应（基于卡方检验），但logistic回归模型拟合失败"
                )
            return conclusion, evidence
    # 似然比检验
    lr_stat = 2 * (model.llf - model_no_center.llf)
    lr_p = 1 - stats.chi2.cdf(lr_stat, df=model.df_model - model_no_center.df_model)
    evidence["lr_p"] = round(lr_p, 8)

    # 4. 综合判断
    if p_chi2 < 0.05 and lr_p < 0.05:
        conclusion = "存在显著的中心效应，中心是缺失的独立影响因素"
    elif p_chi2 < 0.05:
        conclusion = "中心间缺失率有差异，但控制混杂后中心效应不独立"
    elif lr_p < 0.05:
        conclusion = "控制混杂后中心有独立效应，但原始缺失率差异不显著"
    else:
        conclusion = "未发现显著的中心效应"

    return conclusion, evidence, center_effects


def run_multicenter_missing_analysis(df, miss_big, output_dir):
    """
    缺失是否依赖于 Center
    p < 0.05 ⇒ 存在 center-dependent missingness  极其关键（多中心论文必须）
    """

    results = []
    vars_check = []
    center_dep_vars = []
    non_center_dep_vars = []
    print(df.info())
    df_test = df.copy()

    for var in miss_big:
        print("Processing:", var)
        vars_check.append(var)
        df_test["missing"] = df_test[var].isna().astype(int)
        conclusion, evidence, center_effects = judge_center_effect(
            df_test,
            missing_col="missing",
            center_col=config.center,
            covariates=[LABEL_COL, config.gender],
        )

        if "存在" in conclusion:
            center_dep_vars.append(var)
        else:
            non_center_dep_vars.append(var)

        results.append(
            {
                "feature": var,
                "conclusion": conclusion,
                "chi2_p": evidence["chi2_p"],
                "LR_p": evidence["lr_p"],
                "center_p": evidence["center_p"],
                "evidence": evidence,
            }
        )

    res = pd.DataFrame(results)
    res.to_excel(f"{output_dir}/center_missing_compare.xlsx", index=False)
    return center_dep_vars, non_center_dep_vars
