import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

############################################
# Core Cramer's V
############################################

def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x, y)

    chi2 = chi2_contingency(
        confusion_matrix,
        correction=False
    )[0]

    n = confusion_matrix.sum().sum()

    r, k = confusion_matrix.shape

    return np.sqrt(
        (chi2 / n) /
        min(k - 1, r - 1)
    )

############################################
# Multi-center pipeline
############################################

def multi_center_cramers_v(
    df,
    center_col,
    cat_cols,
    threshold=0.1
):

    results = []

    for col in cat_cols:

        try:

            v = cramers_v(
                df[col],
                df[center_col]
            )

            results.append({
                "feature": col,
                "cramers_v": v,
                "flag": v > threshold
            })

        except:

            results.append({
                "feature": col,
                "cramers_v": np.nan,
                "flag": False
            })

    result_df = pd.DataFrame(results)

    return result_df.sort_values(
        "cramers_v",
        ascending=False
    )