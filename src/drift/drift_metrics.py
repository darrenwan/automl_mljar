import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, chi2_contingency

"""
包含：

SMD
Wasserstein
PSI
Cramer's V
"""


############################################
# SMD (Standardized Mean Difference)
############################################

def compute_smd(x_ref, x_cur):

    x_ref = x_ref.dropna()
    x_cur = x_cur.dropna()

    if len(x_ref) == 0 or len(x_cur) == 0:
        return np.nan

    mu1 = np.mean(x_ref)
    mu2 = np.mean(x_cur)

    sd1 = np.std(x_ref, ddof=1)
    sd2 = np.std(x_cur, ddof=1)

    pooled_sd = np.sqrt(
        (sd1**2 + sd2**2) / 2
    )

    if pooled_sd == 0:
        return 0

    return abs(mu1 - mu2) / pooled_sd


############################################
# Wasserstein
############################################

def compute_wasserstein(x_ref, x_cur):

    x_ref = x_ref.dropna()
    x_cur = x_cur.dropna()

    if len(x_ref) == 0 or len(x_cur) == 0:
        return np.nan

    return wasserstein_distance(x_ref, x_cur)


############################################
# PSI
############################################

def compute_psi(ref, cur):

    ref = ref.fillna("MISSING")
    cur = cur.fillna("MISSING")

    categories = list(
        set(ref.unique()) |
        set(cur.unique())
    )

    psi = 0

    for cat in categories:

        ref_pct = (ref == cat).mean()
        cur_pct = (cur == cat).mean()

        ref_pct = max(ref_pct, 1e-6)
        cur_pct = max(cur_pct, 1e-6)

        psi += (
            (cur_pct - ref_pct) *
            np.log(cur_pct / ref_pct)
        )

    return psi


############################################
# Cramer's V
############################################

def compute_cramers_v(x, center):

    cm = pd.crosstab(x, center)

    chi2 = chi2_contingency(cm)[0]

    n = cm.sum().sum()

    r, k = cm.shape

    return np.sqrt(
        chi2 / (n * (min(r - 1, k - 1)))
    )