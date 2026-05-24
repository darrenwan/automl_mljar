# SMD (Cohen's d)
import pingouin as pg

# Cohen's d (SMD)
d = pg.compute_effsize(
    df_A["WBC"],
    df_B["WBC"],
    eftype="cohen"
)

print(d)

# Wasserstein 距离
from scipy.stats import wasserstein_distance
dist = wasserstein_distance(sample_A, sample_B)

#import scorecardpy as sc

psi = sc.psi(
    df_A["SEX"],
    df_B["SEX"]
)

print(psi)

import scipy.stats as stats
import numpy as np

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape

    return np.sqrt(
        chi2 / (n * (min(r-1, k-1)))
    )

# pycm（推荐）：cramer's V   多种分类统计量