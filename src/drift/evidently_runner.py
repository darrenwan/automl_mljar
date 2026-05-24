from evidently import Report
from evidently.legacy.calculations.stattests import psi_stat_test, wasserstein_stat_test

# from evidently.metrics import ColumnDriftMetric
# from evidently.calculations.stattests import (
#     psi_stat_test,
#     wasserstein_stat_test
# )
from evidently.legacy.metrics import ColumnDriftMetric
from evidently.legacy.options.data_drift import DataDriftOptions

options = DataDriftOptions(
    cat_features_stattest="psi", num_features_stattest="wasserstein"
)


def run_evidently(
    df, ref_center, cur_center, cat_features, num_features, center_col, output_file
):

    reference = df[df[center_col] == ref_center]

    current = df[df[center_col] == cur_center]

    metrics = []

    # categorical → PSI
    for col in cat_features:
        metrics.append(ColumnDriftMetric(column_name=col, stattest=psi_stat_test))

    # numerical → Wasserstein
    for col in num_features:
        metrics.append(
            ColumnDriftMetric(column_name=col, stattest=wasserstein_stat_test)
        )

    report = Report(metrics=metrics)

    report.run(reference_data=reference, current_data=current)

    report.save_html(output_file)
