import pandas as pd
from scipy.stats import anderson_ksamp

from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently import ColumnType
from evidently.metrics import ValueDrift
from evidently.metrics import DriftedColumnsCount
from evidently.legacy.calculations.stattests import register_stattest
from evidently.legacy.calculations.stattests import StatTest

#toy data
data = pd.DataFrame(data={
    "column_1": [1, 2, 3, 4, -1, 5],
    "target": [1, 1, 0, 0, 1, 1],
    "prediction": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
})

definition = DataDefinition(
    numerical_columns=["column_1", "target", "prediction"],
    )
dataset = Dataset.from_pandas(
    data,
    data_definition=definition,
)

#implement method
def _addd(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: ColumnType,
    threshold: float,
):
    p_value = anderson_ksamp([reference_data.values, current_data.values])[2]
    return p_value, p_value < threshold


adt = StatTest(
    name="adt",
    display_name="Anderson-Darling",
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=0.1,
)

register_stattest(adt, default_impl=_addd)


report = Report([
    # ValueDrift(column="column_1"),
    ValueDrift(column="column_1", method="adt"),
    DriftedColumnsCount(),
])

snapshot = report.run(dataset, dataset)
snapshot