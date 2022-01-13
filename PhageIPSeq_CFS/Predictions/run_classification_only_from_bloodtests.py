import os

import pandas as pd
from sklearn.model_selection import cross_val_predict, LeaveOneOut

from PhageIPSeq_CFS.config import predictions_outcome_dir, predictors_info
from PhageIPSeq_CFS.helpers import get_data_with_outcome, split_xy_df_and_filter_by_threshold

if __name__ == "__main__":
    os.makedirs(predictions_outcome_dir, exist_ok=True)
    res = {}
    for estimator_name, estimator_info in predictors_info.items():
        estimator = estimator_info['predictor_class'](**estimator_info['predictor_kwargs'])
        x, y = split_xy_df_and_filter_by_threshold(
            get_data_with_outcome(with_oligos=False, with_bloodtests=True, imputed=False))
        res[estimator_name] = pd.Series(index=y.index,
                                        data=cross_val_predict(estimator, x, y,
                                                               cv=LeaveOneOut(),
                                                               method='predict_proba')[:, 1])
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(predictions_outcome_dir, 'blood_test_only_predictions.csv'))
