import os

import pandas as pd
from sklearn.model_selection import LeaveOneOut

from PhageIPSeq_CFS.Predictions.run_classifications_with_oligos import add_level_for_predictions, \
    run_predictions_on_blood_tests
from PhageIPSeq_CFS.config import predictions_outcome_dir, nn_predictors_info
from PhageIPSeq_CFS.helpers import split_xy_df_and_filter_by_threshold, get_data_with_outcome

if __name__ == "__main__":
    # Blood tests only
    os.makedirs(predictions_outcome_dir, exist_ok=True)
    res = {}
    # for estimator_name, estimator_info in nn_predictors_info.items():
    #     estimator = estimator_info['predictor_class'](**estimator_info['predictor_kwargs'])
    #     x, y = split_xy_df_and_filter_by_threshold(
    #         get_data_with_outcome(with_oligos=False, with_bloodtests=True, imputed=False))
    #     loo = LeaveOneOut()
    #     y_preds = {}
    #     for train_idx, test_idx in loo.split(x):
    #         x_train = x.iloc[train_idx]
    #         y_train = y.loc[x_train.index]
    #         x_test = x.iloc[test_idx]
    #         y_test = y.loc[x_test.index]
    #         try:
    #             estimator.fit(x_train, y_train)
    #             y_preds[y_test.index.values[0]] = estimator.predict_proba(x_test)[0][1]
    #         except:
    #             raise IOError('error')
    #     res[estimator_name] = pd.Series(y_preds)
    # res = pd.DataFrame(res)
    # res.to_csv(os.path.join(predictions_outcome_dir, 'NN_blood_test_only_predictions.csv'))

    res = {}
    for estimator_name, estimator_info in nn_predictors_info.items():
        estimator = estimator_info['predictor_class'](**estimator_info['predictor_kwargs'])
        estimator_res = add_level_for_predictions(level_range=[True, False], level_name='with_oligos',
                                                  level_function=run_predictions_on_blood_tests, estimator=estimator,
                                                  imputed=False)
        estimator_res = estimator_res[0]
        res[estimator_name] = estimator_res
        estimator_res.to_csv(os.path.join(predictions_outcome_dir, f"{estimator_name}_predictions.csv"))
