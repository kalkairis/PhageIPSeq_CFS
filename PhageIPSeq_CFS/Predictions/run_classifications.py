import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneOut

from PhageIPSeq_CFS.config import predictors_info
from PhageIPSeq_CFS.helpers import split_xy_df_and_filter_by_threshold, get_data_with_outcome


def get_cross_validation_predictions(x, y, estimator):
    return pd.Series(index=y.index,
                     data=cross_val_predict(estimator, x, y,
                                            cv=LeaveOneOut(),
                                            method='predict_proba')[:, 1]).to_frame()

def add_level_for_predictions(level_range, level_name, level_function, *args, **kwargs):
    res = []
    for level_value in level_range:
        level_res = level_function(level_value, *args, **kwargs)
        num_index_cols = level_res.reset_index().shape[1] - level_res.shape[1]
        level_res[level_name] = level_value
        level_res.set_index(level_name, append=True, inplace=True)
        level_res = level_res.reorder_levels([-1] + list(range(num_index_cols)))
        res.append(level_res)
    res = pd.concat(res)
    return res



def get_predictions_for_all_thresholds(estimator, xy_df):
    def foo(threshold_percent):
        fillna = isinstance(estimator, GradientBoostingClassifier)
        x, y = split_xy_df_and_filter_by_threshold(xy_df, bottom_threshold=threshold_percent / 100, fillna=fillna)
        return get_cross_validation_predictions(x, y, estimator)
    res = add_level_for_predictions(level_range=[0, 1, 5, 10, 20, 50, 95, 100],
                                    level_name='threshold', level_function=foo)
    return res


if __name__ == "__main__":
    res = {}
    for estimator_name, estimator_info in predictors_info.items():
        estimator = estimator_info['predictor_class'](**estimator_info['predictor_kwargs'])
        xy_df = get_data_with_outcome(data_type='fold', subgroup='all', with_bloodtests=False,
                                      with_oligos=True, imputed=False)
        res[estimator_name] = get_predictions_for_all_thresholds(estimator, xy_df)

    print("here")
