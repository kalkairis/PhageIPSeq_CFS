import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneOut

from PhageIPSeq_CFS.config import predictors_info, oligo_families, predictions_outcome_dir
from PhageIPSeq_CFS.helpers import split_xy_df_and_filter_by_threshold, get_data_with_outcome, get_outcome


def get_cross_validation_predictions(x, y, estimator):
    try:
        ret = pd.Series(index=y.index,
                        data=cross_val_predict(estimator, x, y,
                                               cv=LeaveOneOut(),
                                               method='predict_proba', n_jobs=-1)[:, 1]).to_frame()
    except Exception as e:
        print("here")
    return ret


def add_level_for_predictions(level_range, level_name, level_function, *args, **kwargs):
    res = []
    for level_value in level_range:
        kwargs[level_name] = level_value
        level_res = level_function(*args, **kwargs)
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
        if x.shape[1] == 0:
            x['dummy_feature'] = 1
        return get_cross_validation_predictions(x, y, estimator)

    res = add_level_for_predictions(level_range=[0, 1, 5, 10, 20, 50, 95, 100],
                                    level_name='threshold_percent', level_function=foo)
    return res


def get_predictions_on_all_data_types(estimator=None, *args, **kwargs):
    def foo(data_type):
        xy_df = get_data_with_outcome(*args, data_type=data_type, **kwargs)
        return get_predictions_for_all_thresholds(estimator, xy_df)

    res = add_level_for_predictions(level_range=['fold', 'exist'], level_name='data_type', level_function=foo)
    return res


def get_predictions_on_oligo_subgroups(estimator, *args, **kwargs):
    subgroups = oligo_families + ['all']
    res = add_level_for_predictions(level_range=subgroups, level_name='subgroup',
                                    level_function=get_predictions_on_all_data_types, estimator=estimator, *args,
                                    **kwargs)
    return res


def run_predictions_on_blood_tests(estimator, with_oligos=True, *args, **kwargs):
    def foo(with_bloodtests):
        if with_bloodtests or with_oligos:
            return get_predictions_on_oligo_subgroups(estimator, *args, with_oligos=with_oligos,
                                                      with_bloodtests=with_bloodtests, **kwargs)
        return pd.DataFrame()

    res = add_level_for_predictions(level_range=[True, False], level_name='with_bloodtests', level_function=foo)
    return res


if __name__ == "__main__":
    os.makedirs(predictions_outcome_dir, exist_ok=True)
    res = {}
    for estimator_name, estimator_info in predictors_info.items():
        estimator = estimator_info['predictor_class'](**estimator_info['predictor_kwargs'])
        estimator_res = add_level_for_predictions(level_range=[True, False], level_name='with_oligos',
                                                  level_function=run_predictions_on_blood_tests, estimator=estimator,
                                                  imputed=False)
        estimator_res = estimator_res[0]
        res[estimator_name] = estimator_res
        estimator_res.to_csv(os.path.join(predictions_outcome_dir, f"{estimator_name}_predictions.csv"))
