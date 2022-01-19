import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from PhageIPSeq_CFS.Predictions.run_classifications_with_oligos import get_cross_validation_predictions, \
    add_level_for_predictions
from PhageIPSeq_CFS.config import predictors_info, predictions_outcome_dir, oligo_families
from PhageIPSeq_CFS.helpers import get_data_with_outcome, split_xy_df_and_filter_by_threshold


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
        xy_df = pd.concat([
            get_data_with_outcome(*args, data_type=data_type, with_bloodtests=False, with_oligos=True, **kwargs),
            get_data_with_outcome(*args, data_type=data_type,
                                  with_bloodtests=True, with_oligos=False,
                                  **kwargs)[['sex_Binary', 'agegroup_Average']]], axis=1)
        return get_predictions_for_all_thresholds(estimator, xy_df)

    res = add_level_for_predictions(level_range=['fold', 'exist'], level_name='data_type', level_function=foo)
    return res


def get_predictions_on_oligo_subgroups(estimator, *args, **kwargs):
    subgroups = oligo_families + ['all']
    res = add_level_for_predictions(level_range=subgroups, level_name='subgroup',
                                    level_function=get_predictions_on_all_data_types, estimator=estimator, *args,
                                    **kwargs)
    return res


if __name__ == "__main__":
    os.makedirs(predictions_outcome_dir, exist_ok=True)
    res = []
    for predictor_name, predictor_info in predictors_info.items():
        estimator = predictor_info['predictor_class'](**predictor_info['predictor_kwargs'])
        pred_res = get_predictions_on_oligo_subgroups(estimator=estimator)
        pred_res['estimator'] = predictor_name
        pred_res.set_index('estimator', append=True, inplace=True)
        res.append(pred_res)
    res = pd.concat(res)

    for predictor_name in predictors_info.keys():
        predictor_res = res.loc[res.reset_index()['estimator'].eq(predictor_name).values]
        predictor_res.reset_index().rename(
            columns={0: 'value', 'level_2': 'sample_id'}).to_csv(
            os.path.join(predictions_outcome_dir, f"{predictor_name}_oligos_only_with_age_and_gender_predictions.csv"))
