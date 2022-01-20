import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from PhageIPSeq_CFS.config import predictors_info, predictions_outcome_dir
from PhageIPSeq_CFS.helpers import get_outcome, get_data_with_outcome, split_xy_df_and_filter_by_threshold


def get_summary(df, num_rounds=100):
    res = {}
    df['y'] = get_outcome().loc[df.sample_id.values].values
    auc_value = roc_auc_score(df['y'], df['0'])
    auc_results = []
    for _ in range(num_rounds):
        sub_df = train_test_split(df)[0]
        auc_results.append(roc_auc_score(sub_df['y'], sub_df['0']))
    res['auc'] = auc_value
    res['std'] = round(np.std(auc_results), 3)
    assert df.reset_index().drop(
        columns=['0', 'y', 'sample_id', 'index'], errors='ignore').drop_duplicates().shape[0] == 1
    params = df.reset_index().drop(columns=['0', 'y', 'sample_id', 'index'], errors='ignore').drop_duplicates().T[
        0].to_dict()
    return res


if __name__ == "__main__":
    for estimator_name, estimator_info in predictors_info.items():
        # Summarize main results
        predictions = pd.read_csv(os.path.join(predictions_outcome_dir, f"{estimator_name}_predictions.csv"))
        summary = predictions.groupby(['with_oligos', 'with_bloodtests', 'subgroup', 'data_type',
                                       'threshold_percent']).apply(get_summary)
        summary = summary.to_frame()[0].apply(pd.Series)
        # Add number of oligos
        tmp = summary.reset_index().drop(
            columns=['auc', 'std', 'with_bloodtests', 'threshold_percent', 'with_oligos']).drop_duplicates()
        tmp['df'] = tmp.apply(lambda row: get_data_with_outcome(**dict(row)), axis=1)
        tmp.set_index(['subgroup', 'data_type'], inplace=True)
        tmp2 = summary.reset_index().drop(columns=['auc', 'std', 'with_bloodtests', 'with_oligos']).drop_duplicates()
        tmp2['num oligos'] = tmp2.apply(lambda row: split_xy_df_and_filter_by_threshold(tmp.loc[(row['subgroup'], row['data_type'])]['df'], bottom_threshold=row['threshold_percent'] / 100)[0].shape[1], axis=1)
        tmp2['with_oligos'] = True
        summary = summary.reset_index().merge(tmp2, on=['subgroup', 'data_type', 'threshold_percent', 'with_oligos'], how='left').fillna(0).set_index(['with_oligos', 'with_bloodtests', 'subgroup', 'data_type', 'threshold_percent'])
        for with_oligos in [True, False]:
            for with_bloodtests in [True, False]:
                if (not with_oligos) and (not with_bloodtests):
                    continue
                tmp = summary.loc[(with_oligos, with_bloodtests)].unstack(0).T
                tmp = tmp.reorder_levels([1, 0]).sort_index(level=0, sort_remaining=False).T
                tmp.to_csv(os.path.join(predictions_outcome_dir,
                                        f"predictions_summary_{estimator_name}" + (
                                            "_with_oligos" if with_oligos else "") + (
                                            "_with_bloodtests" if with_bloodtests else "") + '.csv'))
        # Summarize results of oligos with age and gender
        predictions = pd.read_csv(
            os.path.join(predictions_outcome_dir, f"{estimator_name}_oligos_only_with_age_and_gender_predictions.csv"),
            index_col=0)
        summary = predictions.rename(columns={'level_3': 'sample_id', 'value': '0'}).groupby(
            ['estimator', 'subgroup', 'data_type', 'threshold_percent']).apply(get_summary).to_frame()[0].apply(
            pd.Series)
        summary = summary.reset_index().merge(tmp2.drop(columns='with_oligos'),
                                              on=['subgroup', 'data_type', 'threshold_percent'], how='left').set_index(
            ['estimator', 'subgroup', 'data_type', 'threshold_percent'])
        summary = summary.loc[estimator_name].unstack(0)
        summary = summary.T.reorder_levels([1, 0]).sort_index(level=0, sort_remaining=False).T
        summary.to_csv(
            os.path.join(predictions_outcome_dir, f"predictions_summary_{estimator_name}_oligos_age_and_gender.csv"))
    print("done")
