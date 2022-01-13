import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from PhageIPSeq_CFS.config import predictors_info, predictions_outcome_dir
from PhageIPSeq_CFS.helpers import get_outcome, get_data_with_outcome


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
    # params = df.drop(columns=['0', 'y', 'sample_id']).iloc[0].to_dict()
    # res['num oligos'] = len([col for col in get_data_with_outcome(**params).columns if col.startswith('agilent_')])
    # res.update(params)
    return res


if __name__ == "__main__":
    for estimator_name, estimator_info in predictors_info.items():
        predictions = pd.read_csv(os.path.join(predictions_outcome_dir, f"{estimator_name}_predictions.csv"))
        summary = predictions.groupby(['with_oligos', 'with_bloodtests', 'subgroup', 'data_type',
                                       'threshold_percent']).apply(get_summary)
        summary = summary.to_frame()[0].apply(pd.Series)
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
    print("done")
