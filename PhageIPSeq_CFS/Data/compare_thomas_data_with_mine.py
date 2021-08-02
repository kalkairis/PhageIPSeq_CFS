import os

import pandas as pd

from PhageIPSeq_CFS.config import repository_data_dir
from PhageIPSeq_CFS.helpers import get_oligos_metadata_subgroup_with_outcome, split_xy_df_and_filter_by_threshold


def get_thomas_data():
    x = pd.read_csv(os.path.join(repository_data_dir, 'tmp', 'x.csv'), index_col=0)
    y = pd.read_csv(os.path.join(repository_data_dir, 'tmp', 'y.csv'), index_col=0)
    return {'x': x, 'y': y}


if __name__ == "__main__":
    thomas_data = get_thomas_data()
    getter_data = dict(zip(['x', 'y'], split_xy_df_and_filter_by_threshold(
        get_oligos_metadata_subgroup_with_outcome(data_type='fold', subgroup='is_PNP'), bottom_threshold=0.05)))
    print("here")
