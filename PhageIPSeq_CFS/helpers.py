import os

import pandas as pd

from PhageIPSeq_CFS.config import repository_data_dir


def get_metadata_df():
    return pd.read_csv(os.path.join(repository_data_dir, 'individuals_metadata.csv'), index_col=0)


def get_oligos_df(data_type: str = 'fold') -> pd.DataFrame:
    """
    Get the oligos DataFrame
    :param data_type: type of data on oligos ('fold', 'exist', or 'p_val'). Default: 'fold'
    :return:
    """
    assert data_type in ['fold', 'exist', 'p_val']
    ret = pd.read_csv(os.path.join(repository_data_dir, f"{data_type}_df.csv"), index_col=0)
    return ret.loc[:, ret.notnull().any()].copy()


def get_outcome(return_type=int) -> pd.Series:
    return get_metadata_df()['catrecruit_Binary'].astype(bool).rename('is_CFS').astype(return_type)


def get_oligos_with_outcome(data_type: str = 'fold') -> pd.DataFrame:
    metadata_df = get_outcome()
    oligos_df = get_oligos_df(data_type=data_type)
    ret = pd.merge(metadata_df,
                   oligos_df,
                   left_index=True,
                   right_index=True,
                   how='inner').set_index('is_CFS',
                                          append=True).reorder_levels([1, 0])
    ret.index.rename(['is_CFS', 'sample_id'], inplace=True)
    return ret


def get_oligos_metadata():
    ret = pd.read_csv(os.path.join(repository_data_dir, 'oligos-metadata.csv'), index_col=0)
    return ret


def get_oligos_metadata_subgroup(data_type: str = 'fold', subgroup: str = 'is_bac_flagella') -> pd.DataFrame:
    oligos_df = get_oligos_df(data_type=data_type)
    metadata = get_oligos_metadata()[subgroup]
    ret = oligos_df.loc[metadata]
    return ret


get_oligos_metadata_subgroup()
