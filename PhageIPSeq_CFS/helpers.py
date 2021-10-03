import os
from typing import Tuple

import pandas as pd
from sklearn.impute import SimpleImputer

from PhageIPSeq_CFS.config import repository_data_dir


def get_individuals_metadata_df():
    meta = pd.read_csv(os.path.join(repository_data_dir, 'individuals_metadata.csv'), index_col=0, low_memory=False)
    return meta.applymap(to_numeric)


def get_oligos_df(data_type: str = 'fold') -> pd.DataFrame:
    """
    Get the oligos DataFrame
    :param data_type: type of data on oligos ('fold', 'exist', or 'p_val'). Default: 'fold'
    :return:
    """
    assert data_type in ['fold', 'exist', 'p_val']
    ret = pd.read_csv(os.path.join(repository_data_dir, f"{data_type}_df.csv"), index_col=0, low_memory=False)
    return ret.loc[:, ret.notnull().any()].copy()


def get_outcome(return_type=int) -> pd.Series:
    return get_individuals_metadata_df()['catrecruit_Binary'].astype(bool).rename('is_CFS').astype(return_type)


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
    ret = pd.read_csv(os.path.join(repository_data_dir, 'oligos_metadata.csv'), index_col=0, low_memory=False)
    return ret


def get_oligos_metadata_subgroup(data_type: str = 'fold', subgroup: str = 'is_bac_flagella') -> pd.DataFrame:
    oligos_df = get_oligos_df(data_type=data_type)
    metadata = get_oligos_metadata()[subgroup]
    ret = oligos_df.loc[:, metadata]
    return ret


def get_oligos_metadata_subgroup_with_outcome(data_type: str = 'fold',
                                              subgroup: str = 'is_bac_flagella') -> pd.DataFrame:
    oligos_df = get_oligos_with_outcome(data_type=data_type)
    if subgroup != 'all':
        metadata = get_oligos_metadata()[subgroup]
        oligos_df = oligos_df.loc[:, metadata]
    return oligos_df


def split_xy_df_and_filter_by_threshold(xy_df: pd.DataFrame, bottom_threshold: float = 0.05) -> Tuple[
    pd.DataFrame, pd.Series]:
    filter_function = pd.notnull if xy_df.isnull().any().any() else lambda x: x != 0
    xy_df = xy_df.loc[:, xy_df.applymap(filter_function).mean().ge(bottom_threshold)]
    y = xy_df.reset_index(level=0)[xy_df.index.names[0]]
    x = xy_df.reset_index(level=0, drop=True).fillna(0)
    return x, y


def get_imputed_individuals_metadata(**impute_kwargs):
    meta = get_individuals_metadata_df().drop(columns='catrecruit_Binary')
    imputes = SimpleImputer(**impute_kwargs)
    ret = pd.DataFrame(data=imputes.fit_transform(meta), columns=meta.columns, index=meta.index)
    return ret


def to_numeric(x):
    if str(x).startswith('<') or str(x).startswith('>'):
        x = x[1:]
    return pd.to_numeric(x)


def get_oligos_blood_with_outcome(data_type: str = 'fold', subgroup: str = 'is_bac_flagella',
                                  **impute_kwargs) -> pd.DataFrame:
    oligos_and_outcome = get_oligos_metadata_subgroup_with_outcome(data_type=data_type, subgroup=subgroup)
    meta = get_imputed_individuals_metadata(**impute_kwargs)
    ret = pd.merge(oligos_and_outcome, meta, right_index=True, left_on='sample_id', how='inner')
    return ret


def get_data_with_outcome(data_type: str = 'fold', subgroup: str = 'all', with_bloodtests: bool = False, with_oligos:bool=True,
                          **impute_kwargs):
    if with_oligos:
        if with_bloodtests:
            return get_oligos_blood_with_outcome(data_type=data_type, subgroup=subgroup, **impute_kwargs)
        else:
            return get_oligos_metadata_subgroup_with_outcome(data_type=data_type, subgroup=subgroup)
    elif with_bloodtests:
        ret = get_imputed_individuals_metadata()
        outcome = get_outcome()
        return pd.merge(ret, outcome, left_index=True, right_index=True).set_index('is_CFS', append=True).reorder_levels([1, 0])
    return pd.DataFrame()

