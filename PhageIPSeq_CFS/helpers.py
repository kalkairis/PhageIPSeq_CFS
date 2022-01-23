import numpy as np
import os
from typing import Tuple

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer

from PhageIPSeq_CFS.config import repository_data_dir, RANDOM_STATE
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split


def get_individuals_metadata_df(threshold_for_removal = 10):
    meta = pd.read_csv(os.path.join(repository_data_dir, 'individuals_metadata.csv'), index_col=0, low_memory=False)
    cols_to_remove = meta.groupby('catrecruit_Binary').count().diff().dropna().T.abs()[1].sort_values(ascending=False)
    cols_to_remove = cols_to_remove[cols_to_remove.gt(threshold_for_removal)].index
    meta.drop(columns=cols_to_remove, inplace=True)
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


def split_xy_df_and_filter_by_threshold(xy_df: pd.DataFrame, bottom_threshold: float = 0.05, fillna=True) -> Tuple[
    pd.DataFrame, pd.Series]:
    filter_function = pd.notnull if xy_df.isnull().any().any() else lambda x: x != 0
    xy_df = xy_df.loc[:, xy_df.applymap(filter_function).mean().ge(bottom_threshold)]
    y = xy_df.reset_index(level=0)[xy_df.index.names[0]]
    x = xy_df.reset_index(level=0, drop=True)
    if fillna:
        x = x.fillna(0)
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


def get_oligos_blood_with_outcome(data_type: str = 'fold', subgroup: str = 'is_bac_flagella', imputed=True,
                                  **impute_kwargs) -> pd.DataFrame:
    oligos_and_outcome = get_oligos_metadata_subgroup_with_outcome(data_type=data_type, subgroup=subgroup)
    if imputed:
        meta = get_imputed_individuals_metadata(**impute_kwargs)
    else:
        meta = get_individuals_metadata_df().drop(columns='catrecruit_Binary')
    ret = pd.merge(oligos_and_outcome, meta, right_index=True, left_on='sample_id', how='inner')
    return ret


def get_data_with_outcome(data_type: str = 'fold', subgroup: str = 'all', with_bloodtests: bool = False,
                          with_oligos: bool = True, imputed=True,
                          **impute_kwargs):
    if with_oligos:
        if with_bloodtests:
            return get_oligos_blood_with_outcome(data_type=data_type, subgroup=subgroup, imputed=imputed,
                                                 **impute_kwargs)
        else:
            return get_oligos_metadata_subgroup_with_outcome(data_type=data_type, subgroup=subgroup)
    elif with_bloodtests:
        if imputed:
            ret = get_imputed_individuals_metadata()
        else:
            ret = get_individuals_metadata_df().drop(columns='catrecruit_Binary')
        outcome = get_outcome()
        return pd.merge(ret, outcome, left_index=True, right_index=True).set_index('is_CFS',
                                                                                   append=True).reorder_levels([1, 0])
    return pd.DataFrame()


def compute_auc_from_prediction_results(prediction_results, return_fprs_tprs=False):
    # create roc curve
    prediction_results = prediction_results.sort_values(by='predict_proba', ascending=False)
    tpr = [0]
    fpr = [0]
    for i in range(prediction_results.shape[0]):
        if prediction_results.iloc[i]['y'] == 1:
            tpr.append(tpr[-1] + 1)
            fpr.append(fpr[-1])
        else:
            tpr.append(tpr[-1])
            fpr.append(fpr[-1] + 1)
    tprs = np.array(tpr) / prediction_results.y.sum()
    fprs = np.array(fpr) / prediction_results.y.eq(0).sum()
    auc_value = auc(fprs, tprs)
    if return_fprs_tprs:
        return auc_value, fprs, tprs
    else:
        return auc_value


def get_x_y(bottom_threshold, data_type, oligos_subgroup, with_bloodtests, imputed=False):
    x, y = split_xy_df_and_filter_by_threshold(
        get_data_with_outcome(data_type=data_type, subgroup=oligos_subgroup, with_bloodtests=with_bloodtests,
                              imputed=imputed),
        bottom_threshold=bottom_threshold, fillna=imputed)
    return x, y


def create_auc_with_bootstrap_figure(num_confidence_intervals_repeats, x, y, predictor_class, color='blue', ax=None,
                                     prediction_results=None, chance_color='r', predictor_name='Predictor', *predictor_args,
                                     **predictor_kwargs):
    if prediction_results is None:
        prediction_results = run_leave_one_out_prediction(x, y, predictor_class, *predictor_args, **predictor_kwargs)
    auc_value, fprs, tprs = compute_auc_from_prediction_results(prediction_results, return_fprs_tprs=True)
    auc_confidence_interval = []
    if ax is None:
        fig, ax = plt.subplots()
    for _ in range(num_confidence_intervals_repeats):
        round_auc, round_fprs, round_tprs = compute_auc_from_prediction_results(train_test_split(prediction_results)[0],
                                                                                True)
        auc_confidence_interval.append(round_auc)
        ax.plot(round_fprs, round_tprs, color='grey', alpha=0.05)
    auc_std = np.std(auc_confidence_interval)
    ax.plot(fprs, tprs, color=color, label=f"{predictor_name} (AUC={round(auc_value, 3)}, std={round(auc_std, 3)})")
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color=chance_color,
            label='Chance', alpha=.8)
    ax.legend()
    ax.set_ylabel('True positive rate')
    ax.set_xlabel('False positive rate')
    return auc_std, auc_value


def run_leave_one_out_prediction(x, y, predictor_class, *predictor_args, **predictor_kwargs):
    ret = {}
    if x.empty:
        return dict()
    for sample in y.index.values:
        train_x = x.drop(index=sample)
        train_y = y.drop(index=sample)
        predictor_kwargs['random_state'] = RANDOM_STATE
        predictor = predictor_class(*predictor_args, **predictor_kwargs)
        predictor.fit(train_x, train_y.values.ravel())
        y_hat = predictor.predict(x.loc[sample].values.reshape(1, -1))[0]
        if isinstance(predictor, RidgeClassifier):
            predict_proba = predictor._predict_proba_lr(x.loc[sample].values.reshape(1, -1))[0][1]
        else:
            predict_proba = predictor.predict_proba(x.loc[sample].values.reshape(1, -1))[0][1]
        ret[sample] = {'y': y.loc[sample], 'y_hat': y_hat, 'predict_proba': predict_proba}
    return pd.DataFrame(ret).transpose()