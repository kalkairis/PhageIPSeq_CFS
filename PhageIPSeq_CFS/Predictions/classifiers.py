import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from LabQueue.qp import qp, fakeqp
from LabUtils.addloglevels import sethandlers
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

from PhageIPSeq_CFS.config import visualizations_dir, logs_path, oligo_families, RANDOM_STATE, num_auc_repeats, \
    predictor_class, predictor_kwargs
from PhageIPSeq_CFS.helpers import split_xy_df_and_filter_by_threshold, \
    get_data_with_outcome, get_outcome, get_imputed_individuals_metadata


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


def get_prediction_results(output_dir, x, y, num_confidence_intervals_repeats=100,
                           predictor_class=GradientBoostingClassifier, *predictor_args, **predictor_kwargs):
    prediction_results = run_leave_one_out_prediction(x, y, predictor_class, *predictor_args, **predictor_kwargs)
    if len(prediction_results) == 0:
        return {}

    os.makedirs(output_dir, exist_ok=True)
    # create scatter of true and predicted values
    pearson_r, pearson_p_value = pearsonr(prediction_results['y'], prediction_results['predict_proba'])
    ax = sns.histplot(data=prediction_results, x='predict_proba', hue='y', element='step')
    ax.set_title(f'r={round(pearson_r, 2)}, p-value={pearson_p_value:.1g}')
    plt.savefig(os.path.join(output_dir, 'predicted_true_scatter.png'))
    plt.close()

    auc_std, auc_value = create_auc_with_bootstrap_figure(num_confidence_intervals_repeats, x, y, predictor_class,
                                                          ax=ax, prediction_results=prediction_results, *predictor_args,
                                                          **predictor_kwargs)
    plt.savefig(os.path.join(output_dir, 'prediction_auc.png'))
    plt.close()
    return {'auc': auc_value, 'auc_std': auc_std,
            'num_oligos': (x.columns.str.startswith('agilent_') | x.columns.str.startswith('twist_')).sum(),
            'num_features': x.shape[1]}


def create_auc_with_bootstrap_figure(num_confidence_intervals_repeats, x, y, predictor_class, ax=None,
                                     prediction_results=None, *predictor_args,
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
    ax = plt.subplot()
    ax.plot(fprs, tprs, color='b', label=f"Predictor (AUC={round(auc_value, 3)}, std={round(auc_std, 3)})")
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    plt.legend()
    return auc_std, auc_value


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


def get_x_y(bottom_threshold, data_type, oligos_subgroup, with_bloodtests):
    x, y = split_xy_df_and_filter_by_threshold(
        get_data_with_outcome(data_type=data_type, subgroup=oligos_subgroup, with_bloodtests=with_bloodtests),
        bottom_threshold=bottom_threshold)
    return x, y


def get_prediction_parameters(data_type, threshold_percent, figures_dir, oligos_subgroup='all',
                              num_confidence_intervals_repeats=100, predictor_class=GradientBoostingClassifier,
                              with_bloodtests=False, **predictor_kwargs):
    bottom_threshold = threshold_percent / 100
    x, y = get_x_y(bottom_threshold, data_type, oligos_subgroup, with_bloodtests)
    output_dir = os.path.join(figures_dir, data_type, str(bottom_threshold))
    return get_prediction_results(output_dir, x, y, num_confidence_intervals_repeats=num_confidence_intervals_repeats,
                                  predictor_class=predictor_class, **predictor_kwargs)


def get_top_prediction_shap_values(data_type, bottom_threshold, figures_dir, oligos_subgroup, predictor_class,
                                   with_bloodtests,
                                   *predictor_args,
                                   **predictor_kwargs):
    out_dir = os.path.join(figures_dir, 'best_run_results')
    os.makedirs(out_dir, exist_ok=True)
    bottom_threshold = bottom_threshold / 100
    x, y = get_x_y(bottom_threshold, data_type, oligos_subgroup, with_bloodtests)
    predictor = predictor_class(*predictor_args, **predictor_kwargs)
    model = predictor.fit(x, y)

    if isinstance(predictor, RidgeClassifier):
        explainer = shap.LinearExplainer(model, x)
        shap_values_ebm = explainer(x, silent=True)
        pd.Series(index=x.columns, data=predictor.coef_[0]).sort_values(ascending=False).to_csv(
            os.path.join(out_dir, 'feature_coefs.csv'))
    else:
        explainer = shap.TreeExplainer(model, x)
        shap_values_ebm = explainer(x)
        pd.Series(index=x.columns, data=predictor.feature_importances_).sort_values(ascending=False).to_csv(
            os.path.join(out_dir, 'feature_coefs.csv'))

    # the waterfall_plot shows how we get from explainer.expected_value to model.predict(X)[sample_ind]
    ax = plt.subplot()
    shap.plots.beeswarm(shap_values_ebm, max_display=14, show=False)
    ax.set_title(f"Shap for {data_type} threshold={bottom_threshold}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_beeswarm.png'))
    plt.close()


def predict_and_run_shap_on_oligo_subgroup(oligos_subgroup, figures_dir, num_repeats_in_auc_ci, predictore_class,
                                           with_bloodtests, **predictor_kwargs):
    curr_figures_dir = os.path.join(figures_dir, oligos_subgroup)
    with qp(f"preds") as q:
        q.startpermanentrun()
        waiton = {}
        for data_type in ['fold', 'exist']:
            for threshold_percent in [0, 1, 5, 10, 20, 50, 95, 100]:
                waiton[(data_type, threshold_percent)] = q.method(get_prediction_parameters,
                                                                  (data_type, threshold_percent, curr_figures_dir,
                                                                   oligos_subgroup, num_repeats_in_auc_ci,
                                                                   predictore_class, with_bloodtests),
                                                                  kwargs=predictor_kwargs)
        all_results = {k: q.waitforresult(v) for k, v in waiton.items()}
    all_results = pd.DataFrame(all_results).transpose().reset_index().rename(
        columns={'level_0': 'data_type', 'level_1': 'bottom_threshold'}).sort_values(by='auc', ascending=False)
    all_results.to_csv(os.path.join(curr_figures_dir, 'results_summary.csv'))
    all_results['oligos_subgroup'] = oligos_subgroup
    best_params = all_results.iloc[0][
        ['data_type', 'bottom_threshold']].to_dict()
    get_top_prediction_shap_values(**best_params, predictor_class=predictore_class, figures_dir=curr_figures_dir,
                                   with_bloodtests=with_bloodtests,
                                   oligos_subgroup=oligos_subgroup, **predictor_kwargs)
    return all_results


if __name__ == "__main__":
    sethandlers()
    os.chdir(logs_path)
    for with_bloodtests in [True, False]:
        if with_bloodtests:
            continue
        for with_oligos in [True, False]:
            if with_oligos == False and with_bloodtests == False:
                continue
            figures_dir = os.path.join(visualizations_dir,
                                       ''.join(['Predictions', '_with_bloodtests' if with_bloodtests else '',
                                                '_with_oligos' if with_oligos else '']))
            if with_oligos:
                all_results = {}
                with fakeqp(f"wpreds", max_u=10) as q:
                    q.startpermanentrun()
                    waiton = {}
                    for oligos_subgroup in oligo_families + ['all']:
                        waiton[oligos_subgroup] = q.method(predict_and_run_shap_on_oligo_subgroup,
                                                           (oligos_subgroup, figures_dir, num_auc_repeats,
                                                            predictor_class,
                                                            with_bloodtests), kwargs=predictor_kwargs)
                    all_results = {k: q.waitforresult(v) for k, v in waiton.items()}
                all_results = (pd
                               .concat(list(all_results.values()))
                               .set_index(['data_type', 'bottom_threshold', 'oligos_subgroup'])
                               .unstack('oligos_subgroup')
                               .swaplevel(axis='columns')
                               .sort_index(axis='columns'))
                all_results.to_csv(
                    os.path.join(figures_dir, 'best_results_summary.csv'))
            else:
                x = get_imputed_individuals_metadata()
                y = get_outcome()
                ret = get_prediction_results(figures_dir, x, y, num_confidence_intervals_repeats=num_auc_repeats,
                                             predictor_class=predictor_class, **predictor_kwargs)
                pd.Series(ret).to_csv(os.path.join(figures_dir, 'best_results_summary.csv'))
