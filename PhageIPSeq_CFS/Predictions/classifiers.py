from sklearn.metrics import roc_curve, auc
from numpy import interp
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

from PhageIPSeq_CFS.config import repository_data_dir, visualizations_dir, logs_path, oligo_families
from PhageIPSeq_CFS.helpers import get_oligos_with_outcome, split_xy_df_and_filter_by_threshold, \
    get_oligos_metadata_subgroup_with_outcome


def run_leave_one_out_prediction(x, y, predictor_class, *predictor_args, **predictor_kwargs):
    ret = {}
    if x.empty:
        return dict()
    for sample in y.index.values:
        train_x = x.drop(index=sample)
        train_y = y.drop(index=sample)
        predictor = predictor_class(*predictor_args, **predictor_kwargs)
        predictor.fit(train_x, train_y.values.ravel())
        y_hat = predictor.predict(x.loc[sample].values.reshape(1, -1))[0]
        if isinstance(predictor, RidgeClassifier):
            predict_proba = predictor._predict_proba_lr(x.loc[sample].values.reshape(1, -1))[0][1]
        else:
            predict_proba = predictor.predict_proba(x.loc[sample].values.reshape(1, -1))[0][1]
        ret[sample] = {'y': y.loc[sample].values[0], 'y_hat': y_hat, 'predict_proba': predict_proba}
    return pd.DataFrame(ret).transpose()


def get_prediction_results(output_dir, x, y, predictor_class, *predictor_args, **predictor_kwargs):
    prediction_results = run_leave_one_out_prediction(x, y, predictor_class, *predictor_args, **predictor_kwargs)
    if len(prediction_results) == 0:
        return {'auc': None, 'pearson_r': None, 'pearson_p_value': None}

    os.makedirs(output_dir, exist_ok=True)
    # create scatter of true and predicted values
    pearson_r, pearson_p_value = pearsonr(prediction_results['y'], prediction_results['predict_proba'])
    ax = sns.histplot(data=prediction_results, x='predict_proba', hue='y', element='step')
    ax.set_title(f'r={round(pearson_r, 2)}, p-value={pearson_p_value:.1g}')
    plt.savefig(os.path.join(output_dir, 'predicted_true_scatter.png'))
    plt.close()

    # create roc curve
    prediction_results.sort_values(by='predict_proba', inplace=True, ascending=False)
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
    ax = plt.subplot()
    ax.plot(fprs, tprs, color='b', label=f"Predictor (AUC={round(auc_value, 3)})")
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'prediction_auc.png'))
    plt.close()
    return {'auc': auc_value, 'pearson_r': pearson_r, 'pearson_p_value': pearson_p_value}


def get_x_y(bottom_threshold, data_type, oligos_subgroup):
    if oligos_subgroup == 'all':
        x, y = split_xy_df_and_filter_by_threshold(get_oligos_with_outcome(data_type=data_type),
                                                   bottom_threshold=bottom_threshold)

    else:
        x, y = split_xy_df_and_filter_by_threshold(
            get_oligos_metadata_subgroup_with_outcome(data_type=data_type, subgroup=oligos_subgroup),
            bottom_threshold=bottom_threshold)
    return x, y


def get_prediction_parameters(data_type, threshold_percent, figures_dir, oligos_subgroup='all'):
    bottom_threshold = threshold_percent / 100
    x, y = get_x_y(bottom_threshold, data_type, oligos_subgroup)
    output_dir = os.path.join(figures_dir, data_type, str(bottom_threshold))
    return get_prediction_results(output_dir, x, y, RidgeClassifier)


def get_top_prediction_shap_values(data_type, bottom_threshold, figures_dir, oligos_subgroup, predictor_class,
                                   *predictor_args,
                                   **predictor_kwargs):
    out_dir = os.path.join(figures_dir, 'best_run_results')
    os.makedirs(out_dir, exist_ok=True)
    bottom_threshold = bottom_threshold / 100
    x, y = get_x_y(bottom_threshold, data_type, oligos_subgroup)
    predictor = predictor_class(*predictor_args, **predictor_kwargs)
    model = predictor.fit(x, y)

    explainer = shap.LinearExplainer(model, x)
    shap_values_ebm = explainer(x, silent=True)

    # the waterfall_plot shows how we get from explainer.expected_value to model.predict(X)[sample_ind]
    ax = plt.subplot()
    shap.plots.beeswarm(shap_values_ebm, max_display=14, show=False)
    ax.set_title(f"Shap for {data_type} threshold={bottom_threshold}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_beeswarm.png'))
    plt.close()

    pd.Series(index=x.columns, data=predictor.coef_[0]).sort_values(ascending=False).to_csv(
        os.path.join(out_dir, 'feature_coefs.csv'))


def predict_and_run_shape_on_oligo_subgroup(oligos_subgroup, figures_dir):
    curr_figures_dir = os.path.join(figures_dir, oligos_subgroup)
    with qp(f"preds") as q:
        q.startpermanentrun()
        waiton = {}
        for data_type in ['fold', 'exist']:
            for threshold_percent in range(0, 100, 5):
                waiton[(data_type, threshold_percent)] = q.method(get_prediction_parameters,
                                                                  (data_type, threshold_percent, curr_figures_dir,
                                                                   oligos_subgroup))
        all_results = {k: q.waitforresult(v) for k, v in waiton.items()}
    all_results = pd.DataFrame(all_results).transpose().reset_index().rename(
        columns={'level_0': 'data_type', 'level_1': 'bottom_threshold'})
    all_results.to_csv(os.path.join(curr_figures_dir, 'results_summary.csv'))
    all_results.dropna(inplace=True)
    best_params = all_results.sort_values(by='auc', ascending=False).iloc[0][
        ['data_type', 'bottom_threshold']].to_dict()
    get_top_prediction_shap_values(**best_params, predictor_class=RidgeClassifier, figures_dir=curr_figures_dir,
                                   oligos_subgroup=oligos_subgroup)


if __name__ == "__main__":
    figures_dir = os.path.join(visualizations_dir, 'Predictions', 'RidgeClassifier')
    all_results = {}
    sethandlers()
    os.chdir(logs_path)
    with fakeqp(f"wpreds") as q:
        q.startpermanentrun()
        waiton = []
        for oligos_subgroup in oligo_families + ['all']:
            waiton.append(q.method(predict_and_run_shape_on_oligo_subgroup, (oligos_subgroup, figures_dir)))
        q.wait(waiton, assertnoerrors=False)
    print("here")

    print("here")
