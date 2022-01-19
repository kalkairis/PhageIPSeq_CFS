import os
import string

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split

from PhageIPSeq_CFS.config import visualizations_dir, predictors_info, predictions_outcome_dir, oligo_families_colors, \
    oligo_order
from PhageIPSeq_CFS.helpers import get_outcome, compute_auc_from_prediction_results, get_x_y

if __name__ == "__main__":
    figures_dir = os.path.join(visualizations_dir, 'figure_3')
    os.makedirs(figures_dir, exist_ok=True)

    for estimator_name, estimator_info in predictors_info.items():
        predictor_kwargs = estimator_info['predictor_kwargs']
        fig = plt.figure(figsize=(25, 10))
        spec = fig.add_gridspec(2, 3, hspace=0.3)
        num_auc_repeats = 100

        # Sub-figure a
        ax = fig.add_subplot(spec[0, 0])
        best_prediction_params = pd.read_csv(
            os.path.join(predictions_outcome_dir, f'predictions_summary_{estimator_name}_with_oligos.csv'),
            index_col=[0, 1], header=[0, 1])
        best_prediction_params = best_prediction_params.stack(0)['auc'].reset_index().sort_values(by='auc',
                                                                                                  ascending=False).groupby(
            'subgroup').first().drop(columns='auc').T.to_dict()
        prediction_results = pd.read_csv(os.path.join(predictions_outcome_dir, f'{estimator_name}_predictions.csv'))
        prediction_results = prediction_results[prediction_results['with_oligos']]
        prediction_results = prediction_results[~prediction_results['with_bloodtests']]
        for subgroup, subgroup_params in best_prediction_params.items():
            if subgroup not in ['all', 'is_PNP', 'is_bac_flagella']:
                continue
            subgroup_predictions = prediction_results[prediction_results['subgroup'].eq(subgroup)]
            subgroup_predictions = subgroup_predictions[
                subgroup_predictions['data_type'].eq(subgroup_params['data_type'])]
            subgroup_predictions = subgroup_predictions[
                subgroup_predictions['threshold_percent'].eq(subgroup_params['threshold_percent'])]
            subgroup_predictions = subgroup_predictions.set_index('sample_id').rename(columns={'0': 'predict_proba'})[
                ['predict_proba']]
            subgroup_predictions['y'] = get_outcome()
            auc_value, fprs, tprs = compute_auc_from_prediction_results(subgroup_predictions, return_fprs_tprs=True)
            auc_std = np.std(
                [compute_auc_from_prediction_results(train_test_split(subgroup_predictions)[0], True)[0] for _ in
                 range(num_auc_repeats)])
            subgroup_name = ' '.join(subgroup.split('_')[1:]) if subgroup.startswith('is_') else subgroup
            ax.plot(fprs, tprs, color=oligo_families_colors[subgroup_name],
                    label=f"{subgroup_name} (AUC={round(auc_value, 3)}, std={round(auc_std, 3)})")
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',
                label='Chance', alpha=.8)
        ax.legend()
        ax.text(-0.1, 1.1, string.ascii_uppercase[0], transform=ax.transAxes, size=20, weight='bold')

        # Sub-figure b
        ax = fig.add_subplot(spec[0, 1])
        df = pd.read_csv(
            os.path.join(predictions_outcome_dir, f'predictions_summary_{estimator_name}_with_oligos.csv'),
            index_col=[0, 1], header=[0, 1])
        best_auc_df = df.stack(level=0).reset_index().sort_values(by='auc', ascending=False).groupby(
            'subgroup').first().reset_index()
        best_auc_df['subgroup'] = best_auc_df['subgroup'].apply(
            lambda subgroup: ' '.join(subgroup.split('_'))).apply(
            lambda subgroup: subgroup[3:] if subgroup.startswith('is ') else subgroup)
        best_auc_df['auc_ci'] = best_auc_df.apply(
            lambda row: np.array(stats.norm(0, row['std']).interval(0.95)), axis=1)
        sns.barplot(data=best_auc_df, x='subgroup', y='auc', ci=None, ax=ax, palette=sns.color_palette(),
                    order=oligo_order)
        params = {'x': oligo_order, 'y': best_auc_df.set_index('subgroup').loc[oligo_order]['auc'].values,
                  'yerr': (best_auc_df.set_index('subgroup').loc[oligo_order]['std'] * stats.norm.ppf(
                      0.95)).values,
                  'fmt': 'none', 'ecolor': 'black', 'capsize': 10}
        ax.errorbar(**params)
        ax.set_ylim(bottom=0.4)
        ax.text(-0.1, 1.1, string.ascii_uppercase[1], transform=ax.transAxes, size=20, weight='bold')

        # Sub-figure c
        ax = fig.add_subplot(spec[0, 2])
        ax.set_axis_off()
        ax.text(-0.1, 1.1, string.ascii_uppercase[2], transform=ax.transAxes, size=20, weight='bold')
        internal_spec = spec[0, 2].subgridspec(1, 2, width_ratios=[1, 6])
        ax = fig.add_subplot(internal_spec[0])
        ax.set_axis_off()
        ax = fig.add_subplot(internal_spec[1])
        x, y = get_x_y(oligos_subgroup='is_PNP', with_bloodtests=False, imputed=True,
                       data_type= best_prediction_params['is_PNP']['data_type'],
                       bottom_threshold=best_prediction_params['is_PNP']['threshold_percent']/100)
        predictor = estimator_info['predictor_class'](**predictor_kwargs)
        model = predictor.fit(x, y)
        explainer = shap.TreeExplainer(model, x)
        shap_values_ebm = explainer(x)
        shap.plots.beeswarm(shap_values_ebm, max_display=15, show=False, plot_size=None, sum_bottom_features=False)
        ax.set_yticklabels(list(
            map(lambda ticklabel: plt.Text(*ticklabel.get_position(), ticklabel.get_text().split("_")[1]),
                ax.get_yticklabels())))
        ax.set_xlabel("SHAP value (PNP model)")

        # Sub-figure d
        ax = fig.add_subplot(spec[1, 0])
        ax.set_axis_off()
        ax.text(-0.1, 1.1, string.ascii_uppercase[3], transform=ax.transAxes, size=20, weight='bold')

        # Sub-figure e
        ax = fig.add_subplot(spec[1, 1])
        ax.set_axis_off()
        ax.text(-0.1, 1.1, string.ascii_uppercase[4], transform=ax.transAxes, size=20, weight='bold')

        # Sub-figure f
        ax = fig.add_subplot(spec[1, 2])
        ax.set_axis_off()
        ax.text(-0.1, 1.1, string.ascii_uppercase[5], transform=ax.transAxes, size=20, weight='bold')

        plt.savefig(os.path.join(figures_dir, f'figure_3_{estimator_name}.png'))
