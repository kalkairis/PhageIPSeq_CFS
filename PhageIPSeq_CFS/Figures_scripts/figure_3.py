import os
import string

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from scipy import stats

from PhageIPSeq_CFS.config import visualizations_dir, predictors_info, predictions_outcome_dir, oligo_families_colors, \
    oligo_order, oligos_group_to_name
from PhageIPSeq_CFS.helpers import get_outcome, get_x_y, get_oligos_metadata, \
    create_auc_with_bootstrap_figure

if __name__ == "__main__":
    figures_dir = os.path.join(visualizations_dir, 'figure_3')
    os.makedirs(figures_dir, exist_ok=True)

    for estimator_name, estimator_info in predictors_info.items():
        predictor_kwargs = estimator_info['predictor_kwargs']
        fig = plt.figure(figsize=(10, 10))
        spec = fig.add_gridspec(2, 2, hspace=0.8, wspace=0.3)
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
        prediction_results = prediction_results.set_index('with_oligos').loc[True]
        prediction_results = prediction_results.set_index('with_bloodtests').loc[False]
        prediction_results = prediction_results.set_index('subgroup').loc['is_PNP']
        for k, v in best_prediction_params['is_PNP'].items():
            prediction_results = prediction_results.set_index(k).loc[v]
        prediction_results = prediction_results.set_index('sample_id').rename(columns={'0': 'predict_proba'})
        prediction_results['y'] = get_outcome()
        create_auc_with_bootstrap_figure(num_auc_repeats, None, None, estimator_info['predictor_class'],
                                         color=oligo_families_colors['Metagenomics\nantigens'], chance_color='black',
                                         ax=ax, prediction_results=prediction_results,
                                         predictor_name='Metagenomics antigens\n', **predictor_kwargs)
        ax.text(-0.1, 1.1, string.ascii_uppercase[0], transform=ax.transAxes, size=20, weight='bold')

        # Sub-figure b
        ax = fig.add_subplot(spec[0, 1])
        df = pd.read_csv(
            os.path.join(predictions_outcome_dir, f'predictions_summary_{estimator_name}_with_oligos.csv'),
            index_col=[0, 1], header=[0, 1])
        best_auc_df = df.stack(level=0).reset_index().sort_values(by='auc', ascending=False).groupby(
            'subgroup').first().reset_index()
        best_auc_df['subgroup'] = best_auc_df['subgroup'].apply(lambda subgroup: oligos_group_to_name[subgroup])
        best_auc_df['auc_ci'] = best_auc_df.apply(
            lambda row: np.array(stats.norm(0, row['std']).interval(0.95)), axis=1)
        sns.barplot(data=best_auc_df, x='subgroup', y='auc', ci=None, ax=ax, palette=sns.color_palette(),
                    order=oligo_order)
        params = {'x': oligo_order, 'y': best_auc_df.set_index('subgroup').loc[oligo_order]['auc'].values,
                  'yerr': (best_auc_df.set_index('subgroup').loc[oligo_order]['std'] * stats.norm.ppf(
                      0.95)).values,
                  'fmt': 'none', 'ecolor': 'black', 'capsize': 10}
        ax.errorbar(**params)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(bottom=0.4)
        ax.set_ylabel('AUC of predictions\nbased on Ig epitope repertoire')
        ax.xaxis.label.set_visible(False)
        ax.text(-0.1, 1.1, string.ascii_uppercase[1], transform=ax.transAxes, size=20, weight='bold')

        # Sub-figure c
        ax = fig.add_subplot(spec[1, 0])
        ax.set_axis_off()
        ax.text(-0.1, 1.1, string.ascii_uppercase[2], transform=ax.transAxes, size=20, weight='bold')
        internal_spec = spec[1, 0].subgridspec(1, 2, width_ratios=[1, 6])
        ax = fig.add_subplot(internal_spec[0])
        ax.set_axis_off()
        ax = fig.add_subplot(internal_spec[1])
        x, y = get_x_y(oligos_subgroup='is_PNP', with_bloodtests=False, imputed=True,
                       data_type=best_prediction_params['is_PNP']['data_type'],
                       bottom_threshold=best_prediction_params['is_PNP']['threshold_percent'] / 100)
        predictor = estimator_info['predictor_class'](**predictor_kwargs)
        model = predictor.fit(x, y)
        explainer = shap.TreeExplainer(model, x)
        shap_values_ebm = explainer(x)

        # Save shap importance table
        shap_df = pd.DataFrame(shap_values_ebm.values, columns=x.columns)
        vals = np.abs(shap_df.values).mean(0)
        shap_importance = pd.DataFrame(list(zip(x.columns, vals)), columns=['col_name', 'feature_importance_vals'])
        shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        shap_importance.set_index('col_name', inplace=True)
        shap_importance['oligo full name'] = get_oligos_metadata()['full name'].loc[shap_importance.index]
        shap_importance.to_csv(os.path.join(figures_dir, f"shap_values_{estimator_name}.csv"))

        shap.plots.beeswarm(shap_values_ebm, max_display=15, show=False, plot_size=None, sum_bottom_features=False)
        ax.set_yticklabels(list(
            map(lambda ticklabel: plt.Text(*ticklabel.get_position(), ticklabel.get_text().split("_")[1]),
                ax.get_yticklabels())))
        ax.set_xlabel("SHAP value (PNP model)")

        # Sub-figure d
        ax = fig.add_subplot(spec[1, 1])
        ax.set_axis_off()
        ax.text(-0.1, 1.1, string.ascii_uppercase[3], transform=ax.transAxes, size=20, weight='bold')

        plt.savefig(os.path.join(figures_dir, f'figure_3_{estimator_name}.png'))
