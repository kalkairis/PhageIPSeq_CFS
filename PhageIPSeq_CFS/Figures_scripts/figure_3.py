import os
import string

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from PhageIPSeq_CFS.ComparePopulations.comparing_metadata import get_blood_test_name, \
    metadata_distribution_figure_single_blood_test
from PhageIPSeq_CFS.Predictions.classifiers import create_auc_with_bootstrap_figure, get_x_y
from PhageIPSeq_CFS.config import visualizations_dir, predictor_class, predictor_kwargs, num_auc_repeats
from PhageIPSeq_CFS.helpers import get_individuals_metadata_df, get_outcome, get_imputed_individuals_metadata


def metadata_distribution_sub_figure(spec, fig):
    # noinspection PyTypeChecker
    metadata = pd.merge(get_individuals_metadata_df().drop(columns='catrecruit_Binary'),
                        get_outcome(return_type=bool).apply(lambda x: 'Sick' if x else 'Healthy'),
                        left_index=True,
                        right_index=True).set_index('is_CFS', append=True)
    metadata.columns = list(map(get_blood_test_name, metadata.columns))
    stacked_metadata = metadata.stack().reset_index(level=2).rename(
        columns={'level_2': 'Blood Test', 0: 'value'}).reset_index()
    blood_tests = metadata.columns
    metadata = metadata.reset_index(level=1)
    internal_spec_for_legend = spec.subgridspec(2, 1, height_ratios=[15, 1], hspace=0.2)
    ax = fig.add_subplot(internal_spec_for_legend[0, :])
    ax.set_axis_off()
    ax.text(-0.02, 1.1, string.ascii_uppercase[0], transform=ax.transAxes, size=20, weight='bold')
    internal_spec = internal_spec_for_legend[0, :].subgridspec(2, len(blood_tests) // 2, wspace=1.2, hspace=0.6)
    for blood_test, single_internal_spec in zip(blood_tests, internal_spec):
        ax = fig.add_subplot(single_internal_spec)
        metadata_distribution_figure_single_blood_test(ax, blood_test, metadata)
    legend_spec = internal_spec_for_legend[1, :]
    ax = fig.add_subplot(legend_spec)
    ax.set_axis_off()
    ax.legend(
        handles=[mpatches.Patch(facecolor=sns.color_palette()[0], label='Sick', edgecolor='black'),
                 mpatches.Patch(facecolor=sns.color_palette()[1], label='Healthy', edgecolor='black')],
        loc='center', ncol=2)


if __name__ == "__main__":
    figures_dir = os.path.join(visualizations_dir, 'figure_3')
    os.makedirs(figures_dir, exist_ok=True)
    fig = plt.figure(figsize=(25, 10))
    spec = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.1)
    top_spec = spec[0, :]
    metadata_distribution_sub_figure(top_spec, fig)
    internal_spec = spec[1, :].subgridspec(1, 3, wspace=0.2)

    # Add prediction only from blood tests sub-figure
    x = get_imputed_individuals_metadata()
    y = get_outcome()
    prediction_results = pd.read_csv(os.path.join(figures_dir, 'blood_tests_only_results.csv'), index_col=0)
    ax = fig.add_subplot(internal_spec[0])
    create_auc_with_bootstrap_figure(num_confidence_intervals_repeats=num_auc_repeats, x=x, y=y,
                                     predictor_class=predictor_class, ax=ax,
                                     prediction_results=prediction_results, **predictor_kwargs)
    ax.set_title('Blood tests alone')
    ax.text(-0.1, 1.1, string.ascii_uppercase[1], transform=ax.transAxes, size=20, weight='bold')

    # Add predictions from blood tests and flagella
    ax = fig.add_subplot(internal_spec[1])
    df = pd.read_csv(
        os.path.join(visualizations_dir, 'Predictions_with_bloodtests_with_oligos', 'best_results_summary.csv'),
        index_col=[0, 1], header=[0, 1])
    best_flagella_params = \
        df.stack(level=0).reset_index().set_index('oligos_subgroup').loc['is_bac_flagella'].reset_index(
            drop=True).sort_values(by='auc', ascending=False)[['data_type', 'bottom_threshold']].iloc[0].to_dict()
    best_flagella_params['bottom_threshold'] = best_flagella_params['bottom_threshold'] / 100.0
    x, y = get_x_y(oligos_subgroup='is_bac_flagella', with_bloodtests=True, **best_flagella_params)
    prediction_results = pd.read_csv(os.path.join(figures_dir, 'blood_tests_with_flagella_results.csv'), index_col=0)
    create_auc_with_bootstrap_figure(num_auc_repeats, x, y, predictor_class,
                                     ax=ax, prediction_results=prediction_results, **predictor_kwargs)
    ax.set_title('Blood tests and flagella')
    ax.text(-0.1, 1.1, string.ascii_uppercase[2], transform=ax.transAxes, size=20, weight='bold')

    # Summary of results
    ax = fig.add_subplot(internal_spec[2])
    best_auc_df = df.stack(level=0).reset_index().head().sort_values(by='auc', ascending=False).groupby(
        'oligos_subgroup').first().reset_index()
    best_auc_df['oligos_subgroup'] = best_auc_df['oligos_subgroup'].apply(
        lambda subgroup: ' '.join(subgroup.split('_'))).apply(
        lambda subgroup: subgroup[3:] if subgroup.startswith('is ') else subgroup)
    best_auc_df['auc_ci'] = best_auc_df.apply(
        lambda row: np.array(stats.norm(0, row['auc_std']).interval(0.95)), axis=1)
    order = best_auc_df['oligos_subgroup'].values
    sns.barplot(data=best_auc_df, x='oligos_subgroup', y='auc', ci=None, ax=ax, color=sns.color_palette()[0],
                order=order)
    params = {'x': order, 'y': best_auc_df.set_index('oligos_subgroup').loc[order]['auc'].values,
              'yerr': (best_auc_df.set_index('oligos_subgroup').loc[order]['auc_std'] * stats.norm.ppf(0.975)).values,
              'fmt': 'none', 'ecolor': 'black'}
    ax.errorbar(**params)
    ax.set_title('Prediction results by subgroup')
    ax.text(-0.1, 1.1, string.ascii_uppercase[3], transform=ax.transAxes, size=20, weight='bold')
    plt.savefig(os.path.join(figures_dir, 'figure_3.png'))
