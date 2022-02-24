import os
import string

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind_from_stats
from statannot import add_stat_annotation

from PhageIPSeq_CFS.config import oligo_families, visualizations_dir, predictions_outcome_dir, predictors_info, \
    oligos_group_to_name
from PhageIPSeq_CFS.helpers import get_data_with_outcome, get_oligos_metadata


def create_single_oligo_group_vs_all_distribution(df, ax, group_name, cfs_label, healthy_label):
    df[group_name] = get_oligos_metadata()[group_name]
    df.sort_values(by=group_name, inplace=True)
    sns.lineplot(data=df, x=cfs_label,
                 y=healthy_label, hue=group_name,
                 palette=['grey', sns.color_palette("Set2")[4]],
                 ax=ax, err_style='bars')
    legend_handels, legend_labels = ax.get_legend_handles_labels()
    legend_handels = [legend_handels[legend_labels.index('True')]]
    legend_labels = [' '.join(group_name.split('_')[1:])]
    ax.legend(handles=legend_handels, labels=legend_labels)


def create_supp_figure_oligo_group_vs_all_distribution(output_dir, overwrite=False):
    fig = plt.figure(figsize=(20, 10))
    spec = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    df = get_data_with_outcome(data_type='exist')
    cfs_label = 'Number of CFS patients in whom a\npeptite is significantly bound'
    healthy_label = 'Number of healthy patients in whom\na peptite is significantly bound'
    df['disease_status'] = df.reset_index()['is_CFS'].astype(bool).apply(
        lambda x: cfs_label if x else healthy_label).values
    df = df.groupby('disease_status').sum().T
    for i, oligo_group in enumerate(oligo_families):
        ax = fig.add_subplot(spec[i // 3, i % 3])
        create_single_oligo_group_vs_all_distribution(df, ax, oligo_group, cfs_label, healthy_label)
        ax.text(-0.1, 1.1, string.ascii_lowercase[i], transform=ax.transAxes, size=20, weight='bold')
    if overwrite:
        plt.savefig(os.path.join(output_dir, 'supp_figure_1.png'))
    else:
        plt.show()
    plt.close()


def create_supp_figure_adding_age_and_gender_to_predictions(output_dir, estimator, overwrite=False):
    predictions_with_age_and_gender = pd.read_csv(os.path.join(predictions_outcome_dir, "_".join(
        ['predictions_summary', estimator, 'oligos_age_and_gender.csv'])), index_col=[0, 1], header=[0, 1]).stack(0)[
        ['auc', 'std']].reset_index('subgroup').sort_values(by='auc').groupby('subgroup').last()
    predictions_only_oligos = pd.read_csv(
        os.path.join(predictions_outcome_dir, '_'.join(['predictions', 'summary', estimator, 'with', 'oligos.csv'])),
        index_col=[0, 1], header=[0, 1]).stack(0)[['auc', 'std']].reset_index('subgroup').sort_values(by='auc').groupby(
        'subgroup').last()
    df = pd.merge(predictions_with_age_and_gender, predictions_only_oligos, left_index=True, right_index=True,
                  suffixes=('_with_age_gender', '_without_age_gender')).rename(index=oligos_group_to_name)
    df_statistics = df.copy()
    df_statistics['t_statistic'], df_statistics['p_value'] = pd.DataFrame(
        df.apply(lambda row: dict(zip(['t_statistic', 'p_value'],
                                      ttest_ind_from_stats(
                                          row['auc_with_age_gender'],
                                          row['std_with_age_gender'], 80,
                                          row['auc_without_age_gender'],
                                          row['std_without_age_gender'],
                                          80))), axis=1).to_dict()).values
    predictions_with_age_and_gender['Prediction input'] = 'with age and gender'
    predictions_only_oligos['Prediction input'] = 'without age and gender'
    df = pd.concat([predictions_with_age_and_gender, predictions_only_oligos], axis=0).rename(
        index=oligos_group_to_name).reset_index()
    ax = sns.barplot(data=df, x='subgroup', hue='Prediction input', y='auc')
    ax.set_ylim(0.4, 0.7 if estimator == 'xgboost' else 0.8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    add_stat_annotation(data=df, x='subgroup', hue='Prediction input', y='auc', ax=ax,
                        box_pairs=[[('Complete library', 'with age and gender'),
                                    ('Complete library', 'without age and gender')],
                                   [('IEDB/controls', 'with age and gender'),
                                    ('IEDB/controls', 'without age and gender')],
                                   [('Antibody-coated\nstrains', 'with age and gender'),
                                    ('Antibody-coated\nstrains', 'without age and gender')],
                                   [('Metagenomics\nantigens', 'with age and gender'),
                                    ('Metagenomics\nantigens', 'without age and gender')],
                                   [('Flagellins', 'with age and gender'),
                                    ('Flagellins', 'without age and gender')],
                                   [('Pathogenic strains', 'with age and gender'),
                                    ('Pathogenic strains', 'without age and gender')],
                                   [('Probiotic strains', 'with age and gender'),
                                    ('Probiotic strains', 'without age and gender')]],
                        test='Mann-Whitney'
                        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'supp_fig_7_{estimator}.png'))
    plt.close()


if __name__ == "__main__":
    supp_figures_dir = os.path.join(visualizations_dir, 'supp_figures')
    os.makedirs(supp_figures_dir, exist_ok=True)
    overwrite = True
    create_supp_figure_oligo_group_vs_all_distribution(supp_figures_dir, overwrite=overwrite)
    for estimator in predictors_info.keys():
        create_supp_figure_adding_age_and_gender_to_predictions(supp_figures_dir, estimator, overwrite)
