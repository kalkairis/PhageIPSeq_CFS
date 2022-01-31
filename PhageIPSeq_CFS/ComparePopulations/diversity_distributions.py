import os
import string

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from statannot import add_stat_annotation

from PhageIPSeq_CFS.config import oligo_families, visualizations_dir
from PhageIPSeq_CFS.helpers import get_oligos_metadata_subgroup_with_outcome, get_oligos_metadata

if __name__ == "__main__":
    ncols = 4
    nrows = 2
    oligo_order = ['IEDB or cntrl', 'PNP', 'patho', 'probio', 'IgA', 'bac flagella']
    figure_1, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))
    axes_flat = axes.flatten()
    i = 0
    figures_dir = os.path.join(visualizations_dir, 'oligo_distributions')
    os.makedirs(figures_dir, exist_ok=True)
    df = get_oligos_metadata_subgroup_with_outcome(data_type='exist', subgroup='all').fillna(0).reset_index(level=0)
    oligos_metadata = get_oligos_metadata()[oligo_families]
    oligos_metadata.rename(columns={col: ' '.join(col.split('_')[1:]) for col in oligos_metadata.columns}, inplace=True)
    mean_df = df.groupby('is_CFS').mean().transpose().rename(
        columns={i: 'Sick' if bool(i) else 'Healthy' for i in [0, 1]})
    minimal_value = sorted(set(mean_df[['Sick', 'Healthy']].values.ravel()))[1]
    mean_df['ratio'] = mean_df.apply(lambda row: row['Sick'] / max(row['Healthy'], minimal_value), axis=1)
    mean_df = mean_df.merge(oligos_metadata, left_index=True, right_index=True)
    exist_df = df.groupby('is_CFS').max().transpose().rename(
        columns={i: 'Sick' if bool(i) else 'Healthy' for i in [0, 1]}).merge(oligos_metadata, left_index=True,
                                                                             right_index=True)
    stacked_ratio = pd.DataFrame(columns=['Sick', 'Healthy', 'ratio', 'Oligo Family'])
    for col in oligos_metadata.columns:
        tmp = mean_df.loc[mean_df[col]][['Sick', 'Healthy', 'ratio']]
        tmp['Oligo Family'] = col
        stacked_ratio = pd.concat([stacked_ratio, tmp])
    ax = axes_flat[i]
    sns.boxplot(data=stacked_ratio.loc[~stacked_ratio[['Sick', 'Healthy']].eq(0).any(axis=1)], x='Oligo Family',
                y='ratio', ax=ax, order=oligo_order)
    ax.xaxis.set_tick_params(rotation=45)
    test_results = add_stat_annotation(ax,
                                       data=stacked_ratio,
                                       x='Oligo Family', y='ratio',
                                       order=oligo_order,
                                       box_pairs=[('PNP', 'bac flagella'),
                                                  ('IgA', 'bac flagella'),
                                                  ('bac flagella', 'patho'),
                                                  ('bac flagella', 'probio'),
                                                  ('IEDB or cntrl', 'PNP')],
                                       test='Mann-Whitney', text_format='star', comparisons_correction='bonferroni',
                                       loc='inside', verbose=False, )
    i += 1
    statistics_results = {}
    for oligo_family in oligo_order:
        ax = axes_flat[i]
        if i % ncols == 0:
            ax.set_axis_off()
            table_ax = ax
            i += 1
            ax = axes_flat[i]

        i += 1
        sns.lineplot(data=mean_df, y='Healthy', x='Sick', hue=oligo_family, err_style="bars", ax=ax)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.3)
        ax.get_legend().remove()
        ax.set_title(oligo_family)
        # plt.savefig(os.path.join(figures_dir, oligo_family + '.png'))
        # plt.close()
        contingency_table_observed = exist_df.groupby(oligo_family)[['Healthy', 'Sick']].sum()
        contingency_table_expected = pd.DataFrame({'Sick': contingency_table_observed.sum(axis=1).div(2),
                                                   'Healthy': contingency_table_observed.sum(axis=1).div(2)})
        single = {}
        oligo_family_mean_df = mean_df.loc[mean_df[oligo_family]].copy()
        oligo_family_mean_df_reference = mean_df.loc[~mean_df[oligo_family]]
        single['chi_squared'], single['chi_squared_p_value'] = stats.chisquare(
            contingency_table_observed.values.ravel(),
            contingency_table_expected.values.ravel())
        single['ttest'], single['ttest_p_value'] = stats.ttest_ind(
            oligo_family_mean_df['ratio'],
            oligo_family_mean_df_reference['ratio'])
        # single['rank_sums_oligo_subgroup'], single['rank_sums_oligo_subgroup_p_value'] = stats.ranksums(
        #     oligo_family_mean_df['Healthy'], oligo_family_mean_df['Sick'])
        # single['rank_sums_oligo_reference'], single['rank_sums_oligo_reference_p_value'] = stats.ranksums(
        #     oligo_family_mean_df_reference['Healthy'], oligo_family_mean_df_reference['Sick'])
        single['rank_sums'], single['rank_sums_p_value'] = stats.ranksums(
            oligo_family_mean_df['ratio'],
            oligo_family_mean_df_reference['ratio'])
        statistics_results[oligo_family] = single
    n = 0
    for col in range(ncols):
        for row in range(nrows):
            ax = axes[row][col]
            ax.text(-0.1, 1.1, string.ascii_lowercase[n], transform=ax.transAxes, size=20, weight='bold')
            n += 1

    res = pd.DataFrame(statistics_results).transpose()
    table_for_figure = res.rename(columns={col: ' '.join(col.split('_')) for col in res.columns})[
        ['rank sums', 'rank sums p value']].rename(columns={'rank sums': 'Ranksums', 'rank sums p value': 'p-value'})
    table_for_figure['Ranksums'] = table_for_figure['Ranksums'].round(1).astype(str)
    table_for_figure['p-value'] = table_for_figure['p-value'].apply(lambda x: format(x, '.1g'))
    table_ax.table(table_for_figure.values,
                   rowLabels=list(map(lambda x: x.replace(' ', '\n', 1), table_for_figure.index.values)),
                   colLabels=table_for_figure.columns,
                   loc='center', cellLoc='center', fontsize=10, bbox=[0.2, 0, 0.8, 1], colWidths=[0.4, 0.4])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'Figure_1.png'))
    res.to_csv(os.path.join(figures_dir, 'statistics.csv'))
