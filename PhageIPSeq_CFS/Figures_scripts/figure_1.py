import os
import string

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from skbio.diversity.alpha import shannon
from statannot import add_stat_annotation
from statsmodels.stats.multitest import multipletests
import matplotlib.image as mpimg


from PhageIPSeq_CFS.config import visualizations_dir, oligo_families, oligo_families_colors, oligos_group_to_name
from PhageIPSeq_CFS.helpers import get_data_with_outcome, get_oligos_metadata, split_xy_df_and_filter_by_threshold, \
    get_individuals_metadata_df

def get_ratio_between_groups(group_name):
    df = get_data_with_outcome('exist')
    df = df.reset_index(level=0).groupby('is_CFS').mean().mul(100).T
    df = df[df.min(axis=1).gt(0)]
    df['in_group'] = get_oligos_metadata()[group_name]
    df['ratio'] = df[1] / df[0]
    df.set_index('in_group', inplace=True)
    stat, p_val = stats.ttest_ind(df.loc[True]['ratio'], df.loc[False]['ratio'])
    return stat, p_val

def create_figure_1(overwrite=True):
    figures_dir = os.path.join(visualizations_dir, 'figure_1')
    os.makedirs(figures_dir, exist_ok=True)
    fig = plt.figure(figsize=(20, 10), constrained_layout=False)
    spec = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)#, top=0.2, bottom=0.5)

    # sub-figure a - Thomas will add manually the overview of the system
    ax = fig.add_subplot(spec[0, 0])
    ax.set_axis_off()
    ax.text(-0.1, 1.1, string.ascii_lowercase[0], transform=ax.transAxes, size=20, weight='bold')
    bounds = list(ax.get_position().bounds)
    bounds[0] -= 0.03
    bounds[1] -= 0.02
    bounds[3] *= 1.1
    bounds[2] *= 1.2
    ax = fig.add_axes(bounds)
    ax.set_axis_off()
    img = mpimg.imread(os.path.join(figures_dir, "Fig.1a.png"))
    ax.imshow(img, origin='upper')

    # sub-figure b - cohort info
    cohort_info = get_individuals_metadata_df()[['catrecruit_Binary', 'sex_Binary', 'agegroup_Average']]
    cohort_info['Sex'] = cohort_info['sex_Binary'].apply(lambda s: 'Male' if bool(s) else 'Female')
    ax = fig.add_subplot(spec[0, 1])
    ax.text(-0.1, 1.1, string.ascii_lowercase[1], transform=ax.transAxes, size=20, weight='bold')
    ax.set_axis_off()
    internal_spec = spec[0, 1].subgridspec(2, 1, height_ratios=[1, 8], hspace=0)
    # adding empty space for table on cohort info
    ax = fig.add_subplot(internal_spec[0])
    ax.set_axis_off()
    bounds = list(ax.get_position().bounds)
    bounds[1] -= 0.04
    bounds[3] *= 3
    ax = fig.add_axes(bounds)
    ax.set_axis_off()
    img = mpimg.imread(os.path.join(figures_dir, "Fig.1b.png"))
    ax.imshow(img, origin='upper')
    internal_figure_spec = internal_spec[1].subgridspec(2, 2, wspace=0, hspace=0, height_ratios=[1, 3])
    ax = fig.add_subplot(internal_figure_spec[1, 0])
    sns.histplot(data=cohort_info[cohort_info['catrecruit_Binary'].eq(1)],
                 y='agegroup_Average', hue='Sex', hue_order=['Female', 'Male'],
                 palette=[sns.color_palette("colorblind")[-4], sns.color_palette("colorblind")[-1]],
                 multiple="stack", bins=4, ax=ax, legend=False)
    ax.set_ylabel('Age group average')
    ax.set_xlabel('ME/CFS')
    ax.set_xlim(*list(ax.get_xlim())[::-1])
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
    ax = fig.add_subplot(internal_figure_spec[1, 1], sharey=ax)
    ax.get_yaxis().set_visible(False)
    ax = sns.histplot(data=cohort_info[cohort_info['catrecruit_Binary'].eq(0)], y='agegroup_Average',
                      bins=4, hue_order=['Female', 'Male'],
                      palette=[sns.color_palette("colorblind")[-4], sns.color_palette("colorblind")[-1]],
                      hue='Sex',
                      multiple="stack", legend=False,
                      ax=ax)

    ax.set_xlabel('Healthy controls')
    ax = fig.add_subplot(internal_figure_spec[0, :])
    ax.set_axis_off()
    ax.legend(
        handles=[mpatches.Patch(facecolor=sns.color_palette("colorblind")[-1], label='Male', edgecolor='black'),
                 mpatches.Patch(facecolor=sns.color_palette("colorblind")[-4], label='Female', edgecolor='black')],
        loc='lower center', ncol=2)

    # sub-figure c
    ax = fig.add_subplot(spec[0, 2])
    ax.set_axis_off()
    ax.text(-0.1, 1.1, string.ascii_lowercase[2], transform=ax.transAxes, size=20, weight='bold')
    df = get_data_with_outcome(data_type='exist').sum(axis=1).reset_index(0).rename(columns={0: 'num_oligos'})
    df['status'] = df['is_CFS'].astype(bool).apply(lambda x: 'ME/CFS' if x else 'Healthy\ncontrols')

    df2 = get_data_with_outcome().fillna(1)
    df2['shannon_div'] = df2.apply(lambda row: shannon(row), axis=1)
    df2['status'] = df2.reset_index()['is_CFS'].astype(bool).apply(lambda x: 'ME/CFS' if x else 'Healthy\ncontrols').values

    internal_figure_spec = spec[0, 2].subgridspec(1, 2, wspace=0.6)
    ax = fig.add_subplot(internal_figure_spec[0])
    sns.boxplot(data=df, x='status', y='num_oligos', palette=sns.color_palette("muted")[-2:][::-1], ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('Number of significantly bound\npeptides per individual')
    add_stat_annotation(data=df, x='status', y='num_oligos', ax=ax,
                        box_pairs=[['ME/CFS', 'Healthy\ncontrols']], test='Mann-Whitney'
                        )

    ax = fig.add_subplot(internal_figure_spec[1])
    ax = sns.boxplot(data=df2, x='status', y='shannon_div', palette=sns.color_palette()[-2:][::-1], ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(r'Shannon $\alpha$-diversity' +' of\nantibody epitope repertoires')
    add_stat_annotation(data=df2, x='status', y='shannon_div', ax=ax,
                        box_pairs=[['ME/CFS', 'Healthy\ncontrols']], test='Mann-Whitney'
                        )

    # sub-figure d - scatter plot of expression
    ax = fig.add_subplot(spec[1, 0])
    df = get_data_with_outcome(data_type='exist')
    cfs_label = 'Percent of ME/CFS patients in whom a\npeptide is significantly bound'
    healthy_label = 'Percent of healthy control patients in\nwhom a peptide is significantly bound'
    df['disease_status'] = df.reset_index()['is_CFS'].astype(bool).apply(
        lambda x: cfs_label if x else healthy_label).values
    df = df.groupby('disease_status').mean().mul(100).T
    df['is_flagellin'] = get_oligos_metadata()['is_bac_flagella']
    df.sort_values(by='is_flagellin', inplace=True)
    sns.scatterplot(data=df, x=cfs_label,
                    y=healthy_label, hue='is_flagellin',
                    palette=['darkgrey', oligo_families_colors['Flagellins']],
                    ax=ax)
    legend_handels, legend_labels = ax.get_legend_handles_labels()
    legend_handels = [legend_handels[legend_labels.index('True')]]

    stat, p_val = get_ratio_between_groups('is_bac_flagella')
    legend_labels = [f'Flagellins']
    ax.text(80, 10, f"Flagellins ratio\noverrepresentation\np-value={p_val:.0e}", horizontalalignment='center')
    ax.legend(handles=legend_handels, labels=legend_labels)

    ax.text(-0.1, 1.1, string.ascii_lowercase[3], transform=ax.transAxes, size=20, weight='bold')

    # sub-figure e
    ax = fig.add_subplot(spec[1, 1])
    x, y = split_xy_df_and_filter_by_threshold(get_data_with_outcome(data_type='exist'))
    df = pd.concat([x, y], axis=1)
    df = df.groupby('is_CFS').sum()
    df = df.loc[:, df.ne(0).all()].copy()
    metadata = get_oligos_metadata()[oligo_families].rename(
        columns=oligos_group_to_name).loc[df.columns]
    metadata['Complete library'] = True
    metadata = metadata.stack()
    metadata = metadata[metadata].reset_index(level=-1).drop(columns=0).rename(columns={'level_1': 'Oligo family'})
    ratio_column = 'Ratios of antibody responses\nin ME/CFS and healthy controls'
    metadata[ratio_column] = df.apply(
        lambda oligo: oligo[int(True)] / oligo[int(False)])
    order = ['Metagenomics\nantigens', 'Pathogenic strains', 'Probiotic strains',
             'Antibody-coated\nstrains', 'Flagellins', 'IEDB/controls', 'Complete library']
    ax = sns.boxplot(data=metadata, x='Oligo family', y=ratio_column, ax=ax, order=order,
                     palette=list(map(lambda family: oligo_families_colors[family], order)))
    pval_res = {}
    for b1 in order:
        for b2 in order:
            if b1 < b2:
                d1 = metadata[metadata['Oligo family'].eq(b1)][ratio_column]
                d2 = metadata[metadata['Oligo family'].eq(b2)][ratio_column]
                pval_res[(b1, b2)] = stats.mannwhitneyu(d1, d2)
    pval_res = pd.DataFrame(pval_res).T
    pval_res['passed_bonf'], pval_res['p_val'], _, _ = multipletests(pval_res[1], method='bonferroni')
    passed_pvals = pval_res[pval_res['passed_bonf']]['p_val'].to_dict()
    add_stat_annotation(ax, data=metadata, x='Oligo family', y=ratio_column, order=order,
                        box_pairs=list(passed_pvals.keys()),
                        pvalues=list(passed_pvals.values()), perform_stat_test=False
                        )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel("Subgroups of the antigen library")
    ax.text(-0.1, 1.1, string.ascii_lowercase[4], transform=ax.transAxes, size=20, weight='bold')

    # sub-figure f
    ax = fig.add_subplot(spec[1, 2])
    ax.set_axis_off()
    df = get_data_with_outcome('exist')
    metadata = get_oligos_metadata()[oligo_families].rename(columns=oligos_group_to_name).loc[df.columns]
    metadata['Complete library'] = True
    rank_sum_res = {}
    for group in metadata.columns:
        df_group = df.loc[:, metadata[group]].sum(axis=1)
        rank_sum_res[group] = stats.ranksums(df_group.loc[1], df_group.loc[0], alternative='greater')
    rank_sum_res = pd.DataFrame(rank_sum_res).T.rename(columns={0: 'Ranksums', 1: 'p-value'})
    rank_sum_res['Ranksums'] = rank_sum_res['Ranksums'].round(1)
    rank_sum_res['Ranksums\np-value'] = rank_sum_res['p-value'].apply('{:.3f}'.format)
    rank_sum_res.sort_values(by='p-value', inplace=True)
    rank_sum_res.drop(columns='p-value', inplace=True)
    ratio_t_tests = pd.DataFrame({oligos_group_to_name[oligo_family]: dict(
        zip(['t-test\nstatistic', 't-test\np-value'], get_ratio_between_groups(oligo_family))) for oligo_family in
                                  oligo_families}).T
    rank_sum_res = rank_sum_res.merge(ratio_t_tests, left_index=True, right_index=True, how='left')
    rank_sum_res['t-test\nstatistic'] = rank_sum_res['t-test\nstatistic'].round()
    rank_sum_res['t-test\np-value'] = rank_sum_res['t-test\np-value'].apply(lambda val: '' if pd.isnull(val) else '{:.0e}'.format(val))
    rank_sum_res.fillna('', inplace=True)
    table = ax.table(cellText=rank_sum_res.astype(str).values, rowLabels=rank_sum_res.index.values,
                     colLabels=rank_sum_res.columns, loc='center right', colWidths=[0.15]*rank_sum_res.shape[1])
    table.scale(1, 2.9)
    ax.text(-0.1, 1.1, string.ascii_lowercase[5], transform=ax.transAxes, size=20, weight='bold')

    fig.subplots_adjust(bottom=0.15)
    if overwrite:
        plt.savefig(os.path.join(figures_dir, 'figure_1.png'))
    else:
        plt.show()


if __name__ == "__main__":
    overwrite = True
    create_figure_1(overwrite=overwrite)
