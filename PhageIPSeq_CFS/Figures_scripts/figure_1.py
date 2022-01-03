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

from PhageIPSeq_CFS.config import visualizations_dir, oligo_families, oligo_families_colors
from PhageIPSeq_CFS.helpers import get_data_with_outcome, get_oligos_metadata, split_xy_df_and_filter_by_threshold, \
    get_individuals_metadata_df


def create_figure_1(with_table=False, overwrite=True):
    figures_dir = os.path.join(visualizations_dir, 'figure_1')
    os.makedirs(figures_dir, exist_ok=True)
    fig = plt.figure(figsize=(20, 10))
    spec = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # sub-figure a - Thomas will add manually the overview of the system
    ax = fig.add_subplot(spec[0, 0])
    ax.set_axis_off()
    ax.text(-0.1, 1.1, string.ascii_uppercase[0], transform=ax.transAxes, size=20, weight='bold')

    # sub-figure b - cohort info
    cohort_info = get_individuals_metadata_df()[['catrecruit_Binary', 'sex_Binary', 'agegroup_Average']]
    cohort_info['Sex'] = cohort_info['sex_Binary'].apply(lambda s: 'Male' if bool(s) else 'Female')
    ax = fig.add_subplot(spec[0, 1])
    ax.text(-0.1, 1.1, string.ascii_uppercase[1], transform=ax.transAxes, size=20, weight='bold')
    ax.set_axis_off()
    internal_spec = spec[0, 1].subgridspec(2, 1, height_ratios=[1, 2])
    ax = fig.add_subplot(internal_spec[0])
    ax.set_axis_off()
    if with_table:
        summary_table = cohort_info.copy()
        summary_table['outcome'] = summary_table['catrecruit_Binary'].apply(
            lambda x: 'CFS' if bool(x) else 'Healthy controls')
        summary_groups = summary_table.groupby('outcome')
        total_number = summary_groups.count().iloc[:, 0].to_frame()
        total_number.columns = ['Total number']
        total_number = total_number.stack()
        total_number = total_number.to_frame().reset_index(-1).rename(columns={'level_1': 'level_2'})
        sex_summary = summary_table.groupby(['outcome', 'Sex']).count().iloc[:, 0].to_frame()
        sex_summary.columns = ['Sex']
        sex_summary = sex_summary.stack()
        sex_summary = pd.DataFrame({k: {'Male/Female': f'{v["Male"]}/{v["Female"]}'} for k, v in
                                    sex_summary.reset_index(-1).drop(columns='level_2')[0].unstack(
                                        -1).T.to_dict().items()}).T.stack()
        sex_summary = sex_summary.to_frame().reset_index(-1)
        sex_summary['level_2'] = 'Sex'
        age_range_summary = summary_groups['agegroup_Average'].agg([min, max]).apply(
            lambda row: f"{row['min']} to {row['max']}", axis=1)
        age_averages = summary_groups['agegroup_Average'].agg(['mean', 'std']).round(1).apply(
            lambda row: f"{row['mean']} +- {row['std']}", axis=1)
        ages = pd.DataFrame({'Range': age_range_summary, r'Average +- SD': age_averages}).stack().to_frame()
        ages['level_2'] = 'Age'
        ages = ages.reset_index(-1)
        summary_table = pd.concat([total_number, sex_summary, ages]).sort_index().reset_index()
        table = ax.table(cellText=summary_table[[0]].astype(str).values,
                         loc='center right', colWidths=[0.3], )
        w = table.get_celld()[(0, 0)].get_width()
        h = table.get_celld()[(0, 0)].get_height()
        for i in range(2):
            cell_1 = table.add_cell((i * 4) + 0, -1, w, h)
            cell_1.visible_edges = "TBR"
            cell_2 = table.add_cell((i * 4) + 0, -2, w, h)
            cell_2.get_text().set_text('Total number')
            cell_2.visible_edges = "TLB"
            cell = table.add_cell((i * 4) + 1, -1, w, h)
            cell.get_text().set_text('Male/Female')
            cell = table.add_cell((i * 4) + 1, -2, w, h)
            cell.get_text().set_text('Sex')
            cell = table.add_cell((i * 4) + 2, -1, w, h)
            cell.get_text().set_text('Range')
            cell = table.add_cell((i * 4) + 3, -1, w, h)
            cell.get_text().set_text(r"Average $\pm$ SD")
            cell_1 = table.add_cell((i * 4) + 2, -2, w, h)
            cell_1.visible_edges = "TRL"
            cell_1.get_text().set_text('Age')
            cell_2 = table.add_cell((i * 4) + 3, -2, w, h)
            cell_2.visible_edges = "BRL"
            cells = [table.add_cell((i * 4) + j, -3, w / 4, h) for j in range(4)]
            cells[0].visible_edges = 'LTR'
            cells[1].visible_edges = 'LR'
            cells[2].visible_edges = 'LR'
            cells[3].visible_edges = 'BLR'
        ax.annotate('CFS', xy=(0.03, 0.7), rotation=90)
        ax.annotate('Healthy', xy=(0.03, 0.005), rotation=90)

    internal_figure_spec = internal_spec[1].subgridspec(2, 2, wspace=0, hspace=0.1, height_ratios=[5, 1])
    ax = fig.add_subplot(internal_figure_spec[0, 0])
    sns.histplot(data=cohort_info[cohort_info['catrecruit_Binary'].eq(1)],
                 y='agegroup_Average', hue='Sex',
                 palette=[sns.color_palette("colorblind")[-4], sns.color_palette("colorblind")[-1]],
                 multiple="stack", bins=4, ax=ax, legend=False)
    ax.set_ylabel('Age group average')
    ax.set_xlabel('CFS')
    ax.set_xlim(*list(ax.get_xlim())[::-1])
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
    ax = fig.add_subplot(internal_figure_spec[0, 1], sharey=ax)
    ax.get_yaxis().set_visible(False)
    ax = sns.histplot(data=cohort_info[cohort_info['catrecruit_Binary'].eq(0)], y='agegroup_Average',
                      bins=4,
                      palette=[sns.color_palette("colorblind")[-4], sns.color_palette("colorblind")[-1]],
                      hue='Sex',
                      multiple="stack", legend=False,
                      ax=ax)

    ax.set_xlabel('Healthy')
    ax = fig.add_subplot(internal_figure_spec[1, :])
    ax.set_axis_off()
    ax.legend(
        handles=[mpatches.Patch(facecolor=sns.color_palette("colorblind")[-4], label='Male', edgecolor='black'),
                 mpatches.Patch(facecolor=sns.color_palette("colorblind")[-1], label='Female', edgecolor='black')],
        loc='lower center', ncol=2)

    # sub-figure c
    ax = fig.add_subplot(spec[0, 2])
    ax.set_axis_off()
    ax.text(-0.1, 1.1, string.ascii_uppercase[2], transform=ax.transAxes, size=20, weight='bold')
    df = get_data_with_outcome(data_type='exist').sum(axis=1).reset_index(0).rename(columns={0: 'num_oligos'})
    df['status'] = df['is_CFS'].astype(bool).apply(lambda x: 'CFS' if x else 'Healthy')

    df2 = get_data_with_outcome().fillna(1)
    df2['shannon_div'] = df2.apply(lambda row: shannon(row), axis=1)
    df2['status'] = df2.reset_index()['is_CFS'].astype(bool).apply(lambda x: 'CFS' if x else 'Healthy').values

    internal_figure_spec = spec[0, 2].subgridspec(1, 2, wspace=0)
    ax = fig.add_subplot(internal_figure_spec[0])
    sns.boxplot(data=df, x='status', y='num_oligos', palette=sns.color_palette("muted")[-2:][::-1], ax=ax)
    ax.set_xlabel('')
    ax.spines['right'].set_linestyle('dashed')
    ax.set_ylabel('Number of oligos per individual')

    ax = fig.add_subplot(internal_figure_spec[1])
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    ax2 = ax.twinx()
    ax2 = sns.boxplot(data=df2, x='status', y='shannon_div', palette=sns.color_palette()[-2:][::-1], ax=ax2)
    ax2.set_xlabel('')
    ax2.spines['left'].set_visible(False)
    ax2.set_ylabel(r'Shannon $\alpha$-diversity')

    # sub-figure d - scatter plot of expression
    ax = fig.add_subplot(spec[1, 0])
    df = get_data_with_outcome(data_type='exist')
    cfs_label = 'Number of CFS patients in whome a\npeptite is significantly bound'
    healthy_label = 'Number of healthy patients in whome\na peptite is significantly bound'
    df['disease_status'] = df.reset_index()['is_CFS'].astype(bool).apply(
        lambda x: cfs_label if x else healthy_label).values
    df = df.groupby('disease_status').sum().T
    df['is_flagellin'] = get_oligos_metadata()['is_bac_flagella']
    df.sort_values(by='is_flagellin', inplace=True)
    sns.scatterplot(data=df, x=cfs_label,
                    y=healthy_label, hue='is_flagellin',
                    palette=['lightgray', oligo_families_colors['bac flagella']],
                    ax=ax)
    legend_handels, legend_labels = ax.get_legend_handles_labels()
    legend_handels = [legend_handels[legend_labels.index('True')]]
    legend_labels = ['Bacterial flagellins']
    ax.legend(handles=legend_handels, labels=legend_labels)

    ax.text(-0.1, 1.1, string.ascii_uppercase[3], transform=ax.transAxes, size=20, weight='bold')

    # sub-figure e
    ax = fig.add_subplot(spec[1, 1])
    x, y = split_xy_df_and_filter_by_threshold(get_data_with_outcome(data_type='exist'))
    df = pd.concat([x, y], axis=1)
    df = df.groupby('is_CFS').sum()
    df = df.loc[:, df.ne(0).all()].copy()
    metadata = get_oligos_metadata()[oligo_families].rename(
        columns={col: ' '.join(col.split('_')[1:]) for col in oligo_families}).loc[df.columns].stack()
    metadata = metadata[metadata].reset_index(level=-1).drop(columns=0).rename(columns={'level_1': 'Oligo family'})
    ratio_column = 'Number of patients with oligo\nexpression CFS to healthy ratio'
    metadata[ratio_column] = df.apply(
        lambda oligo: oligo[int(True)] / oligo[int(False)])
    order = ['PNP', 'patho', 'probio', 'IgA', 'bac flagella', 'IEDB or cntrl']
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
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.text(-0.1, 1.1, string.ascii_uppercase[4], transform=ax.transAxes, size=20, weight='bold')

    # sub-figure f
    ax = fig.add_subplot(spec[1, 2])
    ax.set_axis_off()
    df = get_data_with_outcome(data_type='exist').reset_index(0).groupby('is_CFS').sum().T
    metadata = get_oligos_metadata()[oligo_families].rename(
        columns={col: ' '.join(col.split('_')[1:]) for col in oligo_families}).loc[df.index]
    rank_sum_res = {}
    for col in metadata.columns:
        rank_sum_res[col] = stats.ranksums(*df.loc[metadata[col]].T.values)
    rank_sum_res = pd.DataFrame(rank_sum_res).T.rename(columns={0: 'Ranksums', 1: 'p-value'})
    rank_sum_res['Ranksums'] = rank_sum_res['Ranksums'].round(1)
    rank_sum_res['p-value'] = rank_sum_res['p-value'].apply('{:.0e}'.format)
    table = ax.table(cellText=rank_sum_res.astype(str).values, rowLabels=rank_sum_res.index.values,
                     colLabels=rank_sum_res.columns, loc='center right', colWidths=[0.3, 0.3, 0.3])
    table.scale(1, 2)
    ax.text(-0.1, 1.1, string.ascii_uppercase[5], transform=ax.transAxes, size=20, weight='bold')

    if overwrite:
        plt.savefig(os.path.join(figures_dir, 'figure_1' + ('_with_table' if with_table else '') + '.png'))
    else:
        plt.show()


if __name__ == "__main__":
    overwrite = False
    create_figure_1(with_table=True, overwrite=overwrite)
    create_figure_1(with_table=False, overwrite=overwrite)
