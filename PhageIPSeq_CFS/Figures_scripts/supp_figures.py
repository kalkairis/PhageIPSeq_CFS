import os
import string

import matplotlib.pyplot as plt
import seaborn as sns

from PhageIPSeq_CFS.config import oligo_families, visualizations_dir
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
    cfs_label = 'Number of CFS patients in whome a\npeptite is significantly bound'
    healthy_label = 'Number of healthy patients in whome\na peptite is significantly bound'
    df['disease_status'] = df.reset_index()['is_CFS'].astype(bool).apply(
        lambda x: cfs_label if x else healthy_label).values
    df = df.groupby('disease_status').sum().T
    for i, oligo_group in enumerate(oligo_families):
        ax = fig.add_subplot(spec[i // 3, i % 3])
        create_single_oligo_group_vs_all_distribution(df, ax, oligo_group, cfs_label, healthy_label)
        ax.text(-0.1, 1.1, string.ascii_uppercase[i], transform=ax.transAxes, size=20, weight='bold')
    if overwrite:
        plt.savefig(os.path.join(output_dir, 'supp_figure_1.png'))
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    supp_figures_dir = os.path.join(visualizations_dir, 'supp_figures')
    os.makedirs(supp_figures_dir, exist_ok=True)
    overwrite = True
    create_supp_figure_oligo_group_vs_all_distribution(supp_figures_dir, overwrite=overwrite)
