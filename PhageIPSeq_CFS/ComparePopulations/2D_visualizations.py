import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from LabQueue.qp import qp
from LabUtils.addloglevels import sethandlers
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from PhageIPSeq_CFS.config import visualizations_dir, logs_path, oligo_families
from PhageIPSeq_CFS.helpers import get_oligos_with_outcome, get_oligos_metadata_subgroup_with_outcome


def run_2d_visualization_by_threshold(df, transformer_type='PCA', out_figure_path=None, bottom_threshold=0.1):
    if df.isnull().any().any():
        df = df.loc[:, df.isnull().astype(int).mean().ge(bottom_threshold)].copy()
        df.fillna(0, inplace=True)
    else:
        df = df.loc[:, df.eq(0).astype(int).mean().ge(bottom_threshold)].copy()
    transformer_types = {'PCA': PCA, 'TSNE': TSNE}
    transformer = transformer_types[transformer_type]()
    transformed_df = transformer.fit_transform(df)
    columns = list(map(lambda i: f'{transformer_type} {i + 1}', range(transformed_df.shape[1])))
    transformed_df = pd.DataFrame(data=transformed_df, index=df.index, columns=columns)
    transformed_df.reset_index(0, inplace=True)
    most_different_pcs = transformed_df.groupby('is_CFS').median().diff().iloc[1].abs().sort_values(ascending=False).head(2).index
    sns.scatterplot(data=transformed_df, x=most_different_pcs[0], y=most_different_pcs[1], hue='is_CFS')

    if out_figure_path is None:
        out_figure_path = os.path.join(f"{transformer_type}_threshold_{round(bottom_threshold * 100, 0)}.png")
    plt.savefig(out_figure_path)
    plt.close()


def run_pca_visualization_from_entire_oligos_df(data_type='fold', bottom_threshold=0.9, transformer_type='PCA'):
    df = get_oligos_with_outcome(data_type=data_type)
    pca_dir = os.path.join(visualizations_dir, '2D_visualization')
    os.makedirs(pca_dir, exist_ok=True)
    out_path = os.path.join(pca_dir, f"{transformer_type}_threshold_{round(bottom_threshold * 100, 0)}.png")
    run_2d_visualization_by_threshold(df, transformer_type, out_path)
    return bottom_threshold


def run_2d_visualizations_for_groups():
    out_dir = os.path.join(visualizations_dir, '2D_visualization', 'oligos_subgroups')
    os.makedirs(out_dir, exist_ok=True)
    with qp(f"sub2dvis") as q:
        q.startpermanentrun()
        waiton = []
        for data_type in ['fold', 'exist']:
            for oligo_family in oligo_families:
                df = get_oligos_metadata_subgroup_with_outcome(data_type=data_type, subgroup=oligo_family)
                for transformer_type in ['PCA', 'TSNE']:
                    out_figures_dir = os.path.join(out_dir, transformer_type, data_type)
                    os.makedirs(out_figures_dir, exist_ok=True)
                    for threshold_percent in range(0, 100, 5):
                        out_figure_path = os.path.join(out_figures_dir, f"{oligo_family}_"
                                                                        f"threshold_{round(threshold_percent, 0)}.png")
                        waiton.append(q.method(run_2d_visualization_by_threshold,
                                               (df, transformer_type, out_figure_path, threshold_percent / 100)))
        res = q.wait(waiton)


if __name__ == "__main__":
    sethandlers()
    os.chdir(logs_path)
    for transformer_type in ['PCA', 'TSNE']:
        with qp(f"{transformer_type}_visualization") as q:
            q.startpermanentrun()
            waiton = {
                threshold_percent: q.method(run_pca_visualization_from_entire_oligos_df,
                                            ('fold', threshold_percent / 100, transformer_type))
                for threshold_percent in range(0, 100, 5)}
            res = {k: q.waitforresult(v) for k, v in waiton.items()}
    run_2d_visualizations_for_groups()
