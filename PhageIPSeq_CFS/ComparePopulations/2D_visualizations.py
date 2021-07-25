import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from LabQueue.qp import qp
from LabUtils.addloglevels import sethandlers
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from PhageIPSeq_CFS.config import visualizations_dir, logs_path
from PhageIPSeq_CFS.helpers import get_oligos_with_outcome


def run_2d_visualization_by_threshold(df, transformer_type='PCA', out_figure_path=None):
    transformer_types = {'PCA': PCA, 'TSNE': TSNE}
    transformer = transformer_types[transformer_type]()
    transformed_df = transformer.fit_transform(df)
    columns = list(map(lambda i: f'{transformer_type} {i + 1}', range(transformed_df.shape[1])))
    transformed_df = pd.DataFrame(data=transformed_df, index=df.index, columns=columns)
    transformed_df.reset_index(0, inplace=True)
    sns.scatterplot(data=transformed_df, x=columns[0], y=columns[1], hue='is_CFS')

    if out_figure_path is None:
        out_figure_path = os.path.join(f"{transformer_type}_threshold.png")
    plt.savefig(out_figure_path)
    plt.close()


def run_pca_visualization_from_entire_oligos_df(data_type='fold', bottom_threshold=0.9, transformer_type='PCA'):
    df = get_oligos_with_outcome(data_type=data_type)
    df = df.loc[:, df.isnull().astype(int).mean().ge(bottom_threshold)]
    df.fillna(0, inplace=True)
    pca_dir = os.path.join(visualizations_dir, '2D_visualization')
    os.makedirs(pca_dir, exist_ok=True)
    out_path = os.path.join(pca_dir, f"{transformer_type}_threshold_{round(bottom_threshold * 100, 0)}.png")
    run_2d_visualization_by_threshold(df, transformer_type, out_path)
    return bottom_threshold

def run_2d_visualizations_for_groups(data_type='fold'):
    df = get_oligos_with_outcome(data_type=data_type)


if __name__ == "__main__":
    sethandlers()
    # os.chdir(logs_path)
    # for transformer_type in ['PCA', 'TSNE']:
    #     with qp(f"{transformer_type}_visualization") as q:
    #         q.startpermanentrun()
    #         waiton = {
    #             threshold_percent: q.method(run_pca_visualization_from_entire_oligos_df,
    #                                         ('fold', threshold_percent / 100, transformer_type))
    #             for threshold_percent in range(0, 100, 5)}
    #         res = {k: q.waitforresult(v) for k, v in waiton.items()}
    run_2d_visualizations_for_groups()
