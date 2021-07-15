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


def run_pca_visualization(data_type='fold', threshold=0.9, transformer_type='PCA'):
    df = get_oligos_with_outcome(data_type=data_type)
    df = df.loc[:, df.isnull().astype(int).mean().between(1 - threshold, threshold)]
    df.fillna(0, inplace=True)
    transformer_types = {'PCA': PCA, 'TSNE': TSNE}
    transformer = transformer_types[transformer_type]()
    transformed_df = transformer.fit_transform(df)
    columns = list(map(lambda i: f'{transformer_type} {i + 1}', range(transformed_df.shape[1])))
    transformed_df = pd.DataFrame(data=transformed_df, index=df.index, columns=columns)
    transformed_df.reset_index(0, inplace=True)
    sns.scatterplot(data=transformed_df, x=columns[0], y=columns[1], hue='is_CFS')
    pca_dir = os.path.join(visualizations_dir, 'PCA')
    os.makedirs(pca_dir, exist_ok=True)
    plt.savefig(os.path.join(pca_dir, f"{transformer_type}_threshold_{threshold * 100}.png"))
    plt.close()
    return threshold


if __name__ == "__main__":
    sethandlers()
    os.chdir(logs_path)
    for transformer_type in ['PCA', 'TSNE']:
        with qp(f"{transformer_type}_visualization") as q:
            q.startpermanentrun()
            waiton = {
                threshold_percent: q.method(run_pca_visualization, ('fold', threshold_percent / 100, transformer_type))
                for threshold_percent in range(60, 100, 10)}
            res = {k: q.waitforresult(v) for k, v in waiton.items()}
