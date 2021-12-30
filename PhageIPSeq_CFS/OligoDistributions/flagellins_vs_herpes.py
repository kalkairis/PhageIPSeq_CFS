import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cross_decomposition import CCA

from PhageIPSeq_CFS.helpers import get_data_with_outcome
def perform_CCA(X, Y):
    X = X.fillna(1)
    Y = Y.fillna(1)
    data = pd.concat([X, Y], axis=1)
    # Instantiate the Canonical Correlation Analysis with 2 components
    my_cca = CCA(n_components=2)

    # Fit the model
    my_cca.fit(X, Y)
    # Obtain the rotation matrices
    xrot = my_cca.x_rotations_
    yrot = my_cca.y_rotations_

    # Put them together in a numpy matrix
    xyrot = np.vstack((xrot, yrot))

    nvariables = xyrot.shape[0]

    plt.figure(figsize=(15, 15))
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))

    # Plot an arrow and a text label for each variable
    for var_i in range(nvariables):
        x = xyrot[var_i, 0]
        y = xyrot[var_i, 1]

        plt.arrow(0, 0, x, y)
        plt.text(x, y, data.columns[var_i], color='red' if var_i >= X.shape[1] else 'blue')

    plt.show()


def is_flagellin(oligo_ids):
    flagellin_ids = {'agilent_106628', 'agilent_129819', 'agilent_183889', 'agilent_181972', 'agilent_201880',
                     'agilent_131884', 'agilent_129839', 'agilent_126599', 'agilent_73216', 'agilent_45366',
                     'agilent_177201', 'agilent_209928', 'agilent_131885', 'agilent_56259', 'agilent_206701'}
    return list(map(lambda oligo: oligo in flagellin_ids, oligo_ids))


def is_herpes(oligo_ids):
    herpes_ids = {'agilent_896', 'agilent_212', 'agilent_1075', 'agilent_319', 'agilent_321', 'agilent_7143',
                  'agilent_12017', 'agilent_12005', 'agilent_8059', 'agilent_12016', 'agilent_12006', 'agilent_7115'}
    return list(map(lambda oligo: oligo in herpes_ids, oligo_ids))


if __name__ == "__main__":
    oligos_df = get_data_with_outcome(imputed=False)
    flagellin_df = oligos_df.iloc[:, is_flagellin(oligos_df.columns)]
    herpes_df = oligos_df.iloc[:, is_herpes(oligos_df.columns)]

    # Compare number of oligos exist per group per person
    df = pd.concat([herpes_df.fillna(1).mean(axis=1), flagellin_df.fillna(1).mean(axis=1)], axis=1).rename(
        columns={0: 'herpes', 1: 'flagellin'})
    df.reset_index(level=0, inplace=True)
    sns.scatterplot(data=df, x='herpes', y='flagellin', hue='is_CFS')
    plt.show()
    plt.close()

    perform_CCA(flagellin_df,
                herpes_df)
    print("here")
