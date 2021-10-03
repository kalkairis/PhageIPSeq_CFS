import os

import matplotlib.pyplot as plt
from LabUtils.addloglevels import sethandlers

from PhageIPSeq_CFS.ComparePopulations.comparing_metadata import get_metadata_comparison_sub_figure
from PhageIPSeq_CFS.Predictions.classifiers import create_auc_with_bootstrap_figure, get_x_y
from PhageIPSeq_CFS.config import visualizations_dir, logs_path, num_auc_repeats, predictor_class, predictor_kwargs
from PhageIPSeq_CFS.helpers import get_imputed_individuals_metadata, get_outcome

if __name__ == "__main__":
    figures_dir = os.path.join(visualizations_dir, 'figure_3')
    os.makedirs(figures_dir, exist_ok=True)
    sethandlers()
    os.chdir(logs_path)
    fig = plt.figure(figsize=(50, 5))
    spec = fig.add_gridspec(2, 3)
    # Add metadata subfigure
    get_metadata_comparison_sub_figure(spec[0, :])

    # Add prediction only from blood tests sub-figure
    x = get_imputed_individuals_metadata()
    y = get_outcome()
    create_auc_with_bootstrap_figure(num_auc_repeats, x, y, predictor_class,
                                     ax=fig.add_subfigure(spec[1, 0]).subplots(), **predictor_kwargs)

    # Add predictions from blood tests and flagella
    # TODO: change to best values with flagella
    x, y = get_x_y(bottom_threshold=0.05, data_type='fold', oligos_subgroup='is_bac_flagella', with_bloodtests=True)
    create_auc_with_bootstrap_figure(num_auc_repeats, x, y, predictor_class,
                                     ax=fig.add_subfigure(spec[1, 1]).subplots(), **predictor_kwargs)
    plt.show()

    print("here")
