import os

from LabUtils.addloglevels import sethandlers
from sklearn.ensemble import GradientBoostingClassifier

from PhageIPSeq_CFS.Predictions.classifiers import get_prediction_results, create_auc_with_bootstrap_figure
from PhageIPSeq_CFS.config import logs_path, visualizations_dir, RANDOM_STATE, predictor_class, predictor_kwargs, num_auc_repeats
from PhageIPSeq_CFS.helpers import get_imputed_individuals_metadata, get_outcome


def auc_prediction_figure(ax=None):
    x = get_imputed_individuals_metadata()
    y = get_outcome()
    create_auc_with_bootstrap_figure(num_auc_repeats, x, y, predictor_class, ax=ax,
                                     **predictor_kwargs)


if __name__ == "__main__":
    predictor_class = GradientBoostingClassifier
    predictor_kwargs = {"n_estimators": 2000, "learning_rate": .01, "max_depth": 6, "max_features": 1,
                        "min_samples_leaf": 10, 'random_state': RANDOM_STATE}
    num_auc_repeats = 1000
    sethandlers()
    os.chdir(logs_path)
    figures_dir = os.path.join(visualizations_dir, 'Predictions_only_with_bloodtests', predictor_class.__name__)
    x = get_imputed_individuals_metadata()
    y = get_outcome()
    ret = get_prediction_results(figures_dir, x, y, num_confidence_intervals_repeats=num_auc_repeats,
                                 predictor_class=predictor_class, **predictor_kwargs)
    print(ret)
