import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from LabQueue.qp import qp
from LabUtils.addloglevels import sethandlers
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier, LinearRegression
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

from PhageIPSeq_CFS.config import visualizations_dir, logs_path
from PhageIPSeq_CFS.helpers import get_oligos_blood_with_outcome


def run_prediction_and_shap_for_best_results_single_subgroup_and_blood_tests(output_dir, subgroup, auc_repeats, predictor_class, *predictor_args, **predictor_kwargs):
    output_dir = os.path.join(output_dir, subgroup)
    for data_type in ['fold', 'exist']:
        for threshold_percent in [0, 1, 5, 10, 20, 50, 95, 100]:
            df = get_oligos_blood_with_outcome(data_type=data_type, subgroup=subgroup)


if __name__ == "__main__":
    # predictor_class = GradientBoostingClassifier
    # predictor_kwargs = {"n_estimators": 2000, "learning_rate": .01, "max_depth": 6, "max_features": 1,
    #                     "min_samples_leaf": 10}
    predictor_class = LinearRegression
    num_auc_repeats = 10#00
    predictions_dir = os.path.join(visualizations_dir, 'Predictions', predictor_class.__name__)
    all_results = {}
    sethandlers()
    os.chdir(logs_path)
    subgroup = 'all'

