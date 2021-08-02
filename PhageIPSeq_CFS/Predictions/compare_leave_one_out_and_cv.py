import os

import numpy as np
import pandas as pd
from numpy import interp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from PhageIPSeq_CFS.Predictions.classifiers import run_leave_one_out_prediction
from PhageIPSeq_CFS.config import repository_data_dir


def get_leave_one_out_results(x, y, predictor, **predictor_kwargs):
    ret = run_leave_one_out_prediction(x, y, predictor, **predictor_kwargs)
    fpr, tpr, thresholds = roc_curve(ret.y, ret.predict_proba)
    roc_auc = auc(fpr, tpr)
    print(f"for leave one out: auc={roc_auc}")
    return ret


def get_cross_val_results(X, y, predictor, **predictor_kwargs):
    classifier = predictor(**predictor_kwargs)
    cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=104)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    y_pred = []
    y_true = []
    folds = []
    trhresh_all = []
    for i, (train, test) in enumerate(cv.split(X, y)):
        fitted = classifier.fit(X.iloc[train], y.iloc[train].values.ravel())
        probas_ = fitted.predict_proba(X.iloc[test])
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        trhresh_all += [thresholds]
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        # print(f"CV #{i}: roc_auc={roc_auc}")
        aucs.append(roc_auc)
        y_pred.append(probas_[:, 1])
        y_true.append(y.iloc[test])
        folds.append([i] * len(test))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    print(f"CV: mean auc={mean_auc} std={std_auc}, aucs={aucs}")
    ret = pd.concat(y_true)
    ret.columns = ['y']
    ret['predict_proba'] = np.concatenate(y_pred)
    fpr, tpr, thresholds = roc_curve(ret.y, ret.predict_proba)
    roc_auc = auc(fpr, tpr)
    print(f"CV for all samples together: AUC={roc_auc}")
    return ret


if __name__ == "__main__":
    x = pd.read_csv(os.path.join(repository_data_dir, 'tmp', 'x.csv'), index_col=0)
    y = pd.read_csv(os.path.join(repository_data_dir, 'tmp', 'y.csv'), index_col=0)
    leave_out_out_ret = get_leave_one_out_results(x, y, GradientBoostingClassifier, n_estimators=2000,
                                                  learning_rate=.01, max_depth=6, max_features=1, min_samples_leaf=10)
    cross_val_ret = get_cross_val_results(x, y, GradientBoostingClassifier, n_estimators=2000,
                                          learning_rate=.01, max_depth=6, max_features=1, min_samples_leaf=10)
    print("here")
