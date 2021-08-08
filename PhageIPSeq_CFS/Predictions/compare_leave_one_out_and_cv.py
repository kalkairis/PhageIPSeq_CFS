import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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


def compare_thomas_cross_validation_and_predictions(cross_val_ret, skip_assert=False):
    thomas_cross_val_ret = pd.read_csv(os.path.join(repository_data_dir, 'tmp', 'thomas_predictions.csv'), index_col=0)
    combined_results = pd.merge(cross_val_ret, thomas_cross_val_ret['predict_proba'], left_index=True, right_index=True,
                                how='outer', suffixes=('_mine', '_thomas'))
    ax = plt.subplot()
    sns.scatterplot(data=combined_results, x='predict_proba_mine', y='predict_proba_thomas', hue='y')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()
    plt.close()
    aucs = {}
    for k in ['mine', 'thomas']:
        fpr, tpr, thresholds = roc_curve(combined_results.y, combined_results[f'predict_proba_{k}'])
        aucs[k] = auc(fpr, tpr)
    if not skip_assert:
        assert round(aucs['mine'], 2) == round(aucs['thomas'], 2)


if __name__ == "__main__":
    x = pd.read_csv(os.path.join(repository_data_dir, 'tmp', 'x.csv'), index_col=0)
    y = pd.read_csv(os.path.join(repository_data_dir, 'tmp', 'y.csv'), index_col=0)
    leave_out_out_ret = get_leave_one_out_results(x, y, GradientBoostingClassifier, n_estimators=2000,
                                                  learning_rate=.01, max_depth=6, max_features=1, min_samples_leaf=10)
    cross_val_ret = get_cross_val_results(x, y, GradientBoostingClassifier, n_estimators=2000,
                                          learning_rate=.01, max_depth=6, max_features=1, min_samples_leaf=10)
    compare_thomas_cross_validation_and_predictions(cross_val_ret)
    compare_thomas_cross_validation_and_predictions(leave_out_out_ret, skip_assert=True)
    print("here")
