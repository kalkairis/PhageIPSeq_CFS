import os

import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from PhageIPSeq_CFS.Predictions.neural_networks_classifier import NeuralNetworkClassifier

repository_data_dir = os.path.join(os.path.dirname(__file__), 'Data')
predictions_outcome_dir = os.path.join(repository_data_dir, 'Predictions')
visualizations_dir = os.path.join(os.path.dirname(__file__), 'Figures')
logs_path = '/net/mraid08/export/genie/Lab/Phage/jobs'
oligo_families = ['is_PNP', 'is_patho', 'is_probio', 'is_IgA', 'is_bac_flagella', 'is_IEDB_or_cntrl']
oligos_group_to_name = {'is_PNP': 'Metagenomics\nantigens', 'all': 'Complete library', 'is_patho': 'Pathogenic strains',
                        'is_probio': 'Probiotic strains', 'is_IgA': 'Antibody-coated\nstrains',
                        'is_bac_flagella': 'Flagellins', 'is_IEDB_or_cntrl': 'IEDB/controls'}
oligo_order = ['Complete library', 'IEDB/controls', 'Metagenomics\nantigens', 'Pathogenic strains', 'Probiotic strains',
               'Antibody-coated\nstrains', 'Flagellins']
oligo_families_colors = dict(zip(oligo_order[1:], sns.color_palette()[:len(oligo_order) - 1]))
oligo_families_colors[oligo_order[0]] = sns.color_palette()[len(oligo_order) - 1]
RANDOM_STATE = 156124
predictors_info = {
    # 'xgboost': {'predictor_class': XGBClassifier, 'predictor_kwargs': {'use_label_encoder': False,
    #                                                                    'objective': 'binary:logistic',
    #                                                                    'eval_metric': 'logloss',
    #                                                                    'nthread': 1}},
    # 'GBR': {'predictor_class': GradientBoostingClassifier,
    #         'predictor_kwargs': {"n_estimators": 2000, "learning_rate": .01, "max_depth": 6,
    #                              "max_features": 1,
    #                              "min_samples_leaf": 10}},
    'NN': {'predictor_class': NeuralNetworkClassifier,
           'predictor_kwargs': {}}
}
nn_predictors_info = {'NN': {'predictor_class': NeuralNetworkClassifier,
                             'predictor_kwargs': {}}}
device = 'cpu'
num_auc_repeats = 200
