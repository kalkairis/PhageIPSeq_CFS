import os

import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier

repository_data_dir = os.path.join(os.path.dirname(__file__), 'Data')
visualizations_dir = os.path.join(os.path.dirname(__file__), 'Figures')
logs_path = '/net/mraid08/export/genie/Lab/Phage/jobs'
oligo_families = ['is_PNP', 'is_patho', 'is_probio', 'is_IgA', 'is_bac_flagella', 'is_IEDB_or_cntrl']
oligo_order = ['IEDB or cntrl', 'PNP', 'patho', 'probio', 'IgA', 'bac flagella', 'all']
oligo_families_colors = dict(zip(oligo_order, sns.color_palette()[:len(oligo_order)]))
RANDOM_STATE = 156124
predictor_class = GradientBoostingClassifier
predictor_kwargs = {"n_estimators": 2000, "learning_rate": .01, "max_depth": 6, "max_features": 1,
                    "min_samples_leaf": 10}
num_auc_repeats = 200
