import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statannot import add_stat_annotation

from PhageIPSeq_CFS.config import visualizations_dir
from PhageIPSeq_CFS.helpers import get_individuals_metadata_df, get_outcome


def metadata_distribution_figure(metadata, external_spec):
    blood_tests = metadata.columns
    internal_spec = external_spec.subgridspec(1, len(blood_tests), wspace=0.6)
    metadata = metadata.reset_index(level=1)
    for blood_test, ax in zip(blood_tests, internal_spec.subplots()):
        sns.boxplot(data=metadata, x='is_CFS', y=blood_test, ax=ax)
        ax.set(xlabel=blood_test, ylabel='', xticklabels=[])
        if blood_test in ['creat', 'eHelene Guillaume GFR', 'TBil', 'albumin', 'cpk', 't4', 'RF', 'TTGIgA']:
            add_stat_annotation(ax,
                                data=metadata,
                                x='is_CFS', y=blood_test,
                                test='Mann-Whitney', text_format='star', comparisons_correction='bonferroni',
                                box_pairs=[('Sick', 'Healthy')], loc='inside', verbose=False)
    ax.legend(
        handles=[mpatches.Patch(facecolor=sns.color_palette()[0], label='Sick', edgecolor='black'),
                 mpatches.Patch(facecolor=sns.color_palette()[1], label='Healthy', edgecolor='black')],
        bbox_to_anchor=[1, 1])


def get_blood_test_name(blood_name_original):
    ret = blood_name_original[:-len('bloodb')] if blood_name_original.endswith('bloodb') else blood_name_original
    ret = ret[:-len('BloodB')] if ret.endswith('BloodB') else ret
    ret = ' '.join(ret.split('_'))
    if ret == 'sex Binary':
        ret = 'Sex'
    elif ret == 'agegroup Average':
        ret = 'Age group'
    return ret

def get_metadata_comparison_sub_figure(spec):
    # noinspection PyTypeChecker
    metadata = pd.merge(get_individuals_metadata_df().drop(columns='catrecruit_Binary'),
                        get_outcome(return_type=bool).apply(lambda x: 'Sick' if x else 'Healthy'),
                        left_index=True,
                        right_index=True).set_index('is_CFS', append=True)
    metadata.columns = list(map(get_blood_test_name, metadata.columns))
    stacked_metadata = metadata.stack().reset_index(level=2).rename(
        columns={'level_2': 'Blood Test', 0: 'value'}).reset_index()
    metadata_distribution_figure(metadata, external_spec=spec)
