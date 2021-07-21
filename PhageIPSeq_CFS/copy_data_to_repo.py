import os

import pandas as pd
from LabData.DataLoaders.PhIPSeqLoader import PhIPSeqLoader

from PhageIPSeq_CFS.config import repository_data_dir

# Comment on metadata - catrecruit_Binary is 1 for CFS and 0 for UK healthy controls
if __name__ == "__main__":
    individuals_metadata_path = os.path.join(repository_data_dir, 'individuals_metadata.csv')
    individuals_metadata_df = pd.read_csv(individuals_metadata_path, index_col=0)
    for data_type in ["fold", "exist", "p_val"]:
        loader = PhIPSeqLoader()
        df = loader.get_data(data_type=data_type, library='Agilent').df.loc[individuals_metadata_df.index.values]
        print(f"{data_type} initial size: {df.shape[1]}")
        df = df.iloc[:, df.notnull().any().values]
        df = df.iloc[:, df.ne(0).any().values]
        print(f"{data_type} final size: {df.shape[1]}")
        df.to_csv(os.path.join(repository_data_dir, f"{data_type}_df.csv"))
    oligos_metadata_df = pd.read_pickle(
        os.path.join('/net/mraid08/export/genie/Lab/Phage/Analysis/Cache', 'df_info_agilent_final.pkl')).drop(
        columns=['nuc_seq', 'aa_seq', 'full_aa_seq'])
    oligos_metadata_df.drop(
        columns=[col for col in oligos_metadata_df.columns if col.startswith('hash') or col.startswith('end')],
        inplace=True)
    # oligos_metadata_df.to_csv(
    #     os.path.join(repository_data_dir, 'oligos_metadata.csv'))
    print("here")
