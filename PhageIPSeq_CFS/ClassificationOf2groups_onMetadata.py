# For doing predicitosns just on metata only and not using the PhIP-Seq data
# large parts based on this general purpsoe scrit: /home/vogl/PycharmProjects/PhageIPseq/PhageIPseq/Analyse/Thomas/ComparisonOf2groups/ClassificationOf2groups.py

import pandas as pd
import numpy as np
import random as rd
import time
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt  # NOTE: This was tested with matplotlib v. 2.1.0
from PhageIPseq.Analyse.Thomas.ComparisonOf2groups.ClassificationOf2groups import do_classifier
from PhageIPseq.Analyse.Thomas.ComparisonOf2groups.Join2groups_FCorPval_MVandSDPlusGreaterThanXXPercent import LoadGroupData, createPickle

# Change CWD for storing large files
os.chdir('/net/mraid08/export/genie/Lab/Personal/Thomas/CacheForLoadingPhIPseqData_NewLoader')

#loading CFS metadata (not saved in an offical location yet, so put in the regualr dicoty and load from there)

CFSmetadata = pd.read_csv('CFSmetadataForPredictions.csv', index_col=0)
#remove > and < characters
CFSmetadata = CFSmetadata.replace({'>':'','<':'' }, regex=True)

# ah shit, replacing the >/< does not fix the different datatypes, some are object instead of int64 or float64. That will cause problmes for predicitons I assume
# very stupid way to solve: save as csv and open again
CFSmetadata.to_csv('dummyBackexport_deleteMe.csv')
CFSmetadata = pd.read_csv('dummyBackexport_deleteMe.csv', index_col=0)
#print(CFSmetadata.dtypes)


#calling functions and runncing script
if __name__ == "__main__":

    # Timing
    start_time = time.time()
    print('Starting:')
    print(time.ctime())

    #do predicitons

    #select first x any y

    X=CFSmetadata
    X.rename(columns={'catrecruit_Binary':'resp'}, inplace=True) #just chagnign to the col name susualyl used

    # think about what to do about the nans
    #X = X.fillna(0) # this is proably not ideal, but I want to see if it works. Not a good idea, bc thelahy have much nore NANs, so this biases
    X.drop(['3603443_26_LV1016291828', '3066856_27_LV1016291856'],axis=0, inplace=True) # one sample has many nans, this helps keeping more colims when dropping, the o
    X = X.dropna(axis=1)
    #I think the above aproach with dropping the 2 samples and then the Cols with NAsns is the most sutiale soltion

    #you can sub-select here parts of the metadata by adding the col names to the list
    #X=X[['resp','cpkbloodb']]

    y = X['resp']

    del X['resp']

    #Variables
    typeOfData = 'metadata'
    title = 'Predictions on metadata only as features'
    summaryDf = pd.DataFrame(columns=['dataType', 'percent', 'numberOfIndiviudalsPercent'])
    G1name = 'CFS'
    G2name = 'healthy'
    num_olis_before_cutoff = 'n.a.'
    num_olis_after_cutoff = 'n.a.'
    PercentageXX = 100 #actaully menaingless here and n.a., but cnnot be a string
    G1sampleNo = 40
    G2sampleNo = 40
    antigenColName = 'n.a.'
    ShowFig = 'y'
    SaveFig = 'y'
    SaveGroupSumSeparately = ''

    typeOfFeatureData = 'metadataOnly'


    do_classifier(X,y, typeOfData, title, summaryDf, G1name, G2name, num_olis_before_cutoff, num_olis_after_cutoff, PercentageXX, G1sampleNo, G2sampleNo, antigenColName, ShowFig, SaveFig, SaveGroupSumSeparately, typeOfFeatureData)




    # Print ending time
    print(time.ctime())






    elapsed_time = time.time() - start_time
    # all the following crap just for conerting secs to hours,min,secs
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    print("Finished.")
