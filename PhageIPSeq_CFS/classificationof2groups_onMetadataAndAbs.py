# For doing predicitosn (classifiacin) on metadata AND atibdoy data
# % seleciton etc based on the PCA 2 gorups script

import pandas as pd
import numpy as np
import random as rd
import time
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt  # NOTE: This was tested with matplotlib v. 2.1.0

from PhageIPseq.Analyse.Thomas.ComparisonOf2groups.ClassificationOf2groups import do_classifier, XGBphageABclassificaiton
from PhageIPseq.Analyse.Thomas.ComparisonOf2groups.Join2groups_FCorPval_MVandSDPlusGreaterThanXXPercent import LoadGroupData, createPickle
from PhageIPseq.Analyse.Thomas.SpecificCohorts.CFS.ClassificationOf2groups_onMetadata import CFSmetadata


def callingClassificaitonPlusMetadata(MergedDfExist, MergedDfFC, MergedDfLog, summaryDfPredict, G1name, G2name, G1sampleNo, G2sampleNo, num_olis_before_cutoff, num_olis_after_cutoff, PercentageXX, Group1df, Group2df,antigenColName, ShowFig, SaveFig, SaveGroupSumSeparately, CFSmetadataDfOutside, typeOfFeatureData ):
    # prepare Dfs for predicitons

    MergedDfExist_G1 = MergedDfExist[MergedDfExist.index.isin(Group1df.transpose().index)]
    MergedDfExist_G2 = MergedDfExist[MergedDfExist.index.isin(Group2df.transpose().index)]
    MergedDfFC_G1 = MergedDfFC[MergedDfFC.index.isin(Group1df.transpose().index)]
    MergedDfFC_G2 = MergedDfFC[MergedDfFC.index.isin(Group2df.transpose().index)]
    MergedDfLog_G1 = MergedDfLog[MergedDfLog.index.isin(Group1df.transpose().index)]
    MergedDfLog_G2 = MergedDfLog[MergedDfLog.index.isin(Group2df.transpose().index)]

    #print FYI
    #MergedDfExist_G1.to_csv('G1exist100.csv')
    #MergedDfExist_G2.to_csv('G2exist100.csv')
    #print(MergedDfExist_G1)
    #print(MergedDfExist_G2)


    # to do: add a funciton for doing predictions for all 3 types of data
    for df1, df2, typeOfData in [[MergedDfExist_G1, MergedDfExist_G2, 'Exist'], [MergedDfFC_G1, MergedDfFC_G2, 'FC'], [MergedDfLog_G1, MergedDfLog_G2, 'logFC']]:

        #df1=MergedDfFC_G1 #not sure why I had this, maybe from the very beginning? would have overwririten the abvve soecleto of differn tdatatypes
        #df2=MergedDfFC_G2

        # Prepare ala Nastya style --> in this scritp not necessay, because the group info is part of the metadata
        #df1.loc[:, 'resp'] = 0
        #df2.loc[:, 'resp'] = 1

        X = df1.append(df2)
        print('PhIP-Seq data for the the following number of samples:'+str(len(X.index)))

        CFSmetadataDf = CFSmetadataDfOutside.copy() #if not redeifnign here, would keep using the mofieide one, then it coudl not drop the cols below (would proablry be smarer to set once, but should also work like this...)

        #properly format and selet metadata
        CFSmetadataDf.rename(columns={'catrecruit_Binary': 'resp'}, inplace=True)  # just chagnign to the col name susualyl used

        # think about what to do about the nans
        # X = X.fillna(0) # this is proably not ideal, but I want to see if it works. Not a good idea, bc thelahy have much nore NANs, so this biases
        CFSmetadataDf.drop(['3603443_26_LV1016291828', '3066856_27_LV1016291856'], axis=0,
               inplace=True)  # one sample has many nans, this helps keeping more colims when dropping, the o
        CFSmetadataDf = CFSmetadataDf.dropna(axis=1)

        X = X.join(CFSmetadataDf, how='inner')  # could select here columns of the metataa frame if you want to with CFSmetadata[colnames as list]

        print('PhIP-Seq data AND metadata for the the following number of samples:'+str(len(X.index)))


        y = X['resp']

        del X['resp']

        compTitle = str(G1name + ' & ' + G2name + ' on \''+typeOfData+ '\'\n' + str(G1sampleNo) + ' & ' + str(G2sampleNo) + ' samples. ' + str(num_olis_after_cutoff) + '/' + str(num_olis_before_cutoff) + ' peps in >' + str(PercentageXX) + '%')

        #actually nto sure why here X.fillna(1) needed, should work withotu, should be dealted with in previosu
        sumDfFromCalling = do_classifier(X, y, typeOfData, compTitle, summaryDfPredict, G1name, G2name, num_olis_before_cutoff, num_olis_after_cutoff, PercentageXX, G1sampleNo, G2sampleNo, antigenColName, ShowFig, SaveFig, SaveGroupSumSeparately, typeOfFeatureData) # Nastya had here X.fillna, but the way the dfs are here prepare, that should not be necassy (or would be probletmatic for exist and logFC)

        summaryDfPredict = sumDfFromCalling

    return sumDfFromCalling



#calling functions and runncing script
if __name__ == "__main__":

       # Timing
       start_time = time.time()
       print('Starting:')
       print(time.ctime())

       # Change CWD for storing large files
       os.chdir('/net/mraid08/export/genie/Lab/Personal/Thomas/CacheForLoadingPhIPseqData_NewLoader')

       # variables
       studyIDsG1 = [26]
       studyIDsG2 = [27]
       G1name = 'CFS' # do not use > or <, just write gt, lt (greater than, less than), otherwise the filenames are fucked in iWinows
       G2name = 'UK-healthy' # do not use > or <, just write gt, lt (greater than, less than), otherwise the filenames are fucked in iWinows
       groupby_regG1 = 'first'  # if you want to load all samples, put there >None<, do not put an empty string, just write None without anyhting, other options are: ['first', 'latest', 'largest', 'median', 'mean']
       groupby_regG2 = 'first'  # if you want to load all samples, put there >None<, do not put an empty string, just write None without anyhting, other options are: ['first', 'latest', 'largest', 'median', 'mean']

       Type = "fold"  # only intended for "fold" a
       PercentagesList = [1, 5, 10, 20, 50, 100]  # [1, 5, 10, 20, 50, 100] # [0,1,2,4,6,8,10,20,50,100]if you just want to use one, enter just a single numerbrcent, be carfull, if you want to use this, rather cahnge to a comnined percentat (currntely in each group sepatey), Sigal explained on 2.11.2020, that this can cause issues, miniu nunber in how many people (of one group) an oligo needs to appea for data to be used for PCA
       library = 'Agi'  # Agi or Twi, leave empty '' if you want both;  had to change this to sort on oli cutoff AFTER loading all data. Now always the entire data is loaded and cutoff applied, only then do selcetion for Agi or Twist only nti// Agiletn or Twist???check what the names are to be dropped from loaded data, if none = both (none means here that left empty list '')
       numOlisCutOff = 200  # for exlcusing samples with less than # of oligoes

       # optional export option
       pklOfAllSamples = ''  # put thtere 'y' if you want that. If lowercase yes: saves a pickle with the FcData of all samples per cohrot. if you don't want it, levae emtpy


       ShowFig = ''  # string of 'y' to show the plot
       SaveFig = ''  # string of 'y' to show the plot (by default anyways saving the table, this here depends if you want also every plot)
       SaveGroupSumSeparatelyVar = '' # string of 'y' to save a sepate csv for every group that is predied on

       interestingCols = ['is_PNP' ,'is_patho' ,'is_probio' ,'is_IgA' ,'is_bac_flagella', 'is_IEDB_or_cntrl'] # prefereedred order: ['is_PNP','is_patho','is_probio','is_IgA','is_bac_flagella', 'is_IEDB_or_cntrl']


       # loading, preparation of PhIPseq data with loader

       Group1basicData, picklePathG1 = LoadGroupData(studyIDsG1, library, G1name, groupby_regG1, Type)
       _, _, Group1df = createPickle(Group1basicData, picklePathG1, numOlisCutOff, library, Type, pklOfAllSamples)
       Group2basicData, picklePathG2 = LoadGroupData(studyIDsG2, library, G2name, groupby_regG2, Type)
       _, _, Group2df = createPickle(Group2basicData, picklePathG2, numOlisCutOff, library, Type, pklOfAllSamples)

       '''
       # Loading basic dfs from files (optional, rather directly load from other scripts...)
       G1inputDfPath = 'ExtraExportOfFullData-CFS[26]Agifold.pickle'
       G2inputDfPath = 'ExtraExportOfFullData-UK-healthy[27]Agifold.pickle'
       Group1df = pd.read_pickle(G1inputDfPath) #still has nans, figure out why by llading here does not
       Group2df = pd.read_pickle(G2inputDfPath)
       '''

       summaryDfPredictOut = pd.DataFrame(columns=['dataType', 'percent', 'numberOfIndiviudalsPercent'])

       # classifaiciton
       XGBphageABclassificaiton(Group1df, Group2df, G1name, G2name, PercentagesList, interestingCols, summaryDfPredictOut, ShowFig, SaveFig, SaveGroupSumSeparatelyVar, callingClassificaitonPlusMetadata, CFSmetadata, 'phageAbsData+metadata')

       #need to change here the default fucniton to merge with the metadata df from ClassiXficationOf2groups_onMetadata.py


       #add here merging of metadata with Abs



       # Print ending time
       print(time.ctime())

       elapsed_time = time.time() - start_time
       # all the following crap just for conerting secs to hours,min,secs
       hours, rem = divmod(elapsed_time, 3600)
       minutes, seconds = divmod(rem, 60)
       print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

       print("Finished.")
