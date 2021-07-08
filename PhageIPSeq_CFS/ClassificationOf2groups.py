# For doing predicitosn (classifiacin) with Nastya's code, starting from pickel loaded by join2groups... or ICI loading
# % seleciton etc based on the PCA 2 gorups script

import pandas as pd
import numpy as np
import random as rd
import time
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt  # NOTE: This was tested with matplotlib v. 2.1.0
from PhageIPseq.Analyse.Thomas.ComparisonOf2groups.Join2groups_FCorPval_MVandSDPlusGreaterThanXXPercent import LoadGroupData, createPickle


#functions

def mergeDfsForPred(Group1df, Group2df, G1name, G2name):
    G1sampleNo = len(Group1df.columns)
    G2sampleNo = len(Group2df.columns)
    print('Number of samples G1 (' + str(G1name) + '): ' + str(G1sampleNo))
    print('Number of samples G2 (' + str(G2name) + '): ' + str(G2sampleNo))

    # --> if loading the basic dataframes with differetn scripts, there can be an issue that one Df has the multineidne for cols and the other one just one baroced --> add a fucntion that universally slectes just barcode
    def unifyHeader(df):
        if len(df.columns.names) > 1:
            df.columns = df.columns.get_level_values(
                0)  # not sure that this always works, could be an issue with which sscirpt genreated and that sometime maybe plate number first...
        return df

    Group1df = unifyHeader(Group1df)
    Group2df = unifyHeader(Group2df)

    # Merging the 2 dfs
    # remove multiindex info
    if Group1df.index.nlevels == 2:  # added if satemtn bc for some scirpts (loadingICI e.g.) the addtioanl index layers are already dropped. This here is not univerasl, but should worl in most cases
        Group1df = Group1df.droplevel(level=[1, 2], axis=1)
    if Group1df.index.nlevels == 2:
        Group2df = Group2df.droplevel(level=[1, 2], axis=1)
    # Merging
    MergedDf = Group1df.combine_first(
        Group2df)  # this is not preserivng the order of the oculumsn (= samples), so the two groups are now mixed


    MergedDftrans = MergedDf.transpose()

    # MergedDf.to_csv('mergedDf.csv')

    return(MergedDftrans, MergedDf,  G1sampleNo, G2sampleNo)

def subgroupSelection(MergedDf, colName):
    #selecting specific oligo groups like flaggelins

    # works in princie but add a loop to do on mulitple groups and also add saving to df + plotting
    # Full Agi info:
    base_path = "/net/mraid08/export/genie/Lab/Phage/Analysis"
    cache_path = os.path.join(base_path, "Cache")
    AgiOligoPkl_final_allsrc = pd.read_pickle(os.path.join(cache_path,
                                                           "df_info_agilent_final.pkl"))  # commmetn 21.01.2021: not sure why this  df is called _allsrc, but the pathname is ok as is
    AgiOligoPkl_final_allsrc.index.name = 'order'
    #AgiOligoPkl_final_allsrc.to_csv('agiInfo.csv')

    selectOlis = AgiOligoPkl_final_allsrc.loc[AgiOligoPkl_final_allsrc[colName] == True]

    MergedDf = MergedDf[MergedDf.index.isin(selectOlis.index)]

    MergedDftrans = MergedDf.transpose()

    return(MergedDf, MergedDftrans)



def formatDataforPred(MergedDftrans, MergedDf, PercentageXX):
    print('Processing data for cutoff ' + str(PercentageXX) + '%...')
    # Selecting oligoes over Cutoff
    # just counting smaples
    FcDataDf = MergedDftrans
    FcDataDf[FcDataDf < 0] = 0  # this is because -1 is "not_scored", and not a value. - not sure why necessay here, but otherwise may give negatice vlauses in the plot for comapring precentages
    num_samples_per_oli = FcDataDf.mask(FcDataDf > 0, 1).sum(axis=0)  # replacing any sig. FC with 1 to ocunt them
    num_samples_per_oli.rename("num_samples_per_oli",
                               inplace=True)  # series needs to have a name in order ot append it
    FcDataOlis = FcDataDf.append(num_samples_per_oli)
    # calcusitng perceent
    NumberOfSamples = len(
        FcDataOlis.index) - 1  # for getting the number of samples for the percent calc. -1 because the additonal sumamry line is irrelantve
    print('Number of samples merged groups: ' + str(NumberOfSamples))
    FcDataOlis.loc['AsPercent'] = FcDataOlis.loc['num_samples_per_oli'].apply(lambda x: ((x / NumberOfSamples) * 100))
    FcDataOlis = FcDataOlis.loc[['num_samples_per_oli', 'AsPercent']]

    FcDataOlisTrans = FcDataOlis.transpose()  # transpsoe bc ususally always oigeos as index
    # print(FcDataOlisTrans)

    # Until now it returns a Df with percentage for every oligo (but does not contain all the data per sample anymore)
    # Now apply the cutoff of percent to get only the relevant oligoes above a cutoff (this can then be used for selctein the data from th eorignla df)
    print('Number of oligoes before cutoff (' + str(PercentageXX) + '): ' + str(len(FcDataOlisTrans.index)))
    num_olis_before_cutoff = len(FcDataOlisTrans.index)
    # GreaterThanXX = FcDataOlisTrans[FcDataOlisTrans['AsPercent']].ge(PercentageXX, axis='1') #ge = greter than thingy
    GreaterThanXX = FcDataOlisTrans[FcDataOlisTrans['AsPercent'] >= PercentageXX]  # ge = greter than thingy
    print('Number of oligoes after cutoff (' + str(PercentageXX) + '): ' + str(len(GreaterThanXX.index)))
    num_olis_after_cutoff = len(GreaterThanXX.index)

    # Select full data for relevant oligoes above percent cutoff
    MergedDf = MergedDf[MergedDf.index.isin(
        GreaterThanXX.index)]  # should have better called this just SelecitonDf, but to lazy to change the fucnitons below

    # Doing this on FC, logFC or exist?
    # Sigal said all could be interesitnt, so just plot all
    #  about nans: for existience set to 0, FC set to 1, logFC 0

    MergedDfExist = MergedDf.copy()
    MergedDfExist[MergedDfExist > 1] = 1  # check with Sigal, but I guess exist is FC greater than 1 (bc less than woudl be deptelded)
    MergedDfExist = MergedDfExist.fillna(0)
    MergedDfFC = MergedDf.fillna(1)
    MergedDfLog = np.log(MergedDfFC)
    MergedDfLog = MergedDfLog.replace([np.inf, -np.inf],
                                      np.nan)  # not sure why this is needed, but seems even if filled nans before, maybe by rounding some turnn 0? happens onyl with some cohrot...
    MergedDfLog = MergedDfFC = MergedDf.fillna(0)

    #Transpose to have suitalble format for Nastya's predicitons
    MergedDfExist = MergedDfExist.transpose()
    MergedDfFC = MergedDfFC.transpose()
    MergedDfLog = MergedDfLog.transpose()

    return(MergedDfExist, MergedDfFC, MergedDfLog, num_olis_before_cutoff, num_olis_after_cutoff)

    '''#check
    print('Exist----------------------------------------------')
    print(MergedDfExist)
    print('FC----------------------------------------------')
    print(MergedDfFC)
    print('Log----------------------------------------------')
    print(MergedDfLog)

    MergedDfFC.to_csv('MergedDfFC.csv')
    exit()
    '''



    #########################
    #
    # Do predicitons
    #
    #########################



def do_classifier(X, y, typeOfData, title, summaryDf, G1name, G2name, num_olis_before_cutoff, num_olis_after_cutoff, PercentageXX, G1sampleNo, G2sampleNo, antigenColName, ShowFig, SaveFig, SaveGroupSumSeparately, typeOfFeatureData):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_curve, auc
        from scipy import interp
        import matplotlib.pyplot as plt
        import numpy as np

        y = y[y.index.isin(X.index)].sort_index()
        X = X[X.index.isin(y.index)].sort_index()
        classifier = GradientBoostingClassifier(n_estimators = 2000,learning_rate=.01,max_depth=6,max_features=1,min_samples_leaf=10)
        cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=104)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        y_test_all = []
        y_predict_all = []
        y_pred = []
        y_true = []
        trhresh_all = []
        i = 0
        for i, (train, test) in enumerate(cv.split(X, y)):
            fitted = classifier.fit(X.iloc[train], y.iloc[train])
            probas_ = fitted.predict_proba(X.iloc[test])
            fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
            trhresh_all+=[thresholds]
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3)
            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")

        if antigenColName == None:
            antigenColName = 'noSubSelection'

        figFilename = 'ClassificationOf2groups-'+ G1name + '+' + G2name + '-'+typeOfFeatureData+'-inGreaterThan' + str(PercentageXX) + '%-'+typeOfData+'-'+str(antigenColName)+'.png'
        if SaveFig == 'y':
            plt.savefig(figFilename, bbox_inches='tight')
            print('Saved ' + figFilename + '.')
            print('------------------------------------------------------------------------------------------------------------------------------------------')

        numberOfIndiviudalsPercent = (PercentageXX/100)*(G1sampleNo+G2sampleNo)

        #'numOlisPred', 'predictMV', 'predictSD'
        subgroupColNumOlisPred = antigenColName+'-numOlisPred'
        subgroupColPredictMV = antigenColName+'-predictMV'
        subgroupColPredictSD = antigenColName+'-predictSD'

        listCols = [subgroupColNumOlisPred, subgroupColPredictMV, subgroupColPredictSD]

        if set(listCols).issubset(summaryDf.columns) == False:
            summaryDf = summaryDf.reindex(summaryDf.columns.tolist() + [subgroupColNumOlisPred,subgroupColPredictMV,subgroupColPredictSD], axis=1)

        dictSummEntry = {'dataType':typeOfData,'percent':PercentageXX,'numberOfIndiviudalsPercent':numberOfIndiviudalsPercent, subgroupColNumOlisPred:num_olis_after_cutoff, subgroupColPredictMV:mean_auc, subgroupColPredictSD:std_auc}
        rowIndex = G1name +'(n='+str(G1sampleNo)+')vs.' + G2name +'(n='+str(G2sampleNo)+')-in>' + str(PercentageXX) + '%-'+typeOfData
        summaryDf.loc[rowIndex] = dictSummEntry
        fileNameToReferTo = 'ClassificationOf2groups-'+ G1name + '+' + G2name +'-'+typeOfFeatureData+'-Subgroup-'+str(antigenColName)+'.csv'
        if SaveGroupSumSeparately == 'y':
            summaryDf.to_csv(fileNameToReferTo)
        print('Saved or appendended '+fileNameToReferTo+'.')

        if ShowFig == 'y':
            plt.show()
        else:
            plt.close()

        return summaryDf

def callingClassificaiton(MergedDfExist, MergedDfFC, MergedDfLog, summaryDfPredict, G1name, G2name, G1sampleNo, G2sampleNo, num_olis_before_cutoff, num_olis_after_cutoff, PercentageXX, Group1df, Group2df,antigenColName, ShowFig, SaveFig, SaveGroupSumSeparately, optionalMetadataDf, typeOfFeatureData):
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

        # Prepare ala Nastya style
        df1.loc[:, 'resp'] = 0
        df2.loc[:, 'resp'] = 1

        X = df1.append(df2)

        #X.to_csv('X_beforeShuffeling.csv') for double checking

        #X = X.sample(frac=1) # to shuffle randomly - here not necearryy becasue anyways part of NAstyas gucniton

        #X.to_csv('X_afterShuffeling.csv') #for double checkng
        #print(X)


        y = X['resp']

        del X['resp']

        compTitle = str(G1name + ' & ' + G2name + ' on \''+typeOfData+ '\'\n' + str(G1sampleNo) + ' & ' + str(G2sampleNo) + ' samples. ' + str(num_olis_after_cutoff) + '/' + str(num_olis_before_cutoff) + ' peps in >' + str(PercentageXX) + '%')

        #actually nto sure why here X.fillna(1) needed, should work withotu, should be dealted with in previosu
        sumDfFromCalling = do_classifier(X, y, typeOfData, compTitle, summaryDfPredict, G1name, G2name, num_olis_before_cutoff, num_olis_after_cutoff, PercentageXX, G1sampleNo, G2sampleNo, antigenColName, ShowFig, SaveFig, SaveGroupSumSeparately, typeOfFeatureData) # Nastya had here X.fillna, but the way the dfs are here prepare, that should not be necassy (or would be probletmatic for exist and logFC)

        summaryDfPredict = sumDfFromCalling

    return sumDfFromCalling

#One master function
def XGBphageABclassificaiton(Group1df, Group2df, G1name, G2name, PercentagesList, interestingCols, sumDf, ShowFig, SaveFig, SaveGroupSumSeparately, callingClassificaiton, optionalMetadataDf, typeOfFeatureData):

    MergedDftrans, MergedDf,  G1sampleNo, G2sampleNo = mergeDfsForPred(Group1df, Group2df, G1name, G2name)

    if G1sampleNo == 0 or G2sampleNo == 0:
        print('One of the two groups has 0 samples.')
        GisZeroFileName = 'ClassificationOf2groups-Summary-' + G1name + '+' + G2name + +'-'+typeOfFeatureData+'-EmptyAsOneOfTheGroupsHasZeroSamples(G1-' + str(G1sampleNo)+',G2-'+str(G2sampleNo)+').csv'
        GisZeroFile = open(GisZeroFileName, 'x')

        #thought for a while I may need to check if the file already exsist, but that shoudl never be neceassasry (unless you run the script and don't delte it...)
        #if os.path.isfile(GisZeroFileName):
        #    print('File already exists.')
        #else:
        #    GisZeroFile = open(GisZeroFileName, 'x')

    if G1sampleNo > 0 and G2sampleNo > 0:

        #run once on the full data, then loop over subgorups of antigens. For tht use  universal fuction:

        def loopPercentPred(MergedDf, MergedDftrans, colName, sumDf):
            for PercentageXX in PercentagesList:  # maybe add here try statemetns to also able to work with 0 samples and 0 olis
                MergedDfExist, MergedDfFC, MergedDfLog, num_olis_before_cutoff, num_olis_after_cutoff = formatDataforPred(
                    MergedDftrans, MergedDf, PercentageXX)

                if  num_olis_after_cutoff == 0:
                    if colName == None:
                        colName = 'noSubgroupSelection'
                    print(str(colName)+': There are 0 peptides after applying the cutoff. Numbers before|after cutoff:' +str(num_olis_before_cutoff)+'|'+str(num_olis_after_cutoff) )

                    #'ClassificationOf2groups-Summary-' + G1name + '+' + G2name + '.csv' # not sure why this is here...

                if  num_olis_after_cutoff > 0:
                    summaryDfPredict = callingClassificaiton(MergedDfExist, MergedDfFC, MergedDfLog, sumDf, G1name, G2name,
                                          G1sampleNo,
                                          G2sampleNo, num_olis_before_cutoff, num_olis_after_cutoff, PercentageXX,
                                          Group1df,
                                          Group2df, colName, ShowFig, SaveFig, SaveGroupSumSeparately, optionalMetadataDf, typeOfFeatureData)
                sumDf = summaryDfPredict

            return summaryDfPredict

        #run on all data
        print('------------------------------------------\nProcessing all peptides without subselection....\n------------------------------------------')
        sumDftoExtend = loopPercentPred(MergedDf, MergedDftrans, None, sumDf)
        sumDftoExtendFilename = ('ClassificationOf2groups-'+G1name+'+'+ G2name+'-' +typeOfFeatureData+'-SummaryOfAllSubgroups')
        sumDftoExtend.to_csv(sumDftoExtendFilename+'.csv')


        # loopign over subgrops
        for colName in interestingCols:
            print(
                '------------------------------------------\nProcessing only peptides of subselection:'+str(colName)+'\n------------------------------------------')
            MergedDfLoop, MergedDftransLoop = subgroupSelection(MergedDf, colName)
            sumDfLoop = loopPercentPred(MergedDfLoop, MergedDftransLoop, colName, sumDf) #possibly rather the summaryDfPredict to get aggreatea?

            sumDftoExtend = sumDftoExtend.combine_first(sumDfLoop)

            # change order of the cols in the df back to the same as the list of groups (do this in the end only, kind of annoyign to do in every loop):
            orderOfinterestingCols = []
            for antigenColName in interestingCols:
                orderOfinterestingCols.extend([antigenColName + '-numOlisPred', antigenColName + '-predictMV', antigenColName + '-predictSD']) #these here must the same names used in the summary df further up

            # iisin does ntow rok for listst....
            #orderOfinterestingCols = orderOfinterestingCols.isin(list(sumDftoExtend)) # to select only the ones appearing so far in the loop. I hope this does not mess up the order...

            orderOfinterestingCols = [item for item in orderOfinterestingCols if item in list(sumDftoExtend)]


            sumDftoExtend = sumDftoExtend[['dataType', 'numberOfIndiviudalsPercent', 'percent']+['noSubSelection-numOlisPred','noSubSelection-predictMV','noSubSelection-predictSD']+orderOfinterestingCols]
            sumDftoExtend.sort_values(['percent', 'dataType'], axis=0, inplace=True) # sorting so printed an plotted in nice order

            sumDftoExtend.to_csv(sumDftoExtendFilename+'.csv')

            #plotting
            colsList = ['noSubSelection']
            for antigenColName in interestingCols:
                colsList.extend([antigenColName])  # these here must the same names used in the summary df further up

            #select only relevant columsn
            orderOfinterestingCols_shortNames = [s for s in orderOfinterestingCols if "-predictMV" in s] # to get rid of other entries
            orderOfinterestingCols_shortNames = [s.replace('-predictMV', '') for s in orderOfinterestingCols_shortNames]# to slect only the first part and get rid of "-predictMV"
            selectionForColsList = [item for item in colsList if item in list(orderOfinterestingCols_shortNames)]
            colsList = ['noSubSelection'] # not ideal to do this way and refien afer bore but donT know better and don'T wanna think about it
            colsList.extend(selectionForColsList)

            fig = plt.figure(figsize=(20, 10))
            ax = plt.subplot()

            stuffer = [*range(len(sumDftoExtend.index))]

            xLables = sumDftoExtend.index.values.tolist()

            # loop over MVList and plot
            xOffsetAndWidth = 0.01
            variableThingy = 0.1
            for columnName in colsList:
                stuffer = [x + variableThingy for x in stuffer]
                MVcolName = sumDftoExtend[columnName + '-predictMV']
                SDcolName = sumDftoExtend[columnName + '-predictSD']
                ax.bar(x=stuffer, height=MVcolName, yerr=SDcolName, width=variableThingy, label=columnName)
                xOffsetAndWidth = xOffsetAndWidth + variableThingy
            # ax.subplots_adjust(bottom=0.2)
            ax.set_xticklabels(xLables, rotation=20, Fontsize=6)
            ax.legend(bbox_to_anchor=(1.0, 1.00), fontsize=6)
            ax.set_ylabel("AUC")
            ax.set_ylim(bottom=0.4)
            ax.set_title(sumDftoExtendFilename)

            plt.savefig(sumDftoExtendFilename+'.png')


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
    PercentagesList = [1, 5, 10, 20, 50, 100] #[1, 5, 10, 20, 50, 100] # [0,1,2,4,6,8,10,20,50,100]if you just want to use one, enter just a single numerbrcent, be carfull, if you want to use this, rather cahnge to a comnined percentat (currntely in each group sepatey), Sigal explained on 2.11.2020, that this can cause issues, miniu nunber in how many people (of one group) an oligo needs to appea for data to be used for PCA
    library = 'Agi'  # Agi or Twi, leave empty '' if you want both;  had to change this to sort on oli cutoff AFTER loading all data. Now always the entire data is loaded and cutoff applied, only then do selcetion for Agi or Twist only nti// Agiletn or Twist???check what the names are to be dropped from loaded data, if none = both (none means here that left empty list '')
    numOlisCutOff = 200  # for exlcusing samples with less than # of oligoes

    # optional export option
    pklOfAllSamples = ''  # put thtere 'y' if you want that. If lowercase yes: saves a pickle with the FcData of all samples per cohrot. if you don't want it, levae emtpy


    ShowFig = ''  # string of 'y' to show the plot
    SaveFig = ''  # string of 'y' to show the plot (by default anyways saving the table, this here depends if you want also every plot)
    SaveGroupSumSeparatelyVar = '' # string of 'y' to save a sepate csv for every group that is predied on

    interestingCols = ['is_PNP','is_patho','is_probio','is_IgA','is_bac_flagella', 'is_IEDB_or_cntrl'] # prefereedred order: ['is_PNP','is_patho','is_probio','is_IgA','is_bac_flagella', 'is_IEDB_or_cntrl']


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

     #classifaiciton
    XGBphageABclassificaiton(Group1df, Group2df, G1name, G2name, PercentagesList, interestingCols, summaryDfPredictOut, ShowFig, SaveFig, SaveGroupSumSeparatelyVar, callingClassificaiton, None, 'phageAbsDataOnly')


    # Print ending time
    print(time.ctime())

    elapsed_time = time.time() - start_time
    # all the following crap just for conerting secs to hours,min,secs
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    print("Finished.")
