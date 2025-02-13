# Objective             : Feature Elimination (Reducing the number of features)
# CopyRight (C) 	: Muhammed Cinsdikici ERDEM
# Date                  : 18 November 2023, 19:00
#                       : 16 February 2024, 16:45 (revision 1: Corrected with our dataset)
#                       : 29 February 2024, 16:45 (revision 2: Dataset is renewed and categs are in numeric)
#                       : 07 March    2024, 09:45 (revision 3: SelectKBest and additional methods added)
#                       : 10 March    2024, 23:45 (revision 4: SelectKBest/RFE Modular and Tables are Produced)
#                       : 13 March    2024, 21:45 (revision 5: Lasso/Shallow DL added moduler)
#                       : 27 March    2024, 14:50 (revision 6: Boruta Feature Selection added)
#                       : 30 March    2024, 04:45 (revision 7: MLP is finalized. Results are corrected)
#                       : 05 May      2024, 04:45 (revision 8: Tr/Ts set save)
#                       : 13 May      2024, 01:30 (revision 9: Comfusion_Matrices are obtained for each CLassifier+SelectionMethod)
#                       : 15 May      2024, 16:30 (revision10: Model Summary method is added)
#                       : 16 May      2024, 13:30 (revision11: Sensitivity/Specifity for Each Classifier for Multiclass added)
#                       : 04 July     2024, 15:30 (revision12: The Outputs are prepared for Excel Table with selected feature number
# Code Repo             : ComVIS Lab/ComVISLab_ML
#
# REF:
# [0] https://medium.com/@evertongomede/recursive-feature-elimination-a-powerful-technique-for-feature-selection-in-machine-learning-89b3c2f3c26a
# [1] https://machinelearningmastery.com/rfe-feature-selection-in-python/
"""
The Results for RFE Selected 17 Features using Forest Classifier:
Selected Features: (ranking ==1)  [18 19 25 26 27 28 29 34 35 36 37 38 43 44 45 46 47]
Selected Features: (ranking < 3)  [ 5 18 19 25 26 27 28 29 34 35 36 37 38 43 44 45 46 47]
Accuracy on the Test Set: 0.9545454545454546
Accuracy on the Test Set: 0.961038961038961

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from models.DeepLearningModels import ShallowDL
from models.PDeepLearningModels import ParShallowDL
from models.LSTMPDeepLearningModels import LSTMParShallowDL
from models.CinsConMat import CinsRocRecPrec
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

import sys

# Specify the number of features to select using RFE
if len(sys.argv) < 2:
    print("Command Should be like;")
    print(">>> python3 FeSel240704.py NUM_of_SELECTED_FEATURES")
    exit()
else:
    print ('The Selected Feature Count: ', sys.argv[1])
    num_features_to_select = int(sys.argv[1])


## CREATE RESULTS DIRECTORY
import os
path = './Results_'+sys.argv[1]

# check whether directory already exists
if not os.path.exists(path):
  os.mkdir(path)
  print("Folder %s created!" % path)
else:
  print("Folder %s already exists" % path)
### CRD_ END 


import datetime
gun = datetime.datetime.now()
simdi= gun.strftime("%y")+"_"+gun.strftime("%m")+"_"+gun.strftime("%d")+"_"+gun.strftime("%H")+gun.strftime("%M")
print(simdi)

FigureDir    = path+"/Figures"+simdi
ConfusionDir = path+"/Confusions"+simdi
os.mkdir(FigureDir)
os.mkdir(ConfusionDir)


ResultsFile = path+"/FE_Results_"+simdi+".txt"
with open(ResultsFile, "w") as file:
    file.write(" ************ FEATURE ELIMINATIONs and RESULTS {} *********:  \n".format(simdi))


def Plot_Ranks(NameofMethod,Ranks):
    # Visualize the feature ranking
    plt.figure(figsize=(10, 6))
    plt.title(NameofMethod)
    plt.xlabel("Feature Index")
    plt.ylabel("Ranking")
    plt.bar(range(len(Ranks)), Ranks)
    plt.savefig(FigureDir+"/"+NameofMethod)
    # plt.show()

def only_upper(s):
    return "".join(c for c in s if c.isupper())

def Class_Analysis(CinsDF, DataSetName):

    # Step 1: Check unique values
    unique_classes = CinsDF['Branch'].unique()
    print("Unique classes:", unique_classes)

    # Step 2: Count occurrences
    class_distribution = CinsDF['Branch'].value_counts()
    
    # Plotting
    plt.figure(figsize=(8, 6))
    class_distribution.plot(kind='bar', color='skyblue')
    plt.title('Class Distribution of '+DataSetName)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def Confusion_Analysis(Ytr, Ypr, ModelandFeature):
    print("Confusion_Analysis is starting for ", ModelandFeature)
    if (str(ModelandFeature).find("ShDL") > -1):
        #print("Once Ham Predictionlar")
        #print(Ypr)
        Ypr = np.argmax(Ypr, axis=1)  # Convert probabilities to class labels
        #print("Sonra Duzeltilmis Predictionlar ")
        #print(Ypr)
        #print("Sonra Gercek Test Patternlar ")
        #print(Ytr)
        
    #Model results Ypr is compared with Ytr vector
    cm = confusion_matrix(Ytr, Ypr)
    sensitivity = np.diag(cm) / np.sum(cm, axis=1)
    specificity = []
    for i in range(cm.shape[0]):
        true_negatives = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        false_positives = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(true_negatives / (true_negatives + false_positives))

    # Print sensitivity and specificity for each class
    for i in range(len(sensitivity)):
        print(f"Class {i}: Sensitivity = {sensitivity[i]}, Specificity = {specificity[i]}")

    np.savetxt(ConfusionDir+"/"+ModelandFeature+".csv", cm, fmt='%d;')

    # Step 4: Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.title('Confusion Matrix of '+ ModelandFeature)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(ConfusionDir+"/"+ModelandFeature)
  
    # Step 5: Calculate APPRoximate ROC and Save figure
    rocfig = CinsRocRecPrec(cm)
    rocfig.savefig(ConfusionDir+"/"+"ROC_"+ModelandFeature)
  






SelFeaDic=dict()
ResultDic=dict()


#data   = pd.read_csv("./DataSet_240214/tidml-stage2_max79_HakemDegerlendirmesi_WeightCarpimli.csv")
#data   = pd.read_csv("./DataSet_240214/tidml-stage2_max80_Raw_YalnizcaCihazOlcumleri.csv")
#data   = pd.read_csv("./DataSet_240229/TIDMLStage2_240229.csv")
#data    = pd.read_csv("./DataSet_240229/TIDMLStage2_numericcateg_240229.csv")
data    = pd.read_csv("./DataSet_240229/TIDMLStage2_numericcateg_240404.csv")

X = data.drop(["Branch"],axis=1).values
y = data["Branch"].values

#### Burada Orjinal Set Pattern Dagilimlarini gormek icin ac alt satiri
# Class_Analysis(data,"OrjinalSet")

###########################################################################
# Split the dataset into training and testing sets & WRITE TRAIN.CSV and TEST.CSV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #42 default

dfTrain         = pd.concat([pd.DataFrame(X_train),pd.DataFrame(y_train)],axis=1)
dfTrain.columns = data.columns
dfTrain.to_csv('./DataSet_240229/CinsTrain_'+simdi+'.csv',index=False)
#### Burada Train Set Pattern Dagilimlarini gormek icin ac alt satiri
#Class_Analysis(dfTrain,"TrainSet")

dfTest         = pd.concat([pd.DataFrame(X_test),pd.DataFrame(y_test)],axis=1)
dfTest.columns = data.columns
dfTest.to_csv('./DataSet_240229/CinsTest_'+simdi+'.csv',index=False)
#### Burada Test Set Pattern Dagilimlarini gormek icin ac alt satiri
#Class_Analysis(dfTest, "TestSet")
###########################################################################

#input("Class Analizini incele....")


scal = MinMaxScaler()  # Alternatively use scal=StandardScaler()

X_normalized_train = scal.fit(X_train)
X_normalized_train = scal.transform(X_train)


# Initialise the subplot function using number of rows and columns 
#figure, axis = plt.subplots(2, 3) 


############# SelectKBest: BEGIN  ############################
#apply SelectKBest class to extract top "#Selected" best features
# Selection method of Statistical Based SelectKBest uses ANOVA, Chi2, Mutual Information (MI)
"""
f_classif

    ANOVA F-value between label/feature for classification tasks.
mutual_info_classif

    Mutual information for a discrete target.
chi2

    Chi-squared stats of non-negative features for classification tasks
"""
def SelKBestF(methodname):
    if methodname == chi2:
        methn = "chi2"
    if methodname == f_classif:
        methn = "ANOVA"
    if methodname == mutual_info_classif:
        methn = "MI"

    bestfeatures = SelectKBest(score_func=methodname, k=num_features_to_select)
    # In the belov code, Chi2 method needs Normalized X_train. -In contrary, it gives negative value error-

    fit = bestfeatures.fit(X_normalized_train,y_train)

    dfscores  = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(data.drop(["Branch"],axis=1).columns)

    #concat two dataframes for better visualization 
    featureScores         = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['SelKBest Specs'+methn,'SelKBestScore'+methn]  #naming the dataframe columns

    bestnfeatures     = featureScores.nlargest(num_features_to_select,'SelKBestScore'+methn)  #print #num selected best features
    SelFeaDic["SelKBest_"+methn]=bestnfeatures.index  #print #num selected best features


    print("SelectKBest "+methn+ " (n_largest are selected)",fit.scores_)
    Plot_Ranks("SelectKBest "+methn + " (n_largest are selected)",fit.scores_)
    ############# SelectKBest: END  ############################

SelKBestF(chi2)
SelKBestF(f_classif)
SelKBestF(mutual_info_classif)



############# Lasso: BEGIN  ############################
lasso = Lasso(alpha=0.01)

#Used Normalized inputs (We are using MinMax Scale /Alternatively StandardScaler)
fit = lasso.fit(X_normalized_train,y_train)
#  Retrieve coefficients and select top num_features_to_select
lassoScores = pd.DataFrame(abs(fit.coef_))
lassoColumns= pd.DataFrame(data.drop(['Branch'],axis=1).columns)

LassofeatureScores         = pd.concat([lassoColumns,lassoScores],axis=1)
LassofeatureScores.columns = ['Lasso_Specs','Lasso_Scores']

bestnlassofeatures =LassofeatureScores.nlargest(num_features_to_select,'Lasso_Scores')
SelFeaDic["LassoKBest"]=bestnlassofeatures.index  #print #num selected best features

Plot_Ranks("Lasso_KBest (n_largest are selected)",abs(fit.coef_))
############# Lasso: END    ############################



############# Boruta: BEGIN  ############################
# NUMPY VERSION SHOULD BE 1.23.1

# Step 1: Data is imported
# Original X_train and y_train is used

from boruta import BorutaPy

#Step 2: RF Classifier is selected
model_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced', max_depth=7)

# Step 3: Create an instance of Boruta
boruta_selector = BorutaPy(model_rf, n_estimators='auto', verbose=5, random_state=1)
# Step 4: Fit Boruta to your data
fit = boruta_selector.fit(X_normalized_train, y_train)

boruScores  = pd.DataFrame(fit.ranking_)
boruColumns = pd.DataFrame(data.drop(["Branch"],axis=1).columns)
BorutafeatureScores         = pd.concat([boruColumns,boruScores],axis=1)
BorutafeatureScores.columns = ['Boruta_Specs','Boruta_Scores']


# print("Boruta Selector:", boruta_selector.support_)
# Step 5: Retrieve selected features
# BORUTA Selector Support only takes Ranking = 1 
# selected_features = np.where(boruta_selector.support_)[0]
# Step 6: Print selected features
# print("Selected Features:", selected_features)

bestnfeatures        = BorutafeatureScores.nsmallest(num_features_to_select,'Boruta_Scores')  #print #num selected best featur>
SelFeaDic["BorutaKBest_RF"]= bestnfeatures.index  #print #num selected best features

Plot_Ranks("BorutaKBest (n_largest are selected)",fit.ranking_)
############# Boruta: END  ############################



###### mRMR Feature Selection: BEGIN ######
"""
from mrmr import mrmr_classif

print("MRMR'a Hosgeldin..")
bestmrmrfeatures = mrmr_classif(X_normalized_train, y_train, K=num_features_to_select)
print(bestmrmrfeatures)
"""
###### mRMR Feature Selection: END ######




######## Recursive Feature Elimination: BEGIN  ######

models     = []
modelnames = []

#### Random Forest CLASSIFIER MODEL
# Create a Random Forest Classifier as the base model
models.append( RandomForestClassifier(n_estimators=100, max_depth=7, random_state=23))
modelnames.append('RFC')

#### MLP CLASSIFIER MODEL
# Create a MLP Classifier as the base model2
# models.append( MLPClassifier(random_state=18, max_iter=300))


#### ExtraTrees CLASSIFIER MODEL
models.append(ExtraTreesClassifier())
modelnames.append('ETC')

####  SVM-Linear classifier 
models.append(SVC(kernel = 'linear', C = 10))
modelnames.append('SVC')

####  SVM-Kernel Based classifier 
#models.append(SVC(kernel = 'rbf', gamma=0.1, C = 10))

#### Decision Tree classifier 
models.append( DecisionTreeClassifier(max_depth = 7))
modelnames.append('DTC')



##### KNN classifier 
# models.append (KNeighborsClassifier(n_neighbors = 7))
 



# print(models)

for model_, modeln_ in zip(models,modelnames):
    # Initialize RFE with the model and the number of features to select
    rfe = RFE(estimator=model_, n_features_to_select=num_features_to_select)
    # Fit RFE to the training data
    # rfe.fit(X_train, y_train)
  
    rfe.fit(X_normalized_train, y_train)

    # Get the ranking of each feature
    feature_ranking = rfe.ranking_

    Plot_Ranks("RFE "+ str(model_),feature_ranking)

    df_feature_ranking = pd.DataFrame(feature_ranking)
    df_columns = pd.DataFrame(data.drop(["Branch"],axis=1).columns)

    #concat two dataframes for better visualization 
    print("#####>",modeln_,"<#####")
    ModelAbbr= modeln_
    featureRankings         = pd.concat([df_columns,df_feature_ranking],axis=1)
    featureRankings.columns = [ModelAbbr+"BestSpecs",ModelAbbr+"BestRankings"]  #naming the dataframe columns

    # Selected Features  are the features we want to use  ex: 17 (ranking ==1 )
    # Selected Features2 are the features additional features are in the list  ex: 18 (ranking<3)
    selected_features  = featureRankings.nsmallest(num_features_to_select,ModelAbbr+'BestRankings').index
    selected_features2 = featureRankings[featureRankings[(ModelAbbr+"BestRankings")]<3].index
    #print(selected_features)   # n_features_selected smallest rankings are taken
    #print(selected_features2)   # n_features_selected smallest rankings are taken

    SelFeaDic["RFE_"+ModelAbbr]=selected_features  #print #num selected best features
    
    print(featureRankings.nsmallest(num_features_to_select,ModelAbbr+'BestRankings'))
   

### ALL Selected Features with Their Method Names
# print (SelFeaDic)

with open(ResultsFile,"w") as file:
    file.write("===== SELECTED FEATURES with THEIR SELECTION METHOD ======\n")

list_shared=[]
list_union =[]
search_i = True
for key in SelFeaDic.keys():
    list_tmp = SelFeaDic[key]

    if search_i:
        list_shared = list_tmp
        list_union  = list_tmp
        search_i = False
    else:
        list_shared= list(set(list_shared).intersection(list_tmp))
        list_union = list(set(list_union) | set(list_tmp))

    with open(ResultsFile,"a") as file:
        file.write("Selected Features of {}: {}\n".format(str(key), str(SelFeaDic[key])))

    print(key, ":", SelFeaDic[key])

print("Ortak Feature Set:", str(list_shared))
with open(ResultsFile,"a") as file:
    file.write("Ortak   Feature Set: {}\n".format(str(list_shared)))
    file.write("Bileske Feature Set: {}\n".format(str(list_union)))


"""
with open(ResultsFile, "a") as file:
   file.write("===== SELECTED FEATURES with THEIR SELECTION METHOD ======")
   for key, value in SelFeaDic.keys():
       file.write(key, ":", value)
"""

# To this point we have added models both for as Recursive Feature Elimination Methods and as CLASSIFIER
# From now on, we are going to add other models as CLASSIFIERS.


# Define the parameters for the neural network
# Initialize the custom neural network
def KendiModelimiz():
    bizim_model= ShallowDL(layer_units=[128,128,64],   #En iyi config
                           activations=['sigmoid','sigmoid','sigmoid'], #tanh is alternative
                           input_dim=num_features_to_select, 
                           output_dim=6,
                           Optmzr='adam', #rmsprop is alternative
                           Lossf='sparse_categorical_crossentropy',
                           epochs=200,
                           batch_size=16,
                           validation_split=0.1,
                           learning_rate=0.00001,
			   deepname='ShallowDL')
    bizim_model.build_model()
    bizim_model.compile_model(bizim_model.Optmzr, bizim_model.learning_rate, bizim_model.Lossf)
    bizim_model.summary()
   
    return bizim_model


#models.append(KendiModelimiz())
#modelnames.append("ShDL")


def ParKendiModelimiz():
    bizim_model= ParShallowDL(layer_units=[[128,128,64], [128,128,64], [128,128,64]],
                          activations=[['sigmoid','sigmoid','sigmoid'],
                                       ['sigmoid','sigmoid','sigmoid'],
                                       ['sigmoid','sigmoid','sigmoid']],
                          input_dim=num_features_to_select,
                          output_dim=6,
                          Optmzr='adam',
                          Lossf='sparse_categorical_crossentropy',
                          epochs=200,
                          batch_size=16,
                          validation_split=0.1,
                          learning_rate=0.00001,
			              deepname='ParShallowDL')
    bizim_model.build_model()
    bizim_model.compile_model(bizim_model.Optmzr, bizim_model.learning_rate, bizim_model.Lossf)
    bizim_model.summary()
   
    return bizim_model

models.append(ParKendiModelimiz())
modelnames.append("ParShDL")


def LSTMParKendiModelimiz():

    bizim_model= LSTMParShallowDL(layer_units=[[128,128,128], [128,128,128], [128,128,128]],
                          activations=[['relu','relu','relu'],
                                       ['relu','relu','relu'],
                                       ['relu','relu','relu']],
                          num_timesteps=1,
                          input_dim=num_features_to_select,
                          output_dim=6,
                          Optmzr='adam',
                          Lossf='sparse_categorical_crossentropy',
                          epochs=200,
                          batch_size=16,
                          validation_split=0.1,
                          learning_rate=0.00001,
			              deepname='LSTMParShallowDL')
    bizim_model.build_model()
    bizim_model.compile_model(bizim_model.Optmzr, bizim_model.learning_rate, bizim_model.Lossf)
    bizim_model.summary()
   
    return bizim_model

#models.append(LSTMParKendiModelimiz())
#modelnames.append("LSTMParShDL")



#print (models)

with open(ResultsFile, "a") as file:
    file.write("Classifier:        FeatureSelector:      Accuracy:  \n")

### Now Selected Features Are used with Classifiers to See the Accuracies...
for modelim, modelimn in zip(models,modelnames):
    print ("-----------CLASSIFIER: "+ modelimn+ "------------------")
    print ("Selected Features Count:", num_features_to_select)

    for eachitem in SelFeaDic:
        print ("Classifier: "+modelimn+"Feature Selection Method:" + eachitem)
        ###### FIRST TRAIN WITH SELECTED FEATURES ########
        # Train the final model using the selected features
        
        print(hasattr(modelim,'deepname'), modelimn)
        
        if hasattr(modelim, 'deepname') and (modelimn =='LSTMParShDL'): #Both Shallow and LSTMParShallow falls in here

            num_timesteps=1
            #X_reshaped = modelim.create_rolling_window_sequences(X_train[:, SelFeaDic[eachitem]], num_timesteps)
            X_reshaped = X_train[:, SelFeaDic[eachitem]]
            X_reshaped = X_reshaped.reshape((len(X_reshaped), 1, modelim.input_dim))

            print("LENNN", X_reshaped.shape)
            modelim.train(X_reshaped, y_train, modelim.epochs, modelim.batch_size,modelim.validation_split)
            #Xt_reshaped = modelim.create_rolling_window_sequences(X_test[:, SelFeaDic[eachitem]], num_timesteps)
            
            Xt_reshaped = X_test[:, SelFeaDic[eachitem]]
            Xt_reshaped = Xt_reshaped.reshape((len(Xt_reshaped), 1, modelim.input_dim))
            loss, accuracy = modelim.evaluate(Xt_reshaped, y_test)

            # To construct Confusion Matrix we have to get predicted classes

            y_pred = modelim.predict(Xt_reshaped)

            Confusion_Analysis(y_test, y_pred,modelimn+eachitem)            
        elif (hasattr(modelim,'deepname') and ((modelimn=='ParShDL') or (modelimn=='ShDL'))):
        #if (hasattr(modelim,'deepname')):          
           modelim.train(X_train[:, SelFeaDic[eachitem]], y_train, modelim.epochs, modelim.batch_size,modelim.validation_split)
           loss, accuracy = modelim.evaluate(X_test[:, SelFeaDic[eachitem]], y_test)
           # To construct Confusion Matrix we have to get predicted classes
           y_pred = modelim.predict(X_test[:, SelFeaDic[eachitem]])
           Confusion_Analysis(y_test, y_pred,modelimn+eachitem)            
        else:
            modelim.fit(X_train[:, SelFeaDic[eachitem]], y_train)
            accuracy = modelim.score(X_test[:, SelFeaDic[eachitem]], y_test)
            # To construct Confusion Matrix we have to get predicted classes
            y_pred = modelim.predict(X_test[:, SelFeaDic[eachitem]])
            Confusion_Analysis(y_test, y_pred,modelimn+"_"+eachitem)

        # To understand the effect of selected feature on TRaining Set use the wrong accuracy chcking below
        # accuracy = model.score(X_train[:, selected_features], y_train)
        print("Accuracy on Test:", accuracy, "\n")
        with open(ResultsFile, "a") as file:
            file.write("Classifier: {}  FeatureSelector: {},  Accuracy: {} \n".format(modelimn,eachitem ,accuracy))
            if not (modelimn in ResultDic.keys()):
                ResultDic[modelimn]=[]
            ResultDic[modelimn].append(accuracy)
        ###### OPTIONALLY TRAIN WITH EXPANDED SELECTED FEATURES (such as Ranking < 3) ########
        # Train the final model using the selected features
        # model_.fit(X_train[:, selected_features2], y_train)
        # Evaluate the model on the test set
        # accuracy2 = model_.score(X_test[:, selected_features2], y_test)

        # To understand the effect of selected feature on TRaining Set use the wrong accuracy chcking below
        #accuracy2 = model.score(X_train[:, selected_features2], y_train)
        # print("Accuracy on the Test Set with EXPANDED SELECTED FEATURES (with Ranking<3):", accuracy2)

ResultDF = pd.DataFrame(ResultDic)
# ResultDF.columns=["Classifier",SelFeaDic.keys]
ResultDF.insert(0, "FeatureSelector", SelFeaDic.keys(), True)
ResultDF.to_csv(path+"/ResultDF_"+str(num_features_to_select)+"_Features_"+simdi+".csv",sep=";")
