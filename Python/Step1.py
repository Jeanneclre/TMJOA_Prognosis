import csv
import subprocess
import sys
import os
import numpy as np
import pandas as pd
import pickle

try : 
    import sklearn
except ImportError:
    print("sklearn not found")
    python = sys.executable
    subprocess.check_call([python,'pip', 'install', 'sklearn'], stdout=subprocess.DEVNULL)
    import sklearn

from sklearn import metrics as mt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
# Import the model:
from sklearn.ensemble import RandomForestClassifier


# Import other scripts
import Hyperparameters as hp


def convertToArray(list1):
    arr = np.array(list1)
    return arr

def evaluation(y_valid,y_pred,y_scores):
    """
    Evaluation of the model
    Input: y_valid: Correct answer, y_pred: Predictions from the model, y_scores: Probability of the predictions
    Output: int(Accuracy) + "<Model>_Performances.csv" file
    """
    
    tn,fp,fn,tp = mt.confusion_matrix(y_valid,y_pred).ravel()  
   
    errors = fp+fn
    mse = mt.mean_squared_error(y_valid.astype(int), y_pred.astype(int))
    accuracy = mt.accuracy_score(y_valid,y_pred)*100
    precision = mt.precision_score(y_valid,y_pred)
    recall = mt.recall_score(y_valid,y_pred)
    specificity = tn/(tn+fp)
    f1 = mt.f1_score(y_valid,y_pred)
    auc = mt.roc_auc_score(y_valid,y_scores)
    

    print('-----Model Performance----')
    print('Number of errors: ', errors)
    # print(f'Accuracy : {accuracy} %')
    print(f'Accuracy: {accuracy} %')
    
    print(f'Precision score tp/(tp+fp) : {precision} ') #best value is 1 - worst 0
    print(f'Recall score tp/(fn+tp): {recall} ') # Interpretatiom: High recall score => model good at identifying positive examples
    print(f'Specificity tn/(tn+fp): {specificity} ')
    print(f'F1 : {f1} ') 
    print(f'AUC : {auc} ') # best is 1
    print("Positive MSE :", mse)
    print("---------------------------")

    column_name = ['Nb Errors', 'Accuracy', 'Precision', 'Recall',"Specificity" 'F1', 'AUC', 'MSE']
    list_eval = [errors, accuracy, precision, recall, specificity, f1, auc, mse]
  

    return accuracy,column_name,list_eval

def write_files(filename,listHeader,listParam):
    """
    Write the results in a csv file
    Input: filename: Name of the file, listHeader: List of the header, listParam: List of the parameters
    Output: "<filename>.csv" file
    """
    existing_file = os.path.exists(filename)
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        if not existing_file:
            csvwriter.writerow(listHeader)

        csvwriter.writerow(listParam)

def remove_files(NbFold=10, idx_kept=None,outputpath='out'):
    """ 
    Remove files if they exist
    """
    for i in range(NbFold):
        if os.path.exists(f'{outputpath}/RF{i}.pkl') and i !=idx_kept:
            os.remove(f'{outputpath}/RF{i}.pkl')

def remove_files_results(NbFold=10,outputpath='out'):
    """ 
    Remove files if they exist
    """
    for i in range(NbFold):
        if os.path.exists(f'{outputpath}/predict{i}.csv') :
            os.remove(f'{outputpath}/predict{i}.csv')
        if os.path.exists(f'{outputpath}/result{i}.csv') :
            os.remove(f'{outputpath}/result{i}.csv')

iii=0 # First index in python is 0


method_list = ["lasso", "ridge", "svc", "random_forest", "xgboost", "lda", "neural_network", "gradient_boosting"]


Nfeature_selection = [1, 3, 4, 5, 6, 7]
Npredictive_modeling = list(range(1, 9)) 

# Generate every possible combination of the two lists
vecT = [(val1, val2) for val2 in Npredictive_modeling for val1 in Nfeature_selection ]

i1 = vecT[iii][0]
i2 = vecT[iii][1]


# Input
inputFilename = "./TMJOAI_Long_040422_Norm.csv"
df = pd.read_csv(inputFilename,skiprows=[0])

first_column = convertToArray(df.iloc[:, 0])  
other_columns = convertToArray(df.iloc[:, 1:]  )

NbFold = 10

seed0=2022
np.random.seed(seed0)


skf = StratifiedKFold(n_splits=NbFold, shuffle=True, random_state=seed0)

# stores folds for cross-validation
foldsCVT = list(skf.split(other_columns,first_column))

# Define the model and the hyperparameters
mse_scorer = mt.make_scorer(mt.mean_squared_error, greater_is_better=False) # verify if it's True or False to get the good values

train_control = GridSearchCV(
    estimator=RandomForestClassifier(), 
    param_grid=hp.random_grid_rf, 
    scoring=mse_scorer,
    n_jobs=-1, # you can add n_jobs=2 to run in parallel on computer with 2 cores or "-1" to use all processors
    refit=True,
    cv=NbFold,
    verbose=0,
) 


# Split the data into folds and separate the training data from the validation data
param_values_depth = []
param_values_estimator = []
param_values_leaf=[]
param_values_split =[]
accuracy = []
idx = 0
idx_acc, idx_acc2 = 0,0
acc0 = 0

# Create the folder if it doesn't exist
Outputpath = 'out_RF'
if os.path.isdir(Outputpath) == False:
    os.mkdir(Outputpath)

Outpath = 'out_RF/results'
if os.path.isdir(Outpath) == False:
    os.mkdir(Outpath)

remove_files_results(NbFold,Outpath)
# Loop over the folds
for train_idx, valid_idx in skf.split(other_columns, first_column): 
    print("idx: ", idx)
    # Training data
    X_train, y_train = [other_columns[i] for i in train_idx], [first_column[i] for i in train_idx]
    # Validation data
    X_valid, y_valid = [other_columns[i] for i in valid_idx], [first_column[i] for i in valid_idx]

    X_train,y_train = convertToArray(X_train), convertToArray(y_train)
    X_valid,y_valid = convertToArray(X_valid), convertToArray(y_valid)

    # Predict on the validation data
    train_control.fit(X_train, y_train)

    y_pred = train_control.predict(X_valid)
    y_scores = train_control.predict_proba(X_valid)[:,1]

    acc, evalHeadList, evalList = evaluation(y_valid,y_pred,y_scores)
    
    write_files(f"{Outputpath}/RF_Performances.csv",evalHeadList,evalList)
    
    # Save the best model
    if acc > acc0:
        idx_acc = idx
        acc0 = acc
        # Save the model into "byte stream
        pickle.dump(train_control, open(f"{Outputpath}/"+f'RF{idx_acc}'+'.pkl', 'wb'))
        # load the model from pickle: pickled_model = pickle.load(open('model.pkl','rb'))
        #pickled_model.predict(X_test)
        remove_files(NbFold,idx_acc,Outputpath)
    

    # Save the predictions and the results of the 10 folds

    df_predict = pd.DataFrame({'Actual': y_valid, 'Predicted': y_pred})
    df_predict.to_csv(f'{Outpath}/predict{idx}.csv', index=False)
    
    df_result = pd.DataFrame(train_control.cv_results_)
    df_result = df_result.sort_values(by=['rank_test_score'])
    df_result.to_csv(f'{Outpath}/result{idx}.csv', index=False)
    
    # Save the best parameters of the 10 folds
    best_params = train_control.best_params_

    headerList = ["Index"]
    paramList = [idx]
    for key,values in best_params.items():
        headerList.append(key)
        paramList.append(values)

    write_files(f"{Outputpath}/RF_Best_Parameters.csv",headerList,paramList)


    idx += 1




    