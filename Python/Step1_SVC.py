
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
from sklearn.svm import SVC #estimator = SVC(kernel='linear', probability=True)  # kernel='linear' pour un SVM linéaire


# Import other scripts
import Hyperparameters as hp
import modelFunctions as mf

ModelName = 'SVC'

# Input
inputFilename = "./TMJOAI_Long_040422_Norm.csv"
df = pd.read_csv(inputFilename,skiprows=[0])

first_column = mf.convertToArray(df.iloc[:, 0])  
other_columns = mf.convertToArray(df.iloc[:, 1:]  )

NbFold = 10
seed0=2022
np.random.seed(seed0)


skf = StratifiedKFold(n_splits=NbFold, shuffle=True, random_state=seed0)

# stores folds for cross-validation
foldsCVT = list(skf.split(other_columns,first_column))

# Define the model and the hyperparameters
mse_scorer = mt.make_scorer(mt.mean_squared_error, greater_is_better=False) # verify if it's True or False to get the good values

train_control = GridSearchCV(
    estimator=SVC(probability=True), 
    param_grid=hp.param_grid_svm, 
    scoring=mse_scorer,
    n_jobs=-1, # you can add n_jobs=2 to run in parallel on computer with 2 cores or "-1" to use all processors
    refit=True,
    cv=NbFold,
    verbose=0,
) 


# Split the data into folds and separate the training data from the validation data

idx = 0
idx_acc, idx_acc2 = 0,0
acc0 = 0

# Create the folder if it doesn't exist
Outputpath = f'out_{ModelName}/'
if os.path.isdir(Outputpath) == False:
    os.mkdir(Outputpath)

Outpath = f'out_{ModelName}/results/'
if os.path.isdir(Outpath) == False:
    os.mkdir(Outpath)

mf.remove_files_results(NbFold,Outpath)
# Loop over the folds
for train_idx, valid_idx in skf.split(other_columns, first_column): 
    print("idx: ", idx)
    # Training data
    X_train, y_train = [other_columns[i] for i in train_idx], [first_column[i] for i in train_idx]
    # Validation data
    X_valid, y_valid = [other_columns[i] for i in valid_idx], [first_column[i] for i in valid_idx]

    X_train,y_train = mf.convertToArray(X_train), mf.convertToArray(y_train)
    X_valid,y_valid = mf.convertToArray(X_valid), mf.convertToArray(y_valid)

    # Predict on the validation data
    train_control.fit(X_train, y_train)

    y_pred = train_control.predict(X_valid)
    y_scores = train_control.predict_proba(X_valid)[:,1]

    acc, evalHeadList, evalList = mf.evaluation(y_valid,y_pred,y_scores)
    
    mf.write_files(f"{Outputpath}{ModelName}_Performances.csv",evalHeadList,evalList)
    
    # Save the best model
    if acc > acc0:
        idx_acc = idx
        acc0 = acc
        # Save the model into "byte stream"
        pickle.dump(train_control, open(f"{Outputpath}"+f'{ModelName}{idx_acc}'+'.pkl', 'wb'))
        # load the model from pickle: pickled_model = pickle.load(open('model.pkl','rb'))
        #pickled_model.predict(X_test)
        mf.remove_files(NbFold,idx_acc,Outputpath)
    

    # Save the predictions and the results of the 10 folds

    df_predict = pd.DataFrame({'Actual': y_valid, 'Predicted': y_pred})
    df_predict.to_csv(f'{Outpath}predict{idx}.csv', index=False)
    
    df_result = pd.DataFrame(train_control.cv_results_)
    df_result = df_result.sort_values(by=['rank_test_score'])
    df_result.to_csv(f'{Outpath}result{idx}.csv', index=False)
    
    # Save the best parameters of the 10 folds
    best_params = train_control.best_params_

    headerList = ["Index"]
    paramList = [idx]
    for key,values in best_params.items():
        headerList.append(key)
        paramList.append(values)

    mf.write_files(f"{Outputpath}{ModelName}_Best_Parameters.csv",headerList,paramList)


    idx += 1
