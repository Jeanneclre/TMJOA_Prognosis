
#################################################
##        Contributor: Jeanne CLARET           ##
##                                             ##
##                 STEP1 RF  v2                ##
##                  Nested CV                  ##
#################################################

import subprocess
import sys
import os
import numpy as np
import pandas as pd
import pickle

try :
    import sklearn
except :
    print("imblearn not found")
    python = sys.executable
    subprocess.check_call([python,'pip', 'install', 'sklearn'], stdout=subprocess.DEVNULL)
    import sklearn

from sklearn import metrics as mt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import different models:
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import other scripts 
import Hyperparameters as hp
import modelFunctions as mf



# Import Input Data (csv file)
inputFilename = "./TMJOAI_Long_040422_Norm.csv"
df = pd.read_csv(inputFilename,skiprows=[0])

first_column = mf.convertToArray(df.iloc[:, 0])  
other_columns = mf.convertToArray(df.iloc[:, 1:])

y= first_column
X= other_columns
# Initialize the model
model = LogisticRegression()
feature = RandomForestClassifier()
# pipeline 
pipe = Pipeline([('std', StandardScaler()),
                  ('lr', model)])

# setting param grid
param_grid = hp.param_grid_lr

# Define the model and the hyperparameters
mse_scorer = mt.make_scorer(mt.mean_squared_error, greater_is_better=False) # verify if it's True or False to get the good values

# Nested CV with parameter optimization
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Setting up multiple GridSearchCV objects, 1 for each algorithm
idx = 0
nested_scores = []
outer_scores = []

# Hyperparameters Tuning - Inner Loop
gcv = GridSearchCV(estimator=model,
                        param_grid=hp.random_grid_lr,
                        scoring='accuracy',
                        n_jobs=-1,
                        cv=inner_cv,
                        verbose=0,
                        refit=True)

for train_idx, test_idx in outer_cv.split(X,y):
    idx+=1
    
    nested_score = cross_val_score(gcv, X[train_idx], y[train_idx], cv=outer_cv)
    nested_scores.append(nested_score.mean())

    gcv.fit(X[train_idx], y[train_idx]) # run inner loop hyperparam tuning
    
    # perf on test fold (test_idx)
    outer_scores.append(gcv.best_estimator_.score(X[test_idx], y[test_idx]))

    y_pred = gcv.predict(X[test_idx])
    y_scores = gcv.predict_proba(X[test_idx])[:,1]

    

    print(f'-----------------Model Performance {idx} -------------------')
    acc, evalHeadList, evalList = mf.evaluation(y[test_idx],y_pred,y_scores)

    print('========outside evaluation ========')
    print(f'Best ACC (avg. of inner test folds): {gcv.best_score_ * 100}')
    # print('Best parameters:', gcv.best_params_)
    print('ACC (on outer test fold): %.2f%%' % (outer_scores[-1]*100))
    print('Nested Score : ', round(nested_scores[-1]*100,2))
    print('\n')