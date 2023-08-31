import numpy as np
# dictionnary of hyperparameters according to the model


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 60, stop = 200, num = 30)]
# n_estimators = [100]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(start=60,stop=90, num=10)]
# max_depth = [70]
# Minimum number of samples required to split a node
min_samples_split = [2,5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2]
# Method of selecting samples for training each tree
bootstrap = [True,False]

# Create the random grid
random_grid_rf = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#Glmnet = elasticNet
param_grid_glmnet = {
    'alpha': [int(x) for x in np.linspace(start=0.0001,stop=100.0, num=100)],
    'lambda': [int(x) for x in np.linspace(start=0.0001,stop=1.0, num=10)],
}

#SVM
param_grid_svm = {
    'C': [int(x) for x in np.linspace(start=0.01,stop=10, num=100)],
    'kernel': ['linear', 'rbf','poly','sigmoid','precomputed'],
    'degree': [int(x) for x in np.linspace(start=1,stop=10, num=10)],
    'gamma': ['scale', 'auto'],
}


#LogisticRegression
param_grid_lr = {
    'penalty': ['l1', 'l2','elasticnet',None],         # Penalty type
    'C': [0.01, 0.1, 1.0, 10.0],     # Regularisation parameter
    'solver': ['liblinear', 'lbfgs', 'saga','sag','newton-cg','newton-cholesky'],  # resolution algorithm
    'max_iter': [100, 1000, 2500, 5000],  # Maximum number of iterations
}
