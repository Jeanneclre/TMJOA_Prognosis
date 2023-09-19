import numpy as np
# dictionnary of hyperparameters according to the model


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 60, stop = 200, num = 30)]
# n_estimators = [100]
# Number of features to consider at every split
max_features = ['sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(start=60,stop=90, num=10)]
# max_depth = [70]
# Minimum number of samples required to split a node
min_samples_split = [2,5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2]
# Method of selecting samples for training each tree
bootstrap = [True,False]

# For glmboost
double_learning_rate = [float(x) for x in np.linspace(start=0.01,stop=1.0, num=40)]

#Glmnet = elasticNet
param_grid_glmnet = {
    'alpha': [float(x) for x in np.linspace(start=0.0001,stop=100.0, num=100)],
    'l1_ratio': [float(x) for x in np.linspace(start=0.001,stop=1.0, num=20)],
}

#LogisticRegression --> glmnet 
param_grid_lr = {
    'penalty': ['l1'],         # Penalty type
    'C': [0.01, 0.1, 1.0, 10.0],     # Regularisation parameter
    'solver': ['liblinear'],  # resolution algorithm
    'max_iter': [ 5000],  # Maximum number of iterations
}

#SVM
param_grid_svm = {
    'C': [float(x) for x in np.linspace(start=0.01,stop=10, num=50)],
    'kernel': ['linear', 'rbf','poly','sigmoid'], #'precomputed' can only be used when passing a 9n_samples, n_samples) data matrix
    'degree': [int(x) for x in np.linspace(start=1,stop=10, num=10)],
    'gamma': ['scale', 'auto'],
}


# Random forest
param_grid_rf = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# XGBtree - XGBClassifier()
param_grid_xgb = {
    'n_estimators': [int(x) for x in np.linspace(start=40,stop=200, num=25)],
    'learning_rate': [float(x) for x in np.linspace(start=0.01,stop=1, num=20)],
    'max_depth': max_depth,
    'subsample': [0.5,0.75,1.0],
}


# LDA
param_grid_lda = {
    'solver': ['svd','lsqr'],
    'shrinkage': [None],
    'n_components': [1],
}

# Neural Network
param_grid_nnet = {
    'hidden_layer_sizes': [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': [10000],
}

# Glmboost
param_grid_glmboost = {
    'loss': ['log_loss', 'exponential'],
    'learning_rate': [float(x) for x in np.linspace(start=0.01,stop=1, num=20)],
    'n_estimators': [int(x) for x in np.linspace(start=40,stop=200, num=25)],
    'subsample': [0.5,0.75,1.0],
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'max_features': max_features,

}

# HDDA -- high-dimensional discriminant analysis PCA
param_grid_pca = {
    'n_components': [1, 2, 3, 4, 5],
    'whiten': [True, False],
}

param_grid_hdda = {
    'C' : [int(x) for x in np.linspace(start=1,stop=10, num=10)],
    'th': [float(x) for x in np.linspace(start=0.01,stop=1, num=20)],
    'model' : ['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10'],
    
    }