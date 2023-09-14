import os
import sklearn.metrics as mt
import numpy as np
import csv



def convertToArray(list1):
    arr = np.array(list1)
    return arr

def evaluation(y_true,y_pred,y_scores,idx):
    """
    Evaluation of the model
    Input: y_true: Correct answer, y_pred: Predictions from the model, y_scores: Probability of the predictions
    Output: int(Accuracy) + "<Model>_Performances.csv" file
    """
    # Confusion matrix for lists of predictions

    # y_true = convertToArray(y_true)
    # y_pred = convertToArray(y_pred)

    print('y_true:', y_true)    
    print('y_pred:', y_pred)
    
    
    #confusion matrix with sklearn
    tn, fp, fn, tp = mt.confusion_matrix(y_true,y_pred).ravel()
    errors = fp+fn
    accuracy = round(mt.accuracy_score(y_true,y_pred)*100,3)
    precision = round(mt.precision_score(y_true,y_pred,zero_division=np.nan),3)
    recall = round(mt.recall_score(y_true,y_pred),3)
    specificity = round(tn/(tn+fp),3)
    f1 = round(mt.f1_score(y_true,y_pred),3)

    try:
        auc = round(mt.roc_auc_score(y_true,y_scores),3)
    except :
        auc = 'NaN'

    



    print('-----evaluation-----')
    print('Number of errors: ', errors)
    print(f'Accuracy: {accuracy} %')
    print(f'Precision score tp/(tp+fp) : {precision} ') #best value is 1 - worst 0
    print(f'Recall score tp/(fn+tp): {recall} ') # Interpretatiom: High recall score => model good at identifying positive examples
    print(f'Specificity tn/(tn+fp): {specificity} ')
    print(f'F1 : {f1} ') 
    print(f'AUC : {auc} ') # best is 1


    column_name = ['Total','Nb Errors', 'Accuracy', 'Precision', 'Recall','Specificity','F1', 'AUC']
    list_eval = [len(y_pred),errors, accuracy, precision, recall, specificity, f1, auc]


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

def remove_files(filename,NbFold=10, idx_kept=None):
    """ 
    Remove files if they exist
    """
    for i in range(NbFold):
        if os.path.exists(filename) and i !=idx_kept:
            os.remove(filename)

def remove_files_results(NbFold=10, Outputpath='out/'):
    """ 
    Remove files if they exist
    """
    for i in range(NbFold):
        if os.path.exists(f'{Outputpath}predict{i}.csv') :
            os.remove(f'{Outputpath}predict{i}.csv')
        if os.path.exists(f'{Outputpath}result{i}.csv') :
            os.remove(f'{Outputpath}result{i}.csv')

def delete_file(filename):
    """ 
    Remove files if they exist
    """
    if os.path.exists(filename) :
        os.remove(filename)