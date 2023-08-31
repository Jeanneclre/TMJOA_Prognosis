import os
import sklearn.metrics as mt
import numpy as np
import csv



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

    column_name = ['Nb Errors', 'Accuracy', 'Precision', 'Recall','Specificity','F1', 'AUC', 'MSE']
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

def remove_files(NbFold=10, idx_kept=None, outputpath='out/'):
    """ 
    Remove files if they exist
    """
    for i in range(NbFold):
        if os.path.exists(f'{outputpath}RF{i}.pkl') and i !=idx_kept:
            os.remove(f'{outputpath}RF{i}.pkl')

def remove_files_results(NbFold=10, Outputpath='out/'):
    """ 
    Remove files if they exist
    """
    for i in range(NbFold):
        if os.path.exists(f'{Outputpath}predict{i}.csv') :
            os.remove(f'{Outputpath}predict{i}.csv')
        if os.path.exists(f'{Outputpath}result{i}.csv') :
            os.remove(f'{Outputpath}result{i}.csv')