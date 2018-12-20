import numpy as np 
import pandas as pd
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, roc_curve, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from rfpimp import *
from imblearn.over_sampling import SMOTE
pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 2000)

class Dummify():
    ''' A class to one-hot specific columns of a dataframe and merge one another specific set of columns
    '''
    def __init__(self, df, cols_to_encode, cols_to_merge):
        self.df = df
        self.cols_to_encode = cols_to_encode
        self.cols_to_merge = cols_to_merge
    def get(self):
        self.dums = pd.get_dummies(self.df, drop_first=True)
    def swap(self):
        self.df.drop(self.cols_to_encode, axis=1, inplace=True)
        X = pd.merge(self.df, self.dums, on=self.cols_to_merge)
        return X

def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])

def print_roc_curve(y_test, probabilities, clf_type):
    '''
    Calculates and prints a ROC curve given a set of test classes and probabilities from a trained classifier
    '''
    tprs, fprs, thresh = roc_curve(y_test, probabilities)
    plt.figure(figsize=(12,10))
    plt.plot(fprs, tprs, 
         label=clf_type, 
         color='red')
    plt.plot([0,1],[0,1], 'k:')
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve AUC: {} Recall: {}".format(roc_auc, recall))
    plt.show()

def print_confusion_matrix(conf_mat, model_name, c_map):
    '''
    Prints a formatted confusion matrix as a Seaborn heatmap with appropriate labels and titles.
    
    Parameters:
    ----------
    conf_mat: sklearn confusion matrix of classifier output
    model_name: name of model, for printing plot label
    
    '''
    plt.figure(figsize=(10,6))
    sns.heatmap(conf_mat, annot=conf_mat, cmap=c_map, fmt='d', annot_kws={"size": 40})
    plt.xlabel('Predicted No Click        Predicted Click', fontsize=25)
    plt.ylabel('Actual Click      Actual No Click', fontsize=25)
    plt.title('{} Confusion Matrix'.format(model_name), fontsize=40)
    plt.show()

######################################################################

if __name__ == "__main__":
    
    # import the data
    df = pd.read_csv('data/joined_data.csv')
    X = df.drop(["opened", "clicked"], axis=1)
    y = df["clicked"]

    # define columns for one-hot encoding and merging back
    drop_cols = ['email_text', 'email_version', 'weekday', 'user_country']
    merge_cols = ['email_id', 'hour', 'user_past_purchases']

    # dummify
    d = Dummify(X, drop_cols, merge_cols)
    d.get()
    X = d.swap()

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,   random_state=42)

    # smote
    cols = X_train.columns
    sm = SMOTE(random_state=66) # oversample minority class
    X_train, y_train = sm.fit_resample(X_train, y_train)
    X_train = pd.DataFrame(data=X_train, columns=list(cols)) # return to dataframe

    # svc = SVC() # recall 0.14
    knn = KNeighborsClassifier()

    params = {
            "n_neighbors": [3, 5, 7],
            "p": [1, 2],
    }
    gscv = GridSearchCV(knn, 
            param_grid=params, 
            scoring='recall', 
            cv=5,
            n_jobs=-1,
            verbose=1)

    clf = gscv.fit(X_train, y_train)
    
    '''
    # best model as determined by gridsearch
    clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=1,
           weights='uniform')

    clf.fit(X_train, y_train)
    
    # initial evaluation
    print('Best parameters: {}'.format(clf.best_params_))
    cv = cross_validate(clf, X_train, y_train, cv=5, n_jobs=-1, scoring="recall")
    print(cv)

    roc_auc_cv = (cross_val_score(clf, X_train, y_train, scoring = 'roc_auc', cv=5))
    recall_cv = cross_val_score(clf, X_train, y_train, scoring = 'recall', cv=5)
    precision_cv = cross_val_score(clf, X_train, y_train, scoring = 'precision', cv=5)
    accuracy_cv = cross_val_score(clf, X_train, y_train, scoring = 'accuracy', cv=5)
    f1_cv = cross_val_score(clf, X_train, y_train, scoring = 'f1_micro', cv=5)

    print('Best clf: {}'.format(clf))
    print('Best clf parameters: {}'.format(clf.best_params_))
    print('Roc Auc: {}'.format(roc_auc_cv))
    print('Recall Score: {}'.format(recall_cv))
    print('Precision Score: {}'.format(precision_cv))
    print('Accuracy Score: {}'.format(accuracy_cv))
    print('F1 Micro: {}'.format(f1_cv))
    

    # clf = pickle.load(open('email_model.p', 'rb'))

    # final clf evaluation
    predictions = clf.predict(X_test)
    roc_auc = roc_auc_score(y_test, predictions)
    probas = clf.predict_proba(X_test)[:, :1]
    tprs, fprs, thresh = roc_curve(y_test, probas)
    recall = recall_score(y_test, predictions)
    conf_mat = standard_confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    print_roc_curve(y_test, probas, 'Random Forest')
    print_confusion_matrix(conf_mat, "RF", "viridis")
    print('Best clf: {}'.format(clf))
    print('\nRoc Auc: {}'.format(roc_auc))
    print('\nRecall Score: {}'.format(recall))
    print('\nClassification Report:\n {}'.format(class_report))
    print('\nConfusion Matrix:\n {}'.format(standard_confusion_matrix(y_test, predictions)))
    '''