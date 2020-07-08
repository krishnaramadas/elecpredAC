#-*- coding: utf-8 -*-
"""
Created on Sat Jun 27 20:30:39 2020

@author: Krishna
"""
'''Data is derived from STATISTICAL REPORT OF ELECTION RESULTS pdf files downloaded from the election comission of India.'''
'''They are converted into csv files and only the DETAILED RESULT part of the report is used for analysis.'''

#Importing necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

'''Data Preparation'''

#Path to your csv file directory
mycsvdir = 'D:/ElectionAgeCat-prediction/Data'

#Fetching all the csv files in that directory
csvfiles = glob.glob(os.path.join(mycsvdir, '*.csv'))

#Loop through the files and read them in with pandas
dataframes = []  # a list to hold all the individual pandas DataFrames
for csvfile in csvfiles:
    df = pd.read_csv(csvfile)
    dataframes.append(df)

#Concatenate them all together
dataset = pd.concat(dataframes, ignore_index=True)

#information regarding the dataset
dataset.info()

"""**EDA**
Checking for Missing Values and handling them
"""

#check whether there is NA values
dataset.isna().sum()


'''Name column have one missing value. Hence dropping it.'''
# drop rows with null and NA values
dataset = dataset[dataset['NAME'].notna()]

#Rechecking for NA values
dataset.isna().sum()

'''The turnout for each election'''
#finding the total turnout 
dataset["TURNOUT"] =dataset["VOTERS"] /dataset["ELECTORS"]

#Checking for NA values
dataset.isna().sum()

#dropping rows with value NA in turnout column
dataset = dataset[dataset['TURNOUT'].notna()]

#Rechecking for NA values
dataset.isna().sum()

#Summary statistics of the dataset
dataset.describe()

#Defining a new variable to divide the parties into important ones and others
def POLITICAL_PARTY (row):
    if row['PARTY'] == "ADMK":
        return "ADMK"
    if row['PARTY'] == "DMK":
        return "DMK"
    if row['PARTY'] == "INC":
        return "INC"
    if row['PARTY'] == "BJP":
        return "BJP"
    if row["PARTY"] == "IND":
        return "IND"
    if row["PARTY"] == "CPI":
        return "CPI"
    if row["PARTY"] == "CPI(ML)(L)":
        return "CPI(M)"
    if row["PARTY"] == "CPI(M)":
        return "CPI(M)"
    if row["PARTY"] == "CPM":
        return "CPI(M)"  
    else:
        return "OTHERS"
dataset['POLITICAL_PARTY']= dataset.apply(lambda row:POLITICAL_PARTY(row),axis=1)

#Separating train features and label
y = dataset["WINNER"]
X = dataset.drop(labels=["WINNER","NAME"], axis=1)

'''LABEL ENCODING'''

# label encode categorical columns
from sklearn.preprocessing import LabelEncoder
lblEncoder_cons = LabelEncoder()
lblEncoder_cons.fit(X['CONSTITUENCY'])
X['CONSTITUENCY'] = lblEncoder_cons.transform(X['CONSTITUENCY'])

lblEncoder_party = LabelEncoder()
lblEncoder_party.fit(X['PARTY'])
X['PARTY'] = lblEncoder_party.transform(X['PARTY'])

lblEncoder_winner = LabelEncoder()
lblEncoder_winner.fit(y)
y = lblEncoder_winner.transform(y)

lblEncoder_gender = LabelEncoder()
lblEncoder_gender.fit(X['GENDER'])
X['GENDER'] = lblEncoder_gender.transform(X['GENDER'])

lblEncoder_party = LabelEncoder()
lblEncoder_party.fit(X['POLITICAL_PARTY'])
X['POLITICAL_PARTY'] = lblEncoder_party.transform(X['POLITICAL_PARTY'])

lblEncoder_category = LabelEncoder()
lblEncoder_category.fit(X['CATEGORY'])
X['CATEGORY'] = lblEncoder_category.transform(X['CATEGORY'])

''''Feature Standardization'''

from sklearn.preprocessing import MinMaxScaler
# scaling values into 0-1 range
scaler = MinMaxScaler(feature_range=(0, 1))
features = ['YEAR','CONSTITUENCY', 'PARTY', 'GENDER', 'AGE', 'CATEGORY','VOTES', 'ELECTORS', 'VOTERS', 'VALID VOTES', 'POLITICAL_PARTY']
X[features] = scaler.fit_transform(X[features])

'''Building Model'''

''' The Classifiaction classes are highly unbalanced and so resampling is required.'''

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X, y= rus.fit_resample(X, y)

# split dataset into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



'''First Training model'''

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
#Create the random forest Classifier
clf=RandomForestClassifier()
#First training model
clf.fit(X_train,y_train)

'''There are two variables denoting the same feature viz PARTY and POLITICAL_PARTY, so one has to be dropped'''
'''This tool is a prediction tool the features YEAR,VOTES, VOTERS, VALID VOTES and TURNOUT hvae to be dropped'''
X_train2=X_train.drop(labels=["YEAR","PARTY", "VOTES", "VOTERS", "VALID VOTES", "TURNOUT"], axis=1)
X_test2=X_test.drop(labels=["YEAR","PARTY", "VOTES", "VOTERS", "VALID VOTES", "TURNOUT"], axis=1)

'''Rerunning the model'''
rf_clf=RandomForestClassifier()
rf_clf.fit(X_train2,y_train)

'''Getting preictions from the model'''
y_pred=rf_clf.predict(X_train2)


'''Evaluating the Performance of the model'''

#Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

rf_clf_cv_score = cross_val_score(rf_clf, X, y, cv=5, scoring='roc_auc')

print("=== Confusion Matrix ===")
print(confusion_matrix(y_train, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_train, y_pred))
print('\n')
print("=== All AUC Scores ===")
print(rf_clf_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rf_clf_cv_score.mean())

'''To generate the ROC curve'''

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# generate a no win prediction (majority class)
nw_probs = [0 for _ in range(len(y_train))]
# predict probabilities
rf_probs = rf_clf.predict_proba(X_train2)
# keep probabilities for the positive outcome only
rf_probs = rf_probs[:, 1]
# calculate scores
nw_auc = roc_auc_score(y_train, nw_probs)
rf_auc = roc_auc_score(y_train, rf_probs)
# summarize scores
print('No Win: ROC AUC=%.3f' % (nw_auc))
print('Winner: ROC AUC=%.3f' % (rf_auc))
# calculate roc curves
nw_fpr, nw_tpr, _ = roc_curve(y_train, nw_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_train, rf_probs)
# plot the roc curve for the model
from matplotlib import pyplot
pyplot.plot(nw_fpr, nw_tpr, linestyle='--', label='No Win')
pyplot.plot(rf_fpr, rf_tpr, marker='.', label='Winner')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

'''Hyper parameter tuning for optimal parameters'''

#Configuring parameteers and values to be serched
tuned_parameters=[{'max_depth':[10,15,20],
                   'n_estimators':[100, 200, 300],
                   'max_features':['sqrt',0.2]}]
#Configuring serch with the tunable parameters
from sklearn.model_selection import GridSearchCV
clfn=GridSearchCV(rf_clf, tuned_parameters,cv=5,scoring='roc_auc')

#Fitting the training set
clfn.fit(X_train2, y_train)

#finding the optimal parameters
clfn.best_params_

'''Building the final model with Optimal Parameters'''
rf_clf=RandomForestClassifier(max_depth=10,n_estimators= 200, max_features='sqrt')
rf_clf.fit(X_train2, y_train)

'''Getting prediction from the final model'''
yhat=rf_clf.predict(X_test2)

'''Evaluating the Performance of the model'''

#Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

rf_clf_cv_score = cross_val_score(rf_clf, X, y, cv=5, scoring='roc_auc')

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, yhat))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, yhat))
print('\n')
print("=== All AUC Scores ===")
print(rf_clf_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rf_clf_cv_score.mean())

'''To generate the ROC curve'''

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# generate a no win prediction (majority class)
nw_probs = [0 for _ in range(len(y_test))]
# predict probabilities
rf_probs = rf_clf.predict_proba(X_test2)
# keep probabilities for the positive outcome only
rf_probs = rf_probs[:, 1]
# calculate scores
nw_auc = roc_auc_score(y_test, nw_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
# summarize scores
print('No Win: ROC AUC=%.3f' % (nw_auc))
print('Winner: ROC AUC=%.3f' % (rf_auc))
# calculate roc curves
nw_fpr, nw_tpr, _ = roc_curve(y_test, nw_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
# plot the roc curve for the model
from matplotlib import pyplot
pyplot.plot(nw_fpr, nw_tpr, linestyle='--', label='No Win')
pyplot.plot(rf_fpr, rf_tpr, marker='.', label='Winner')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

#saving model to disk
import pickle
pickle.dump(rf_clf, open('model.pkl', 'wb'))

# load the model from disk
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[204,2,0,141692,24,0]]))

print(model.predict([[202,4,1,141692,18,1]]))
