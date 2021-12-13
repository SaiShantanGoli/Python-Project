# -*- coding: utf-8 -*-

#importing necessary libraries
#!pip3 install dataprep
import numpy as np
import pandas as pd
import seaborn as sns
from dataprep.eda import *
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,accuracy_score,recall_score,classification_report 


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

df = pd.read_excel("trainset.xlsx")
df

df[["TRAN_AMT","FRAUD_NONFRAUD"]]

df["FRAUD_NONFRAUD"].value_counts()

sns.countplot('FRAUD_NONFRAUD', data=df,hue="FRAUD_NONFRAUD")
plt.title('Class Distributions ', fontsize=14)

#Boxplot for the Amount feature, in order to visualiza the outliers.
sns.boxplot(x=df['FRAUD_NONFRAUD'], y=df['TRAN_AMT'])

df.isnull().sum()

df = df[['TRAN_AMT','ACCT_PRE_TRAN_AVAIL_BAL','CUST_AGE','OPEN_ACCT_CT','WF_dvc_age','CARR_NAME','RGN_NAME','STATE_PRVNC_TXT','ALERT_TRGR_CD','DVC_TYPE_TXT','CUST_ZIP','CUST_STATE','FRAUD_NONFRAUD']]

df.isnull().sum()
#create_report(df)

df['CARR_NAME'] = df['CARR_NAME'].fillna('None')
df['RGN_NAME'] = df['RGN_NAME'].fillna('None')
df['STATE_PRVNC_TXT'] = df['STATE_PRVNC_TXT'].fillna('None')
df['DVC_TYPE_TXT'] = df['DVC_TYPE_TXT'].fillna('None')
df['CUST_STATE'] = df['CUST_STATE'].fillna('None')

df["FRAUD_NONFRAUD"].value_counts()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Categorical boolean mask
categorical_feature_mask = df.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = df.columns[categorical_feature_mask].tolist()

categorical_cols

# apply le on categorical feature columns
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
df

le.transform(["Fraud","Non-Fraud"])
#create_report(df)

df

X = df.drop('FRAUD_NONFRAUD',axis=1)
Y = df['FRAUD_NONFRAUD']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42,stratify=Y)

# X_train = X_train.values
# X_test = X_test
# y_train = y_train.values
# y_test = y_test.values

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

score = accuracy_score(y_test,y_pred)
score

print(classification_report(y_test,y_pred))
#plot_confusion_matrix(cm)

svm = SVC()
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
score = accuracy_score(y_test,y_pred)
score

print(classification_report(y_test,y_pred))
#plot_confusion_matrix(cm)

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "xgb" : GradientBoostingClassifier()
}

from sklearn.model_selection import cross_val_score


for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

    
# Classifiers:  LogisticRegression Has a training score of 79.0 % accuracy score
# Classifiers:  KNeighborsClassifier Has a training score of 83.0 % accuracy score
# Classifiers:  SVC Has a training score of 70.0 % accuracy score
# Classifiers:  DecisionTreeClassifier Has a training score of 92.0 % accuracy score

xgb = GradientBoostingClassifier(
    max_features='auto',
    min_samples_leaf=1,
    n_estimators=400,
    learning_rate=0.5,
    max_depth=5,
    random_state=1,
    )
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)

score = accuracy_score(y_test,y_pred)
score
print(classification_report(y_test,y_pred))
#plot_confusion_matrix(cm)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# train neural netork

clas.fit(X_train,y_train,batch_size=256,epochs=5,verbose=2)

# evalute model

scores = clas.evaluate(X_test,y_test,verbose=0)

# print results

print('Error',scores[0],'\nAccuracy',scores[1])



plt.plot(history_org.history['acc'])
plt.plot(history_org.history['val_acc'])
plt.title('Evaluation of Data Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy of Data')
plt.legend(['TrainData', 'ValidationData'], loc='upper left')
plt.show()





"""# Randomn Under sampling"""

df1 = pd.read_excel("trainset.xlsx")
df1

fraud_df = df1.loc[df1['FRAUD_NONFRAUD'] == "Fraud"]
#len(fraud_df)
nonfraud_df = df1.loc[df1['FRAUD_NONFRAUD'] == "Non-Fraud"][:len(fraud_df)]

equal_df = pd.concat([fraud_df, nonfraud_df])

equal_df

sns.countplot('FRAUD_NONFRAUD', data=equal_df,hue="FRAUD_NONFRAUD")
plt.title('Class Distributions ', fontsize=14)

equal_df = equal_df[['TRAN_AMT','ACCT_PRE_TRAN_AVAIL_BAL','CUST_AGE','OPEN_ACCT_CT','WF_dvc_age','CARR_NAME','RGN_NAME','STATE_PRVNC_TXT','ALERT_TRGR_CD','DVC_TYPE_TXT','CUST_ZIP','CUST_STATE','FRAUD_NONFRAUD']]

equal_df['CARR_NAME'] = equal_df['CARR_NAME'].fillna('None')
equal_df['RGN_NAME'] = equal_df['RGN_NAME'].fillna('None')
equal_df['STATE_PRVNC_TXT'] = equal_df['STATE_PRVNC_TXT'].fillna('None')
equal_df['DVC_TYPE_TXT'] = equal_df['DVC_TYPE_TXT'].fillna('None')
equal_df['CUST_STATE'] = equal_df['CUST_STATE'].fillna('None')

# Categorical boolean mask
categorical_feature_mask = equal_df.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = equal_df.columns[categorical_feature_mask].tolist()

# apply le on categorical feature columns
equal_df[categorical_cols] = equal_df[categorical_cols].apply(lambda col: le.fit_transform(col))
equal_df

X_new = equal_df.drop('FRAUD_NONFRAUD',axis=1)
Y_new = equal_df['FRAUD_NONFRAUD']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_new, Y_new, test_size=0.2, random_state=44)

lr = LogisticRegression()
lr.fit(X_train1,y_train1)
y_pred = lr.predict(X_test1)

score = accuracy_score(y_test1,y_pred)
print(score)
print(classification_report(y_test1,y_pred))
#plot_confusion_matrix(cm)

svm = SVC()
svm.fit(X_train1,y_train1)
y_pred = svm.predict(X_test1)
score = accuracy_score(y_test1,y_pred)
print(score)
print(classification_report(y_test1,y_pred))
#plot_confusion_matrix(cm)

xgb = GradientBoostingClassifier(
    max_features='auto',
    min_samples_leaf=1,
    n_estimators=400,
    learning_rate=0.5,
    max_depth=5,
    random_state=1,
    )
xgb.fit(X_train1,y_train1)
y_pred = xgb.predict(X_test1)
score = accuracy_score(y_test1,y_pred)
print(score)
print(classification_report(y_test1,y_pred))

le.transform(["Fraud","Non-Fraud"])
#create_report(df)

#predicting with original
y_pred = xgb.predict(X_test)
score = accuracy_score(y_test,y_pred)
print(score)
print(classification_report(y_test,y_pred))

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')
    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

clas = Sequential()
clas.add(Dense(units=16,kernel_initializer='uniform',activation='relu',input_dim=12))
clas.add(Dropout(rate=0.1))
clas.add(Dense(units=32,kernel_initializer='uniform',activation='relu',input_dim=12))
clas.add(Dropout(rate=0.1))
clas.add(Dense(units=2,kernel_initializer='uniform',activation='softmax'))
clas.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# train neural netork

clas.fit(X_train1,y_train1,batch_size=10,epochs=5,verbose=2)

# evalute model

scores = clas.evaluate(X_test1,y_test1,verbose=0)

# print results

print('Error',scores[0],'\nAccuracy',scores[1])

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train1= sc.fit_transform(X_train1)
X_test1 = sc.fit_transform(X_test1)

# train neural netork

clas.fit(X_train1,y_train1,batch_size=10,epochs=15,verbose=2)

# evalute model

scores = clas.evaluate(X_test1,y_test1,verbose=0)

# print results

print('Error',scores[0],'\nAccuracy',scores[1])

scores = clas.evaluate(X_test,y_test,verbose=0)

# print results

print('Error',scores[0],'\nAccuracy',scores[1])

