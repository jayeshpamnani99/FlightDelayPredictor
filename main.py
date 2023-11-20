import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve

import matplotlib.pyplot as plt

#Reading the csv file and segregating x and y
irisDataset = genfromtxt('2002_updated.csv', delimiter=',',dtype=None,encoding=None)
x = irisDataset[1:,:13]
y = irisDataset[1:,13]

y = np.where(y == 'NA', 0, y)
y=y.astype('int')
y = np.where(y>5,1,0)

encoder = OrdinalEncoder()

X_8 = x[:,8].reshape (-1, 1)
X_8_encoded = encoder.fit_transform (X_8)
X_9 = x[:,9].reshape (-1, 1)
X_9_encoded = encoder.fit_transform (X_9)
x = np.hstack((x, X_8_encoded, X_9_encoded))
x = np.delete(x, [8, 9], axis=1)


# Splitting the dataset into training and testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# #---------Decision Tree on the Flight Dataset--------------------

print("\n------Finding AUC of Decision Tree Algoritm on the Flight Dataset------\n")

# #Creating & Training the Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)

dt_prediction=dt_model.predict(x_test)
classificationReport=metrics.classification_report(dt_prediction,y_test)

dt_fpr, dt_tpr, threshold =roc_curve(y_test,dt_prediction)
auc_dt = auc(dt_fpr, dt_tpr)
print("Accuracy for the Decision Tree Model: ", dt_model.score(x_test,y_test))
print("Classification Report for the Decision Tree Model: \n", classificationReport)


# #---------Logistic Regression on the Flight Dataset--------------------

print("\n\n------Finding AUC of Logistic Regression Algoritm on the Flight Dataset------\n")

#Creating & Training the Logistic Regression Model
lr_model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=5000)
lr_model.fit(x_train,y_train)

x_test = x_test.astype(float)
lr_prediction=lr_model.predict(x_test)
classificationReport=metrics.classification_report(lr_prediction,y_test)


lr_fpr, lr_tpr, threshold =roc_curve(y_test,lr_prediction)
auc_lr = auc(lr_fpr, lr_tpr)

print("Accuracy for the Logistic Regression Model: ", lr_model.score(x_test,y_test))
print("Classification Report for the Logistic Regression Model: \n", classificationReport)


plt.plot(lr_fpr, lr_tpr, marker='.',label='Logistic Regression (auc=%0.3f)' % auc_lr)
plt.plot(dt_fpr, dt_tpr, linestyle='-', label='Decision Tree (auc=%0.3f)' % auc_dt)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Baseline')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()