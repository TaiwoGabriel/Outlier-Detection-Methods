# Outliers in dataset could help reduce the mean absolute errors. Now, we will remove the outliers in the
# training data set only using four automatic outliers detection methods: Isolation Forest,
# Minimum Covariance Determinant, Local Outlier Factors and One-class SVM.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# load the dataset
raw_data = 'C:/Users/Gabriel/Desktop/housing.csv'
df = pd.read_csv(raw_data, header=None)
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# summarize the shape of the dataset
print(X.shape, y.shape)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the train and test sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# identify outliers using Isolation Forest in the training dataset
iso = IsolationForest(contamination=0.1)
# Perhaps the most important hyperparameter in the model is the “contamination” argument, which is used
# to help estimate the number of outliers in the dataset. This is a value between 0.0 and 0.5 and by default is
# set to 0.1.
yhat = iso.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)


"""
# identify outliers in Minimum Covariance Determinant in the training dataset
ee = EllipticEnvelope(contamination=0.01)
# It provides the “contamination” argument that defines the expected ratio of outliers to be observed in
# practice. In this case, we will set it to a value of 0.01, found with a little trial and error
yhat = ee.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
"""

"""
# identify outliers using Local Outlier Factor in the training dataset
lof = LocalOutlierFactor()
# The model provides the “contamination” argument, that is the expected percentage of outliers in the
# dataset, be indicated and defaults to 0.1
yhat = lof.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
"""

"""
# identify outliers using One Class SVM in the training dataset
ee = OneClassSVM(nu=0.01)
# The class provides the “nu” argument that specifies the approximate ratio of outliers in the dataset, which
# defaults to 0.1. In this case, we will set it to 0.01, found with a little trial and error.
yhat = ee.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
"""

# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('The mean absolute error is: %.3f' % mae)
