# Multiple Linear Regression
#y = b0 + b1x1 + b2x2 + .......... + bnxn

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding the dummy variable trap
X = X[:, 1:] #we need to remove one dummy variable, generally linear regression library takes care of it

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling will be taken care by the library

# Fitting multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predecting the test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination(Because a few variables might have greater impact and a few not significant)
#assuming significance level as 0.05 i.e., 5%
import statsmodels.formula.api as sm
# this stats model doesnt consider b0 constant so we add a column with 1's so b0*x0 stays
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
# new regressor from statsmodels library
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()#Ordinary Least Squares
regressor_OLS.summary()
#removing 2nd predictor because of higher p value
X_opt = X[:,[0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#removing 1st predictor because of higher p value
X_opt = X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#removing 4th predictor because of higher p value
X_opt = X[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#removing 5th predictor because of higher p value
X_opt = X[:,[0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#creating new training and testing sets with the X_opt matrix and see the predictions
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

'''we reduce number of variables not because we will get a better model, but because we can get a model with the same performance and fewer variables. Our advantages are the following:
1) We get rid of variables which have negligible predictive power, and learn what variables have actual influence on our outcome (independent, target) variable. 
2) We reduce amount of data which we need to mine, clean, pre-process and store, thus reducing possible data errors and saving resources.
3) We get more time and memory efficient models.'''


