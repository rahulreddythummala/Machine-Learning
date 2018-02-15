#Polynomial Regression
# y = b0 + b1x1 + b2x1^2 + ........... + bnx1^n

# it is still linear because its not with the variable it is called because of the coefficient 
# if it was b0/b1 + .... its not linear

#Bluffing Detector

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# As the data is too small no splittibg into traning and test sets for accuracy

#Fitting Linear Regression to the dataset(to compare with PR)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
#created column of 1's for b0 constant and also a column of squares
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#visulazing the linear Regression Model
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X))
plt.title('Truth or Bluff Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show() 
#this model says for 6.5 level its $300K

#visulazing the Polynomial Regression Model
X_grid = np.arange(min(X), max(X), 0.1)# for a smoother curve in steps of 0.1
X_grid = X_grid.reshape(len(X_grid),1)# since we need a matrix
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)))
plt.title('Truth or Bluff Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#this model says for 6.5 level its around $200K with degree 2
#this model says for 6.5 level its around $170K with degree 3
#this model says for 6.5 level its around $160K with degree 4 more accurate

#Predicting a new result with Linear Regression for 6.5level
lin_reg.predict(6.5)  # $330K

#Predicting a new result with Linear Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))   #   $158862