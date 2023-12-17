# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# Loading the dataset from a CSV file
dataset = pd.read_csv('Data.csv')

# Extracting features (X) and target variable (y) from the dataset
X = dataset.iloc[:, :-1].values  # Selecting all rows and all but the last column as features
y = dataset.iloc[:, -1].values   # Selecting all rows and only the last column as the target variable

# Splitting the dataset into Training and Test sets
# Test set size is 20% of the entire dataset, and the split is randomized
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating polynomial features of degree 4
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)  # Transforming the training set

# Creating a Linear Regression model
regressor = LinearRegression()

# Fitting (training) the Linear Regression model on the polynomially transformed training data
regressor.fit(X_poly, y_train)

# Predicting the target variable (y) for the polynomially transformed test set
y_pred = regressor.predict(poly_reg.transform(X_test))

# Setting display options for numpy arrays to show precision of 2 decimal points
np.set_printoptions(precision=2)

# Printing the predicted values and actual values side by side for comparison
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1), "\n")

# Calculating the R-squared score to evaluate the performance of the model
r2 = r2_score(y_test, y_pred)
print(r2)  # R-squared score provides an indication of the goodness of fit of the model

