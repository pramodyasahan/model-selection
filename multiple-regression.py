# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Loading the dataset from a CSV file
dataset = pd.read_csv('Data.csv')

# Extracting features (X) and target variable (y) from the dataset
X = dataset.iloc[:, :-1].values  # All rows, all columns except the last one
y = dataset.iloc[:, -1].values  # All rows, only the last column

# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)  # 20% of data is used for testing

# Creating a Linear Regression model
regressor = LinearRegression()

# Fitting (training) the Linear Regression model on the training data
regressor.fit(X_train, y_train)

# Predicting the target variable (y) for the test set
y_pred = regressor.predict(X_test)

# Setting display options for numpy arrays to show precision of 2 decimal points
np.set_printoptions(precision=2)

# Printing the predicted values and actual values side by side for comparison
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1), "\n")

# Calculating the R-squared score to evaluate the performance of the model
r2 = r2_score(y_test, y_pred)
print(r2)  # R-squared score provides an indication of the goodness of fit of the model
