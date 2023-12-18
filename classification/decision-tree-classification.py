# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Loading the dataset from a CSV file
dataset = pd.read_csv('Data.csv')

# Extracting features (X) and target variable (y) from the dataset
X = dataset.iloc[:, :-1].values  # Selecting all rows and all but the last column as features
y = dataset.iloc[:, -1].values  # Selecting all rows and only the last column as the target variable

# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)  # 25% data for testing

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating a Decision tree model
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)  # Training the model on the training set

# Predicting the target variable for the test set
y_pred = classifier.predict(X_test)

# Creating a confusion matrix and calculating accuracy
cm = confusion_matrix(y_test, y_pred)
print(cm)  # Printing the confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)  # Printing the accuracy of the model
