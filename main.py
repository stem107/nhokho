import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

print('Libraries Imported')

# Creating Dataset and including the first row by setting no header as input
dataset = pd.read_csv('Raisin.csv')
# Renaming the columns
dataset.columns = [
     'Area', 'MajorAxisLength', 'MinorAxisLength',
    'Eccentricity', 'ConvexArea', 'Extent', 'Perimeter', 'class'
]

print('Shape of the dataset: ' + str(dataset.shape))
print(dataset.head())

# Creating the dependent variable class
factor = pd.factorize(dataset['class'])
dataset['class'] = factor[0]
definitions = factor[1]
print(dataset['class'].head())
print(definitions)

# Splitting the data into independent and dependent variables
X = dataset.iloc[:, :-1].values  # Independent variables: 'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'ConvexArea', 'Extent', 'Perimeter'
y = dataset.iloc[:, -1].values   # Dependent variable: 'class'

print('The independent features set: ')
print(X[:5,:])
print('The dependent variable: ')
print(y[:5])

# Creating the Training and Test set from data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']))

print(list(zip(dataset.columns[:-1], classifier.feature_importances_)))
joblib.dump(classifier, 'randomforestmodel.pkl')

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")