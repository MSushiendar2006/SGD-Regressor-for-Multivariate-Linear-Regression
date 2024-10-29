# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries:Import necessary libraries for data handling, model building, and evaluation (e.g., NumPy, scikit-learn).

2.Load Dataset:Fetch the California housing dataset using fetch_california_housing().

3.Prepare Features and Targets:Select relevant features (e.g., the first three columns of the dataset):Create target variables as a combination of house prices (target) and the number of occupants (another feature, e.g., column 6).
    
4.Split the Data:Use train_test_split() to divide the dataset into training and testing sets. Set aside a portion (e.g., 20%) for testing.

5.Scale the Data:Standardize the feature set (X) using StandardScaler() and Standardize the target set (Y) using another StandardScaler().

6.Initialize SGD Regressor:Create an instance of SGDRegressor with specified parameters (e.g., max_iter and tol).

7.Multi-Output Regression:Wrap the SGD regressor with MultiOutputRegressor to handle multiple outputs (house price and number of occupants).

8.Train the Model:Fit the model on the training data using the fit() method.

9.Make Predictions:Predict the target values for the test set using the predict() method.

10.Inverse Transform Predictions:Transform the predicted and actual target values back to their original scale using the inverse transform of the scaler.

11.Evaluate the Model.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Sushiendar M
RegisterNumber: 212223040217 
*/
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
df.info()
X=df.drop(columns=['AveOccup','target'])
X.info()
Y=df[['AveOccup','target']]
Y.info()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X.head()
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
print(X_train)
# Initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

# Use MultiOutputRegressor to handle multiple output variables
multi_output_sgd = MultiOutputRegressor(sgd)

# Train the model
multi_output_sgd.fit(X_train, Y_train)

# Predict on the test data
Y_pred = multi_output_sgd.predict(X_test)

# Initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

# Use MultiOutputRegressor to handle multiple output variables
multi_output_sgd = MultiOutputRegressor(sgd)

# Train the model
multi_output_sgd.fit(X_train, Y_train)

# Predict on the test data
Y_pred = multi_output_sgd.predict(X_test)

# Inverse transform the predictions to get them back to the original scale
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

# Optionally, print some predictions
print("\nPredictions:\n", Y_pred[:5])
```

## Output:
![2024-09-11](https://github.com/user-attachments/assets/d8c3bb06-cf9f-4b0d-ae33-865a049cbf59)
![2024-09-11 (1)](https://github.com/user-attachments/assets/6d5f6303-9b47-4b44-a9c7-e60d3c404ca3)
![2024-09-11 (3)](https://github.com/user-attachments/assets/be31213c-d973-43da-bb38-a4fd146ca6ea)




## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
