import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
crop_dataset = pd.read_csv('Dataset/Dataset.csv')
crop_dataset.fillna(0, inplace=True)

# Convert 'Production' column to integer type
crop_dataset['Production'] = crop_dataset['Production'].astype(np.int64)

# Encode categorical columns
le = LabelEncoder()
crop_dataset['State_Name'] = pd.Series(le.fit_transform(crop_dataset['State_Name']))
crop_dataset['District_Name'] = pd.Series(le.fit_transform(crop_dataset['District_Name']))
crop_dataset['Season'] = pd.Series(le.fit_transform(crop_dataset['Season']))
crop_dataset['Crop'] = pd.Series(le.fit_transform(crop_dataset['Crop']))

# Prepare the data for model training
crop_datasets = crop_dataset.values
cols = crop_datasets.shape[1] - 1
X = crop_datasets[:, 0:cols]
Y = crop_datasets[:, cols]
Y = Y.astype('uint8')
X = normalize(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Train a DecisionTreeRegressor model
clf = DecisionTreeRegressor(max_depth=100, random_state=0, max_leaf_nodes=20, max_features=5, splitter="random")
clf.fit(X_train, y_train)

# Make predictions
predict = clf.predict(X_test)
print("Predictions:", predict)

# Load the test data for prediction
test = pd.read_csv('Dataset/test.csv')
test.fillna(0, inplace=True)

# Encode the test data
test['State_Name'] = pd.Series(le.fit_transform(test['State_Name']))
test['District_Name'] = pd.Series(le.fit_transform(test['District_Name']))
test['Season'] = pd.Series(le.fit_transform(test['Season']))
test['Crop'] = pd.Series(le.fit_transform(test['Crop']))

# Prepare the test data for prediction
test = test.values
test = normalize(test)
test = test[:, 0:test.shape[1]]

# Make predictions on the test data
predictions_on_test = clf.predict(test)
print("Predictions on test data:", predictions_on_test)

# Calculate Mean Squared Error and Root Mean Squared Error
mse = mean_squared_error(predict, y_test)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)
