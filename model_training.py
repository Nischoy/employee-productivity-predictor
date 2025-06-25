# Importing Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading .csv file:
df = pd.read_csv('garments_worker_productivity.csv')
print("Dataset:")
print(df)
print("======================================================================")

# Correlation Analysis:
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
print("Correlation Matrix:")
print(correlation_matrix)
print("======================================================================")

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Descriptive Analysis:
print("Descriptive Analysis:")
print(df.describe())
print("======================================================================")

# Checking for Null Values:
print("Null Value Count:")
print(df.isnull().sum())
print("======================================================================")

# Dropping Feature with missing values:
df.drop(columns=['wip'], inplace=True)
print("Remaining columns after dropping 'wip':\n")
print(df.columns)
print("======================================================================")

# Shape of Dataset:
print("Shape of Dataset:")
print(df.shape)
print("======================================================================")

# Data Type:
print("Dataset Info:")
df.info()
print("======================================================================")

# Convert 'date' to datetime format:
df['date'] = pd.to_datetime(df['date'])

# Extract the month index (1–12) and save it to a new column:
df['month'] = df['date'].dt.month

# Drop the original 'date' column as it's no longer needed:
df.drop(columns=['date'], inplace=True)

print("Unique values in Department:")
print(df['department'].unique())
print("======================================================================")

# Merging 'finishing ' & 'finishing':
df['department'] = df['department'].str.strip()

print("Unique values in Department After Merging:")
print(df['department'].unique())
print("======================================================================")

# MultiColumnLabelEncoder:
from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        output = X.copy()
        for col in self.columns:
            output[col] = self.encoders[col].transform(X[col])
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

categorical_columns = ['quarter', 'department', 'day']

# Initialize encoder:
mcle = MultiColumnLabelEncoder(columns=categorical_columns)

# Apply encoding:
df = mcle.fit_transform(df)

# Train Test Split:
from sklearn.model_selection import train_test_split

# Selected Features:
X = df[['quarter', 'department', 'day', 'team', 'targeted_productivity',
        'smv', 'over_time', 'incentive', 'idle_time', 'idle_men',
        'no_of_workers', 'no_of_style_change', 'month']]

# Target Variable
y = df['actual_productivity']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=45)

# Linear Regression:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize Linear Regression model:
model_lr = LinearRegression()

# Train the model:
model_lr.fit(X_train, y_train)

# Predict on the test set:
pred_test = model_lr.predict(X_test)

# Model Evaluation:
mae_lr = mean_absolute_error(y_test, pred_test)
mse_lr = mean_squared_error(y_test, pred_test)
r2_lr = r2_score(y_test, pred_test)

# Results:
print("Linear Regression Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_lr:.4f}")
print(f"Mean Squared Error (MSE): {mse_lr:.4f}")
print(f"R² Score: {r2_lr:.4f}")
print("======================================================================")

# Random Forest:
from sklearn.ensemble import RandomForestRegressor

# Initialize Random Forest model:
model_rf = RandomForestRegressor()

# Train the model:
model_rf.fit(X_train, y_train)

# Predict on the test set:
pred = model_rf.predict(X_test)

# Model Evaluation:
mae_rf = mean_absolute_error(y_test, pred)
mse_rf = mean_squared_error(y_test, pred)
r2_rf = r2_score(y_test, pred)

# Results:
print("Random Forest Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_rf:.4f}")
print(f"Mean Squared Error (MSE): {mse_rf:.4f}")
print(f"R² Score: {r2_rf:.4f}")
print("======================================================================")

# XGBoost:
from xgboost import XGBRegressor

# Initialize XGBoost Regressor:
model_xgb = XGBRegressor()

# Train the model:
model_xgb.fit(X_train, y_train)

# Predict on the test set:
pred3 = model_xgb.predict(X_test)

# Model Evaluation:
mae_xgb = mean_absolute_error(y_test, pred3)
mse_xgb = mean_squared_error(y_test, pred3)
r2_xgb = r2_score(y_test, pred3)

# Results:
print("XGBoost Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_xgb:.4f}")
print(f"Mean Squared Error (MSE): {mse_xgb:.4f}")
print(f"R² Score: {r2_xgb:.4f}")
print("======================================================================")

# Model Performance Comparison:
comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'MAE': [mae_lr, mae_rf, mae_xgb],
    'MSE': [mse_lr, mse_rf, mse_xgb],
    'R² Score': [r2_lr, r2_rf, r2_xgb]
})

print("Model Performance Comparison:")
print(comparison_df)
print("======================================================================")

# Saving the Model:
import pickle

# Save the model to a file
with open('XGBoost.pkl', 'wb') as file:
    pickle.dump(model_xgb, file)
