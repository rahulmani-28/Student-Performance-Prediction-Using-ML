# importing all the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import warnings



# dataset dataset
df = pd.read_csv("D:\Machinelearing\internship\project1\StudentsPerformance 3.csv")

# Check for missing values
print(df.isnull().sum())

# Impute missing numeric values instead of dropping
nums_cols = df.select_dtypes(exclude="object").columns
imputer = SimpleImputer(strategy='mean')
df[nums_cols] = imputer.fit_transform(df[nums_cols])

# Define features and target
X = df.drop(columns=["math score"])
Y = df["math score"]

# Preprocessing: Scaling numeric data and one-hot encoding categorical data
nums_cols = X.select_dtypes(exclude="object").columns
cats_cols = X.select_dtypes(include="object").columns

nums_trans = StandardScaler()
oh_trans = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_trans, cats_cols),
        ("StandardScaler", nums_trans, nums_cols)
    ]
)

# Create pipeline
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', LinearRegression())])

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=5, test_size=0.25)

# Fit the model
pipe.fit(x_train, y_train)

# Predicting the values of x_test
y_pred = pipe.predict(x_test)

# Calculate errors
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
score = r2_score(y_test, y_pred) * 100

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² Score: {score}")

# Cross-validation
cross_val = cross_val_score(pipe, X, Y, cv=5, scoring='r2')
print(f"Cross-Validated R² Scores: {cross_val}")
print(f"Mean Cross-Validation R² Score: {cross_val.mean()}")

# Grid Search for hyperparameter tuning
param_grid = {
    'model__fit_intercept': [True, False]
}
grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print(f"Best Params: {grid_search.best_params_}")

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Distribution of residuals
sns.histplot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.show()

# Pairplot
sns.pairplot(df, hue='gender')
plt.show()
