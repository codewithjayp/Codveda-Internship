import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
# Make sure the file is in the same folder as this script
file_name = "./preprocessed_data/processed_4) house Prediction Data Set.csv"
try:
    df = pd.read_csv(file_name)
    print(f"Successfully loaded {file_name}")
except FileNotFoundError:
    print(f"Error: Could not find '{file_name}'. Check your folder path.")
    exit()

# 2. Preprocess the data
# Based on the Boston Housing structure, column '13' is the target (House Price)
# and columns '0' through '12' are the input features.
target_col = '13'

# If pandas read the columns as strings or integers, ensure we grab the right one
if target_col not in df.columns and 13 in df.columns:
    target_col = 13

X = df.drop(columns=[target_col])  # Features (Inputs)
y = df[target_col]                 # Target (Output we want to predict)

# Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features so all numbers are on a similar scale (Mean=0, Variance=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 4. Make predictions on the unseen test data
y_pred = model.predict(X_test_scaled)

print(f"\n--- Predictions on Test Data --- : {y_pred[:5]}")  # Display first 5 predictions for brevity


# 5. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.4f}")

# 6. Interpret the model coefficients
print("\n--- Model Coefficients ---")
print("These values show how much the House Price changes when a feature increases by 1 standard deviation:")
for col_name, coef in zip(X.columns, model.coef_):
    print(f"Feature '{col_name}': {coef:.4f}")