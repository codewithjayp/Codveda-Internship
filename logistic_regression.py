import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc

# 1. Load the dataset
# We will use the 80% churn dataset file you processed earlier
file_name = "preprocessed_data\\processed_churn-bigml-80.csv"
if not os.path.exists(file_name):
    print(f"[ERROR] Could not find '{file_name}'. Please check the folder.")
    exit()

df = pd.read_csv(file_name)
print(f"[INFO] Successfully loaded {file_name}")

# 2. Preprocess the Data
# 'Churn' is the target column (1 = left company, 0 = stayed)
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features so large numbers (minutes) don't overpower small numbers (calls)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. Make Predictions
y_pred = model.predict(X_test_scaled)
y_pred_probabilities = model.predict_proba(X_test_scaled)[:, 1] # Get probabilities for the ROC curve

# 5. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")

# 6. Interpret Coefficients and Odds Ratios
print("\n--- Feature Impact (Odds Ratios) ---")
print("An Odds Ratio > 1 increases the chance of Churn.")
print("An Odds Ratio < 1 decreases the chance of Churn.")

coefficients = model.coef_[0]
# Calculate Odds Ratios using the exponential function (e^coef)
odds_ratios = np.exp(coefficients)

# Combine feature names, coefficients, and odds ratios into a readable table
impact_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'Odds_Ratio': odds_ratios
}).sort_values(by='Odds_Ratio', ascending=False)

print(impact_df.to_string(index=False))

# 7. Plot the ROC Curve
# Calculate the False Positive Rate (FPR) and True Positive Rate (TPR)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probabilities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (Area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Baseline (random guessing)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Predicting Churn mistakenly)')
plt.ylabel('True Positive Rate (Catching actual Churn)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Save the plot as an image file so you can easily view it in VS Code
image_name = "roc_curve_churn.png"
plt.savefig(image_name)
print(f"\n[INFO] ROC Curve saved as '{image_name}' in your folder.")