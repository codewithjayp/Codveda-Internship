import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
# We use the 80% dataset for training to give the forest more data to learn from
file_name = os.path.join("preprocessed_data", "processed_churn-bigml-80.csv")
if not os.path.exists(file_name):
    # Fallback to current directory if not in preprocessed_data folder
    file_name = "processed_churn-bigml-80.csv" 

df = pd.read_csv(file_name)
print(f"[INFO] Successfully loaded dataset: {file_name}")

# 'Churn' is our target. 1 = Customer left, 0 = Customer stayed.
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split data: 80% for training the forest, 20% for the final exam
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. HYPERPARAMETER TUNING (GridSearchCV)
# ==========================================
print("\n[INFO] Training the Random Forest and searching for the best settings...")
print("This might take a few seconds as it builds hundreds of trees!")

# We give the AI a few options for 'n_estimators' (number of trees) 
# and 'max_depth' (how many questions each tree can ask)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# The RandomForestClassifier model
rf_base = RandomForestClassifier(random_state=42)

# GridSearchCV tests every possible combination in param_grid using 5-Fold Cross-Validation
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"\n[SUCCESS] Best settings found: {grid_search.best_params_}")

# ==========================================
# 3. EVALUATE THE MODEL
# ==========================================
y_pred = best_rf.predict(X_test)

print("\n--- Final Model Performance ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}% (When we predict Churn, we are right X% of the time)")
print(f"Recall:    {recall_score(y_test, y_pred) * 100:.2f}% (Out of all real Churners, we caught X%)")
print(f"F1-Score:  {f1_score(y_test, y_pred) * 100:.2f}% (Balance between Precision and Recall)")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Cross-Validation Score Check
cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy (5-Fold): {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

# ==========================================
# 4. FEATURE IMPORTANCE ANALYSIS
# ==========================================
# Extract the importance of each feature calculated by the Random Forest
feature_importances = best_rf.feature_importances_

# Create a DataFrame to sort and plot them
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
plt.title('Top 10 Most Important Features (Random Forest)')
plt.xlabel('Importance Score (How much this feature drives the Churn decision)')
plt.ylabel('Feature Name')
plt.tight_layout()

# Save the plot
plt.savefig('rf_feature_importance.png')
print("\n[INFO] Feature importance chart saved as 'rf_feature_importance.png'.")