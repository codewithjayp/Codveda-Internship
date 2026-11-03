import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
file_name = "preprocessed_data/processed_1) iris.csv"
try:
    df = pd.read_csv(file_name)
    print(f"[INFO] Successfully loaded {file_name}")
except FileNotFoundError:
    print(f"[ERROR] Could not find '{file_name}'. Ensure it is in the same folder.")
    exit()

# 2. Separate Features (inputs) and Target (outputs)
# We assume the last column is the category you want to predict.
# iloc[:, :-1] selects all columns except the last one.
# iloc[:, -1] selects only the last column.
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale the features
# KNN calculates the distance between points. Scaling ensures all columns 
# have an equal impact on the calculation, regardless of their original numbers.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the KNN Classifier (starting with K=3)
default_k = 3
knn = KNeighborsClassifier(n_neighbors=default_k)
knn.fit(X_train_scaled, y_train)

# 6. Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# 7. Evaluate the model's performance
print(f"\n--- Detailed Evaluation for K = {default_k} ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report (Precision & Recall):")
print(classification_report(y_test, y_pred))

# 8. Compare different values of K
print("\n--- Comparing Different Values of K ---")
k_values_to_test = [1, 3, 5, 7, 9, 11]

for k in k_values_to_test:
    # Create and train a new model for each K
    temp_knn = KNeighborsClassifier(n_neighbors=k)
    temp_knn.fit(X_train_scaled, y_train)
    
    # Check accuracy
    temp_accuracy = accuracy_score(y_test, temp_knn.predict(X_test_scaled))
    print(f"When K = {k} -> Accuracy is {temp_accuracy * 100:.2f}%")