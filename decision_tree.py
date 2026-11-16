import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score

# 1. Load the Dataset
file_name = "preprocessed_data/processed_1) iris.csv"
if not os.path.exists(file_name):
    print(f"[ERROR] Could not find '{file_name}'. Please check the folder.")
    exit()

df = pd.read_csv(file_name)
print(f"[INFO] Successfully loaded {file_name}")

# 2. Separate Features (Inputs) and Target (Output)
X = df.drop(columns=['species'])
y = df['species']

# 3. Split the data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train and PRUNE the Decision Tree
# Pruning (max_depth=3) stops the tree from growing too deep. 
# Without this, the tree will overfit (memorize the training data) and fail on new data.
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

# 5. Make Predictions
y_pred = tree_model.predict(X_test)

# 6. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
# We use 'weighted' F1-score because there are 3 different flower species
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")

# 7. Visualize the Tree Structure
plt.figure(figsize=(12, 8))
plot_tree(
    tree_model, 
    feature_names=X.columns, 
    class_names=['Species 0', 'Species 1', 'Species 2'], 
    filled=True, 
    rounded=True
)
plt.title("Pruned Decision Tree for Iris Dataset")

# Save the drawing as an image file
image_name = "iris_decision_tree.png"
plt.savefig(image_name)
print(f"\n[INFO] Tree visualization saved as '{image_name}'. Open it in VS Code to see the rules!")