import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
file_name = "Data Set For Task\\1) iris.csv"
if not os.path.exists(file_name):
    print(f"[ERROR] Could not find '{file_name}'.")
    exit()

df = pd.read_csv(file_name)
print(f"[INFO] Successfully loaded {file_name}")



#target variable 'species' ko X aur y mein split karna
X = df.drop(columns=['species'])
le = LabelEncoder()
y = le.fit_transform(df['species']) # Yeh 'species' text ko 0, 1, 2 numbers mein badal dega

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CRITICAL FOR SVM: Scaling
# SVM calculates geometric distances. We must scale so petal length and sepal length have equal weight.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. TRAIN & COMPARE SVM KERNELS
# ==========================================
print("\n[INFO] Training SVM models...")

print("\n[INFO] Training SVM models...")

# Model A: Linear Kernel
base_svm_linear = SVC(kernel='linear', random_state=42)
svm_linear = CalibratedClassifierCV(base_svm_linear, ensemble=False)
svm_linear.fit(X_train_scaled, y_train)

# Model B: RBF Kernel
base_svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf = CalibratedClassifierCV(base_svm_rbf, ensemble=False)
svm_rbf.fit(X_train_scaled, y_train)

# ==========================================
# 3. EVALUATE PERFORMANCE
# ==========================================
def evaluate_model(model, name):
    y_pred = model.predict(X_test_scaled)
    # Using 'ovr' (One-vs-Rest) and 'macro' average for multi-class AUC
    y_prob = model.predict_proba(X_test_scaled) 
    
    print(f"\n--- Performance: {name} ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro') * 100:.2f}%")
    print(f"Recall:    {recall_score(y_test, y_pred, average='macro') * 100:.2f}%")
    print(f"AUC Score: {roc_auc_score(y_test, y_prob, multi_class='ovr'):.4f}")

evaluate_model(svm_linear, "SVM (Linear Kernel)")
evaluate_model(svm_rbf, "SVM (RBF Kernel)")

# ==========================================
# 4. VISUALIZE DECISION BOUNDARY (PCA 2D)
# ==========================================
# The Iris dataset has 4 features, but a graph can only show 2 dimensions. 
# We use PCA to squash the 4 columns down into 2 principal components.
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# Train a special 2D SVM just for the plot (using RBF)
svm_2d = SVC(kernel='rbf', random_state=42)
svm_2d.fit(X_train_pca, y_train)

# Create a mesh grid for the background colors
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict every pixel on the background grid
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
# Draw the background decision regions
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set2')
# Plot the actual flower dots
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='Set2', edgecolors='k', s=60)
plt.title('SVM Decision Boundary (RBF Kernel) - 2D PCA Projection for Iris')
plt.xlabel('Principal Component 1 (Squashed Data)')
plt.ylabel('Principal Component 2 (Squashed Data)')
plt.legend(*scatter.legend_elements(), title="Species (0, 1, 2)")

image_name = "svm_iris_decision_boundary.png"
plt.savefig(image_name)
print(f"\n[INFO] Decision boundary plot saved as '{image_name}'.")