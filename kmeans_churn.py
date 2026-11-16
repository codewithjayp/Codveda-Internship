import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD THE DATASET
# ==========================================
# File path adjust kiya gaya hai as per your folder structure
file_name = os.path.join("preprocessed_data", "processed_churn-bigml-80.csv")

if not os.path.exists(file_name):
    print(f"[ERROR] '{file_name}' not found. Please check your folder path.")
    exit()

df = pd.read_csv(file_name)
print(f"[INFO] Successfully loaded {file_name}")

# Clustering "Unsupervised" hai, isliye hum AI se 'Churn' (Target) chupa lete hain.
# Hum chahte hain AI khud customers ka behavior samjhe.
if 'Churn' in df.columns:
    X = df.drop(columns=['Churn'])
else:
    X = df.copy()

# ==========================================
# 2. PREPROCESS THE DATA (SCALING)
# ==========================================
# Scale karna ZAROORI hai! 'Total day minutes' 300+ ho sakte hain, 
# aur 'Customer service calls' sirf 1 ya 2. Bina scaling ke AI calls ko ignore kar dega.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. THE ELBOW METHOD (Find Optimal Clusters)
# ==========================================
inertia_values = []
k_range = range(1, 11)

print("\nRunning Elbow Method to find the best number of customer groups...")
for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_test.fit(X_scaled)
    inertia_values.append(kmeans_test.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia_values, marker='o', linestyle='--', color='darkblue')
plt.title('The Elbow Method (Customer Segmentation)')
plt.xlabel('Number of Customer Groups (K)')
plt.ylabel('Inertia (Distance within groups)')
plt.grid(True)
plt.savefig('churn_elbow.png')
print("[INFO] Elbow curve saved as 'churn_elbow.png'.")

# ==========================================
# 4. APPLY K-MEANS WITH OPTIMAL K
# ==========================================
# Churn data mein usually 3 main customer types hote hain (High users, Low users, Angry users)
optimal_k = 3
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Customer_Segment'] = final_kmeans.fit_predict(X_scaled)

# ==========================================
# 5. VISUALIZE CLUSTERS (2D SCATTER PLOT)
# ==========================================
# Pura data 20 columns ka hai, par graph sirf 2D ho sakta hai. 
# Toh hum 2 sabse important features dekhte hain: "Day Minutes" vs "Customer Service Calls"
plt.figure(figsize=(10, 6))

feature_x = 'Total day minutes'
feature_y = 'Customer service calls'

# Note: Hum original 'df' use kar rahe hain graph ke liye taaki numbers real lagen (jaise 200 minutes)
sns.scatterplot(x=df[feature_x], y=df[feature_y], hue=df['Customer_Segment'], palette='Set1', alpha=0.7)

plt.title(f'Customer Segmentation (K={optimal_k})')
plt.xlabel('Total Day Minutes (Talk time)')
plt.ylabel('Customer Service Calls (Complaints)')
plt.legend(title="Customer Type")
plt.savefig('churn_clusters.png')
print("[INFO] Cluster scatter plot saved as 'churn_clusters.png'.")

# ==========================================
# 6. INTERPRET THE RESULTS FOR BUSINESS
# ==========================================
print("\n--- Business Interpretation (Averages per Segment) ---")
# Hum sirf kuch important columns ka average dekh rahe hain poori table ki jagah
important_cols = ['Customer_Segment', 'Total day minutes', 'Total day charge', 'Customer service calls', 'International plan']
cluster_summary = df[important_cols].groupby('Customer_Segment').mean().round(2)
print(cluster_summary)