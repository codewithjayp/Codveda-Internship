# Codeveda-Internship: Machine Learning Portfolio

A complete, end-to-end Machine Learning & Deep Learning repository built during the **Codeveda Internship**. This project covers everything from automated data preprocessing to classical ML models and Deep Learning with TensorFlow/Keras—explained in simple, real-life terms (**ELI10**).

---

## Project Structure & Dataset Mapping

| Script File | Algorithm / Task | Dataset Used | Generated Visuals |
| :--- | :--- | :--- | :--- |
| `task1.py` | Automated Data Preprocessing | All 6 raw datasets in `Data Set For Task/` | Saved clean CSVs in `preprocessed_data/` |
| `simple_linear_regression_model.py` | Linear Regression | `processed_4) house Prediction Data Set.csv` | Console Metrics (MSE, R²) |
| `knn_classifier_code.py` | K-Nearest Neighbors (KNN) | `processed_2) Stock Prices Data Set.csv` | Console Metrics & K-Tuning |
| `logistic_regression.py` | Logistic Regression | `processed_churn-bigml-80.csv` | `roc_curve_churn.png` |
| `decision_tree.py` | Decision Tree Classifier | `processed_1) iris.csv` | `iris_decision_tree.png` |
| `kmeans_churn.py` | K-Means Clustering | `processed_churn-bigml-80.csv` | `churn_elbow.png`, `churn_clusters.png` |
| `random_forest_churn.py` | Random Forest Classifier | `processed_churn-bigml-80.csv` | `rf_feature_importance.png` |
| `svm_iris_classifier.py` | Support Vector Machine (SVM) | `Data Set For Task/1) iris.csv` | `svm_iris_decision_boundary.png` |
| `neural_network_churn.py` | Feed-Forward Neural Network | `processed_churn-bigml-80.csv` | `nn_loss_curve.png` |

---

## Models Explained Like You're 10 (ELI10)

### 1. Data Preprocessing (`task1.py`)
* **Datasets:** Iris, Stock Prices, Sentiment, House Prediction, and Churn (`churn-bigml-20` & `churn-bigml-80`).
* **ELI10 Explanation:** Before cooking a meal, you wash and chop the vegetables. Raw data is messy—it has missing values, text labels, and different number scales. This script automatically cleans all 6 datasets, fills missing blanks, converts text into numbers, and saves ready-to-use files into `preprocessed_data/`.
* **Real-Life Example:** Sorting and washing laundry by color before putting it into the washing machine so nothing gets ruined.

---

### 2. Linear Regression (`simple_linear_regression_model.py`)
* **Dataset:** `processed_4) house Prediction Data Set.csv`
* **ELI10 Explanation:** Imagine drawing a straight line through a bunch of dots on a graph to guess a number. If you know how big a house is and how many rooms it has, the AI draws a trend line to predict its exact price.
* **Real-Life Example:** Guessing how much a pizza will cost based on its size—bigger size means a higher price along a steady line.

---

### 3. K-Nearest Neighbors / KNN (`knn_classifier_code.py`)
* **Dataset:** `processed_2) Stock Prices Data Set.csv`
* **ELI10 Explanation:** "Tell me who your neighbors are, and I'll tell you who you are." When a new data point arrives, KNN looks at the `K` closest past examples and takes a majority vote to classify it.
* **Real-Life Example:** Moving to a new neighborhood—if your 5 closest neighbors all own golden retrievers, someone might guess you probably like dogs too.

---

### 4. Logistic Regression (`logistic_regression.py`)
* **Dataset:** `processed_churn-bigml-80.csv`
* **Output:** `roc_curve_churn.png`
* **ELI10 Explanation:** Unlike Linear Regression (which predicts an exact price), Logistic Regression predicts a **Yes or No** probability (0% to 100%). Here, it predicts whether a telecom customer will **Churn (leave the company)** or **Stay**.
* **Real-Life Example:** A weather app telling you there is an 85% chance of rain tomorrow—if it's over 50%, you take an umbrella ("Yes").

---

### 5. Decision Tree Classifier (`decision_tree.py`)
* **Dataset:** `processed_1) iris.csv`
* **Output:** `iris_decision_tree.png`
* **ELI10 Explanation:** Think of the game **"20 Questions."** The AI asks simple Yes/No questions like *"Is the petal length smaller than 2.45?"* to sort flowers into their exact species. We also **prune** the tree (`max_depth=3`) so it learns the big rules instead of memorizing the textbook.
* **Real-Life Example:** A doctor's checklist: *"Do you have a fever? → Yes → Do you have a cough? → Yes → Likely the flu."*

---

### 6. K-Means Clustering (`kmeans_churn.py`)
* **Dataset:** `processed_churn-bigml-80.csv`
* **Outputs:** `churn_elbow.png`, `churn_clusters.png`
* **ELI10 Explanation:** This is **Unsupervised Learning**—we hide the answers (`Churn`) and ask the AI to group similar customers together on its own. We use the **Elbow Method** to find the best number of groups (like VIP users, normal users, and frustrated users).
* **Real-Life Example:** Dumping a box of mixed LEGO bricks on the floor and automatically sorting them into piles by shape and size without looking at the instruction manual.

---

### 7. Random Forest Classifier (`random_forest_churn.py`)
* **Dataset:** `processed_churn-bigml-80.csv`
* **Output:** `rf_feature_importance.png`
* **ELI10 Explanation:** Asking one friend to guess how many jellybeans are in a jar might be wrong. Asking **200 friends** and taking the majority vote is way more accurate! A Random Forest builds hundreds of Decision Trees, tunes them with `GridSearchCV`, and even tells us which clues (like `Total day minutes` and `Customer service calls`) mattered most.
* **Real-Life Example:** A bank's fraud detection system where hundreds of mini-checkers vote on whether a 3 AM credit card swipe in another country is fraud.

---

### 8. Support Vector Machine / SVM (`svm_iris_classifier.py`)
* **Dataset:** `Data Set For Task/1) iris.csv`
* **Output:** `svm_iris_decision_boundary.png`
* **ELI10 Explanation:** Imagine red, blue, and yellow blocks scattered on a rug. SVM builds fences (**Decision Boundaries**) to separate them. A **Linear Kernel** uses straight wooden rulers, while an **RBF Kernel** uses stretchy rubber bands to curve around tricky groups. We use PCA to squash the 4D flower data into a 2D map.
* **Real-Life Example:** An email spam filter drawing a strict boundary line between "Spam" and "Important" emails based on the words inside.

---

### 9. Feed-Forward Neural Network (`neural_network_churn.py`)
* **Dataset:** `processed_churn-bigml-80.csv`
* **Output:** `nn_loss_curve.png`
* **ELI10 Explanation:** Built with **TensorFlow/Keras**, this works like a factory assembly line of mini-brains (**neurons**). Raw customer data enters the **Input Layer**, middle managers (**Hidden Layers**) spot complex patterns, and the boss (**Output Layer**) predicts if the customer will churn. Using **Backpropagation** and **EarlyStopping**, the network learns from its mistakes and stops training right at the sweet spot before it starts memorizing (overfitting).
* **Real-Life Example:** How your phone unlocks with Face ID or how a bank calculates your exact loan approval probability from dozens of financial signals.

---

## How to Run This Project

1. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn tensorflow# Codeveda-Internship