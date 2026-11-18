import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
file_name = "preprocessed_data/processed_churn-bigml-80.csv"
if not os.path.exists(file_name):
    print(f"[ERROR] Could not find '{file_name}'. Ensure it is in the current folder.")
    exit()

df = pd.read_csv(file_name)
print(f"[INFO] Successfully loaded {file_name}")

# Separate Features (X) and Target (y)
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split the data: 80% for training the network, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CRITICAL FOR NEURAL NETWORKS: Scale the data
# Neural networks crash or fail to learn if numbers are too big. We must scale 
# everything to small numbers (usually between -1 and 1).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. DESIGN THE NEURAL NETWORK ARCHITECTURE
# ==========================================
print("\n[INFO] Building the Neural Network...")
model = Sequential()

model.add(Input(shape=(X_train_scaled.shape[1],)))
# Input Layer & First Hidden Layer
# We add 16 "neurons" (mini-brains). 'relu' is an activation function that helps the network learn complex patterns.
# input_shape tells the network how many columns are coming in.
model.add(Dense(16, activation='relu'))

# Second Hidden Layer
# 8 neurons to narrow down the clues.
model.add(Dense(8, activation='relu'))

# Output Layer
# 1 single neuron because we only need a Yes/No answer (Churn or Not). 
# 'sigmoid' squashes the final output into a probability between 0% and 100%.
model.add(Dense(1, activation='sigmoid'))

# Compile the Model
# 'adam' is the optimizer that manages the Backpropagation (learning from mistakes).
# 'binary_crossentropy' is the math formula used to calculate the Loss (mistake score).
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==========================================
# 3. TRAIN THE MODEL (Backpropagation in action)
# ==========================================
print("\n[INFO] Training the model. Watch the Loss go down and Accuracy go up!")
# epochs=50 means the AI will review the entire dataset 50 times.
# batch_size=32 means it learns from 32 customers at a time before updating its rules.

# Create the "smart teacher"
early_stop = EarlyStopping(
    monitor='val_loss',        # Watch the orange line (Validation Loss)
    patience=5,                # If the score doesn't improve for 5 rounds, stop training
    restore_best_weights=True  # Rewind the AI's brain to the absolute best epoch (e.g., Epoch 25)
)

print("\n[INFO] Training the model with Early Stopping...")

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,                 # Set high; EarlyStopping will cut it short automatically
    batch_size=32,
    callbacks=[early_stop],     # <--- Add the callback here
    verbose=1
)

# ==========================================
# 4. EVALUATE AND VISUALIZE
# ==========================================
# Check final score on the test data
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n--- Final Test Performance ---")
print(f"Accuracy: {test_accuracy * 100:.2f}%")
print(f"Loss:     {test_loss:.4f}")

# Plot the Training vs Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss (Mistakes on practice test)')
plt.plot(history.history['val_loss'], label='Validation Loss (Mistakes on real exam)')
plt.title('Neural Network Learning Curve (Loss)')
plt.xlabel('Epochs (Practice Rounds)')
plt.ylabel('Loss (Mistake Score - Lower is Better)')
plt.legend()
plt.grid(True)

image_name = "nn_loss_curve.png"
plt.savefig(image_name)
print(f"\n[INFO] Loss curve saved as '{image_name}'.")