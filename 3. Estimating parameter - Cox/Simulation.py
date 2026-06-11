# ============================================
# INSTALL AND IMPORT FOR SIMULATION
# ============================================
!pip install tensorflow==2.15.0 pandas==1.5.3 tensorflow-probability==0.23.0 -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import time

print("TensorFlow version:", tf.__version__)

# ============================================
# GENERATE SIMULATION DATA (9 CLASSES)
# ============================================
np.random.seed(42)

# Parameters
n_samples = 5000
grid_size = 60
n_classes = 9
latent_dim = 4

print("Generating simulation data for 9 classes...")

# Generate spatial coordinates on a grid
x_coords = np.linspace(0, 100, grid_size)
y_coords = np.linspace(0, 100, grid_size)
xx, yy = np.meshgrid(x_coords, y_coords)
all_coords = np.column_stack([xx.ravel(), yy.ravel()])
n_cells = len(all_coords)

# Generate class centers (spatial means)
class_centers_x = np.array([20, 30, 40, 50, 60, 70, 80, 30, 70])
class_centers_y = np.array([30, 30, 30, 50, 50, 50, 50, 70, 70])

# Generate intensities based on distance to class centers
intensities = np.zeros((n_cells, n_classes))
for i in range(n_classes):
    distances = np.sqrt((all_coords[:, 0] - class_centers_x[i])**2 + 
                        (all_coords[:, 1] - class_centers_y[i])**2)
    intensities[:, i] = np.exp(-distances / 25) * np.random.gamma(2, 0.5, n_cells)

# Generate counts from Poisson distribution
counts = np.random.poisson(intensities)

# Assign class labels based on maximum intensity
class_labels = np.argmax(counts, axis=1)

print(f"Generated {n_cells} spatial cells")
print(f"Class distribution:")
unique, counts_class = np.unique(class_labels, return_counts=True)
for i in range(n_classes):
    print(f"  Class {i+1}: {counts_class[i]} cells ({counts_class[i]/n_cells*100:.1f}%)")

# ============================================
# CREATE TRAIN-TEST SPLIT
# ============================================
from sklearn.model_selection import train_test_split

# Features: coordinates + counts
X_coords = all_coords
X_counts = counts
X = np.hstack([X_coords, X_counts])

# Target: class labels
y = class_labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Normalize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes=n_classes)
y_test_cat = to_categorical(y_test, num_classes=n_classes)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Input features: {X_train.shape[1]}")

# ============================================
# BUILD CLASSIFICATION MODEL
# ============================================
tf.keras.backend.clear_session()

# Input layer
inputs = Input(shape=(X_train.shape[1],), name="input")

# Hidden layers
x = Dense(128, activation='relu')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

# Output layer
outputs = Dense(n_classes, activation='softmax', name="class_output")(x)

# Build model
model = Model(inputs=inputs, outputs=outputs, name="SpatialClassifier")

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# TRAIN CLASSIFICATION MODEL
# ============================================
epochs = 50
batch_size = 64
save_path = "simulation_result"

os.makedirs(save_path, exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

start_time = time.time()

history = model.fit(
    X_train_scaled, y_train_cat,
    validation_data=(X_test_scaled, y_test_cat),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\nTraining time: {training_time:.2f} seconds")

# ============================================
# EVALUATE MODEL
# ============================================
# Evaluate
test_loss, test_acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.6f}")

# Predictions
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float32') / cm.sum(axis=1)[:, np.newaxis]

# ============================================
# VISUALIZATION - TRAINING HISTORY
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=1.5)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=1.5)
axes[0].set_title('Model Loss per Epoch', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.4)

# Accuracy
axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=1.5)
axes[1].plot(history.history['val_accuracy'], label='Validation Acc', linewidth=1.5)
axes[1].set_title('Model Accuracy per Epoch', fontsize=12)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(f"{save_path}/training_history.png", dpi=300)
plt.show()

# ============================================
# CONFUSION MATRIX VISUALIZATION
# ============================================
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[f'Class {i+1}' for i in range(n_classes)],
            yticklabels=[f'Class {i+1}' for i in range(n_classes)])
plt.title('Confusion Matrix - Observed vs Predicted', fontsize=14)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.tight_layout()
plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300)
plt.show()

# ============================================
# CLASSIFICATION REPORT
# ============================================
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, 
                            target_names=[f'Class_{i+1}' for i in range(n_classes)]))

# ============================================
# SPATIAL VISUALIZATION OF PREDICTIONS
# ============================================
# Get predictions for all cells
all_pred_proba = model.predict(scaler.transform(X), verbose=0)
all_pred = np.argmax(all_pred_proba, axis=1)

# Reshape to grid for visualization
pred_grid = all_pred.reshape(grid_size, grid_size)
true_grid = class_labels.reshape(grid_size, grid_size)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# True labels
im1 = axes[0].imshow(true_grid, cmap='tab10', interpolation='nearest')
axes[0].set_title('True Class Labels', fontsize=12)
axes[0].set_xlabel('X coordinate')
axes[0].set_ylabel('Y coordinate')
plt.colorbar(im1, ax=axes[0], ticks=range(n_classes))

# Predicted labels
im2 = axes[1].imshow(pred_grid, cmap='tab10', interpolation='nearest')
axes[1].set_title('Predicted Class Labels', fontsize=12)
axes[1].set_xlabel('X coordinate')
axes[1].set_ylabel('Y coordinate')
plt.colorbar(im2, ax=axes[1], ticks=range(n_classes))

plt.suptitle('Spatial Classification Results - 9 Classes', fontsize=14)
plt.tight_layout()
plt.savefig(f"{save_path}/spatial_classification.png", dpi=300)
plt.show()

# ============================================
# PER-CLASS INTENSITY VISUALIZATION
# ============================================
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i in range(n_classes):
    intensity_grid = intensities[:, i].reshape(grid_size, grid_size)
    im = axes[i].imshow(intensity_grid, cmap='hot', interpolation='bilinear')
    axes[i].set_title(f'Class {i+1} Intensity', fontsize=10)
    axes[i].set_xlabel('X')
    axes[i].set_ylabel('Y')
    plt.colorbar(im, ax=axes[i])

plt.suptitle('Ground Truth Intensity Maps for 9 Classes', fontsize=14)
plt.tight_layout()
plt.savefig(f"{save_path}/class_intensities.png", dpi=300)
plt.show()

# ============================================
# SAVE RESULTS
# ============================================
# Save model
os.makedirs(f"{save_path}/model", exist_ok=True)
model.save(f"{save_path}/model/classifier_model.h5")

# Save predictions
results_df = pd.DataFrame({
    'x_coord': X[:, 0],
    'y_coord': X[:, 1],
    'true_class': class_labels,
    'predicted_class': all_pred
})
results_df.to_csv(f"{save_path}/predictions.csv", index=False)

# Save metrics
metrics_df = pd.DataFrame({
    'accuracy': [test_acc],
    'loss': [test_loss],
    'training_time_sec': [training_time]
})
metrics_df.to_csv(f"{save_path}/metrics.csv", index=False)

print(f"\nResults saved to: {save_path}/")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("SIMULATION STUDY - FINAL SUMMARY")
print("="*60)
print(f"Number of samples: {n_cells}")
print(f"Number of classes: {n_classes}")
print(f"Grid size: {grid_size}x{grid_size}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Training time: {training_time:.2f} seconds")
print("="*60)