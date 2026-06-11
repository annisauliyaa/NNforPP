# ============================================
# MOUNT GOOGLE DRIVE FIRST
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================
# INSTALL AND IMPORT LIBRARIES
# ============================================
!pip install tensorflow==2.15.0 pandas==1.5.3 tensorflow-probability==0.23.0 -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import os
import time
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import mean_squared_error, mean_absolute_error

tfd = tfp.distributions
print("TensorFlow version:", tf.__version__)

# ============================================
# CORRECT PATH FOR GOOGLE COLAB
# ============================================
# Use the CORRECT Google Colab path (not I:/)
base_path = "/content/drive/My Drive/code Review Paper for NN -desy/Multivariate LGCP/Dataset"

# Verify the path exists
print(f"Checking path: {base_path}")
if not os.path.exists(base_path):
    print(f"ERROR: Path not found!")
    print("\nPlease make sure your Google Drive has this exact folder structure:")
    print("My Drive/")
    print("  └── code Review Paper for NN -desy/")
    print("      └── Multivariate LGCP/")
    print("          └── Dataset/")
    print("              ├── bubble_covars.csv")
    print("              └── bubble_full_temporal.csv")
    
    # List what's in your Drive to help debug
    drive_path = "/content/drive/My Drive"
    if os.path.exists(drive_path):
        print(f"\nContents of '/content/drive/My Drive':")
        for item in os.listdir(drive_path):
            print(f"  - {item}")
    raise FileNotFoundError("Please create the correct folder structure in Google Drive")

# Load data (using the correct filename as in your error message)
bubble_covars = pd.read_csv(f"{base_path}/bubble_covars.csv")
bubble_data = pd.read_csv(f"{base_path}/bubble_full_temporal.csv")

print("\n" + "="*50)
print("DATA LOADED SUCCESSFULLY!")
print("="*50)
print("Covariates shape:", bubble_covars.shape)
print("Bubble data shape:", bubble_data.shape)
print("Covariates columns:", bubble_covars.columns.tolist())
print("First 3 rows of bubble_data:")
print(bubble_data.head(3))

# ============================================
# DATA PREPROCESSING
# ============================================
# Parameters
n_species = 9  # 9 treatment combinations (3x3 factorial)
n_time = 8

# Extract species columns (sp_1 to sp_9)
species_cols = [f"sp_{i}" for i in range(1, n_species + 1)]

# Slice data - keep only time, cell_id, and species columns
bubble_data_processed = bubble_data.iloc[:, :2 + n_species]
bubble_data_processed = bubble_data_processed.loc[bubble_data_processed['time'] <= n_time]

# Get cell centers from covariates
if 'x_center' in bubble_covars.columns and 'y_center' in bubble_covars.columns:
    cell_centers = bubble_covars[['x_center', 'y_center']].copy()
else:
    print("Warning: Using first two columns as coordinates")
    cell_centers = bubble_covars.iloc[:, :2].copy()
    cell_centers.columns = ['x_center', 'y_center']

# Standardize spatial covariates
covars = (cell_centers - cell_centers.mean(axis=0)) / cell_centers.std(axis=0)

print(f"\nTotal observations: {bubble_data_processed.shape[0]}")
print(f"Species columns: {species_cols}")
print(f"Cell centers shape: {cell_centers.shape}")

# ============================================
# TRAIN-TEST SPLIT
# ============================================
n_obs = bubble_data_processed.shape[0]

# Set seed for reproducibility
np.random.seed(42)
test_idx = np.random.choice(n_obs, size=int(0.3 * n_obs), replace=False)
test_idx.sort()

# Split data
bubble_train = np.delete(bubble_data_processed.values, test_idx, axis=0)
bubble_test = bubble_data_processed.iloc[test_idx, :].values

# Split cell centers (subtract 1 because cell_id starts from 1)
cell_centers_train = cell_centers.iloc[bubble_train[:, 1].astype(int) - 1, :]
cell_centers_test = cell_centers.iloc[bubble_test[:, 1].astype(int) - 1, :]

# Split covariates
covars_train = covars.iloc[bubble_train[:, 1].astype(int) - 1, :]
covars_test = covars.iloc[bubble_test[:, 1].astype(int) - 1, :]

# Extract time
time_train = bubble_train[:, 0].reshape(-1, 1)
time_test = bubble_test[:, 0].reshape(-1, 1)

# Create dummy time (for compatibility)
dummy_time_train = np.ones((time_train.shape[0], 1))
dummy_time_test = np.ones((time_test.shape[0], 1))

# Remove time and cell id columns from counts
bubble_train_counts = np.delete(bubble_train, [0, 1], axis=1)
bubble_test_counts = np.delete(bubble_test, [0, 1], axis=1)

print("\n" + "="*50)
print("DATA SPLIT COMPLETE")
print("="*50)
print(f"Training shapes:")
print(f"  Counts: {bubble_train_counts.shape}")
print(f"  Covariates: {covars_train.shape}")
print(f"  Cell centers: {cell_centers_train.shape}")
print(f"  Time: {time_train.shape}")
print(f"\nTest shapes:")
print(f"  Counts: {bubble_test_counts.shape}")
print(f"  Covariates: {covars_test.shape}")

# ============================================
# BUILD NEURAL NETWORK MODEL (PDL)
# ============================================
tf.keras.backend.clear_session()

# Input layers
xi = Input(shape=(n_species,), name="counts_input")
zi = Input(shape=(2,), name="covariates_input")
ui = Input(shape=(2,), name="cell_centers")
ti = Input(shape=(1,), name="time_input")
dti = Input(shape=(1,), name="dummy_time")

# Merge all inputs
merged = Concatenate(axis=1)([xi, zi, ui, ti, dti])

# Hidden layers
h1 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(merged)
h1 = BatchNormalization()(h1)
h1 = Dropout(0.3)(h1)

h2 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(h1)
h2 = BatchNormalization()(h2)
h2 = Dropout(0.3)(h2)

h3 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(h2)
h3 = BatchNormalization()(h3)
h3 = Dropout(0.2)(h3)

# Output layer
output = Dense(n_species, activation='linear', name="output")(h3)

# Build model
model = Model(
    inputs=[xi, zi, ui, ti, dti],
    outputs=output,
    name="PDL_Multivariate_LGCP"
)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss='mse',
    metrics=['mae']
)

print("\n" + "="*50)
print("MODEL ARCHITECTURE")
print("="*50)
model.summary()

# ============================================
# LEARNING RATE SCHEDULER
# ============================================
def lr_scheduler(epoch, lr):
    if epoch < 25:
        return lr
    else:
        return max(lr * np.exp(-0.05), 1e-5)

lr_callback = LearningRateScheduler(lr_scheduler)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    lr_callback
]

# ============================================
# TRAIN MODEL
# ============================================
epochs = 30  # Reduced for faster testing, change to 50 for full training
batch_size = 200
save_path = "/content/drive/My Drive/code Review Paper for NN -desy/Multivariate LGCP/result"

os.makedirs(save_path, exist_ok=True)

print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50)

start_time = time.time()

# Convert to float32 for better performance
history = model.fit(
    x=[bubble_train_counts.astype(np.float32), 
       covars_train.values.astype(np.float32), 
       cell_centers_train.values.astype(np.float32), 
       time_train.astype(np.float32), 
       dummy_time_train.astype(np.float32)],
    y=bubble_train_counts.astype(np.float32),
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    validation_data=(
        [bubble_test_counts.astype(np.float32), 
         covars_test.values.astype(np.float32), 
         cell_centers_test.values.astype(np.float32), 
         time_test.astype(np.float32), 
         dummy_time_test.astype(np.float32)],
        bubble_test_counts.astype(np.float32)
    ),
    verbose=1
)

end_time = time.time()
training_time = (end_time - start_time) / 60
print(f"\nTraining time: {training_time:.2f} minutes")

# Save model
os.makedirs(f"{save_path}/model", exist_ok=True)
model.save(f"{save_path}/model/pdl_model.h5")
print(f"Model saved to {save_path}/model/pdl_model.h5")

# ============================================
# PLOT TRAINING HISTORY
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot MSE Loss
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=1.5)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=1.5)
axes[0].set_title('Model Loss per Epoch', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.4)

# Plot MAE
axes[1].plot(history.history['mae'], label='Train MAE', linewidth=1.5)
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=1.5)
axes[1].set_title('Model Mean Absolute Error per Epoch', fontsize=12)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(f"{save_path}/training_history.png", dpi=300)
plt.show()

# ============================================
# MODEL EVALUATION
# ============================================
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

eval_metrics = model.evaluate(
    x=[bubble_test_counts.astype(np.float32), 
       covars_test.values.astype(np.float32), 
       cell_centers_test.values.astype(np.float32),
       time_test.astype(np.float32), 
       dummy_time_test.astype(np.float32)],
    y=bubble_test_counts.astype(np.float32),
    batch_size=batch_size,
    verbose=1
)

print(f"\nEvaluation Results:")
print(f"  MSE: {eval_metrics[0]:.6f}")
print(f"  MAE: {eval_metrics[1]:.6f}")

# ============================================
# MAKE PREDICTIONS
# ============================================
y_pred = model.predict(
    [bubble_test_counts.astype(np.float32), 
     covars_test.values.astype(np.float32), 
     cell_centers_test.values.astype(np.float32),
     time_test.astype(np.float32), 
     dummy_time_test.astype(np.float32)],
    batch_size=batch_size,
    verbose=1
)

y_pred_df = pd.DataFrame(y_pred, columns=species_cols)
print("\nPrediction summary:")
print(y_pred_df.describe())

# ============================================
# VISUALIZE OBSERVED VS PREDICTED
# ============================================
test_cells = cell_centers_test.values
test_counts = bubble_test_counts

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
species_to_plot = [0, 4, 8]  # sp_1, sp_5, sp_9

for idx, sp_idx in enumerate(species_to_plot):
    # Observed
    sc1 = axes[0, idx].scatter(test_cells[:, 0], test_cells[:, 1], 
                               c=test_counts[:, sp_idx], s=3, cmap='jet', alpha=0.6, vmin=0, vmax=5)
    axes[0, idx].set_title(f'Observed - {species_cols[sp_idx]}', fontsize=10)
    axes[0, idx].set_xlabel('x_center')
    axes[0, idx].set_ylabel('y_center')
    plt.colorbar(sc1, ax=axes[0, idx])
    
    # Predicted
    sc2 = axes[1, idx].scatter(test_cells[:, 0], test_cells[:, 1],
                               c=y_pred[:, sp_idx], s=3, cmap='jet', alpha=0.6)
    axes[1, idx].set_title(f'Predicted - {species_cols[sp_idx]}', fontsize=10)
    axes[1, idx].set_xlabel('x_center')
    axes[1, idx].set_ylabel('y_center')
    plt.colorbar(sc2, ax=axes[1, idx])

plt.suptitle('Observed vs Predicted Bubble Intensity Maps', fontsize=14)
plt.tight_layout()
plt.savefig(f"{save_path}/observed_vs_predicted.png", dpi=300)
plt.show()

# ============================================
# ESTIMATE MODEL PARAMETERS
# ============================================
# Get mu from predictions
est_mu = y_pred.mean(axis=0)
est_mu_df = pd.DataFrame([est_mu], columns=species_cols)

print("\n" + "="*50)
print("ESTIMATED PARAMETERS")
print("="*50)
print("\n=== Mu Parameters (Base Intensity) ===")
print(est_mu_df.T)

# Extract theta (weights from dense layers)
theta_params = {}
dense_layers = [layer for layer in model.layers if 'dense' in layer.name.lower() and 'output' not in layer.name]

for layer in dense_layers:
    weights = layer.get_weights()
    if len(weights) > 0:
        theta_params[layer.name] = {
            'weights_mean': np.mean(weights[0]),
            'weights_std': np.std(weights[0])
        }

if theta_params:
    theta_df = pd.DataFrame([
        [k, v['weights_mean'], v['weights_std']]
        for k, v in theta_params.items()
    ], columns=['Layer', 'Mean_W', 'Std_W'])
    print("\n=== Theta Parameters (Dense Layer Weights) ===")
    print(theta_df)

# Extract beta (batch normalization coefficients)
beta_params = {}
bn_layers = [layer for layer in model.layers if 'batch_normalization' in layer.name.lower()]

for layer in bn_layers:
    weights = layer.get_weights()
    if len(weights) >= 2:
        beta_params[layer.name] = {
            'gamma_mean': np.mean(weights[0]),
            'beta_mean': np.mean(weights[1])
        }

if beta_params:
    beta_df = pd.DataFrame([
        [k, v['gamma_mean'], v['beta_mean']]
        for k, v in beta_params.items()
    ], columns=['Layer', 'Gamma_mean', 'Beta_mean'])
    print("\n=== Beta Parameters (BatchNorm Coefficients) ===")
    print(beta_df)

# ============================================
# VISUALIZE PARAMETER DISTRIBUTIONS
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Mu parameters
axes[0].bar(range(1, n_species + 1), est_mu, color='steelblue')
axes[0].set_title(r'$\hat{\mu}$ (Base Intensity)', fontsize=12)
axes[0].set_xlabel('Species Index')
axes[0].set_ylabel('Mean Value')
axes[0].grid(True, linestyle='--', alpha=0.4)

# Theta parameters
if theta_params:
    theta_means = [v['weights_mean'] for v in theta_params.values()]
    axes[1].bar(range(len(theta_means)), theta_means, color='coral')
    axes[1].set_title(r'$\hat{\theta}$ (Dense Layer Weights)', fontsize=12)
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Mean Weight')
    axes[1].grid(True, linestyle='--', alpha=0.4)
else:
    axes[1].text(0.5, 0.5, 'No dense layers found', ha='center', va='center')

# Beta parameters
if beta_params:
    x = np.arange(len(beta_params))
    width = 0.35
    gamma_means = [v['gamma_mean'] for v in beta_params.values()]
    beta_means = [v['beta_mean'] for v in beta_params.values()]
    axes[2].bar(x - width/2, gamma_means, width, label='Gamma', color='seagreen')
    axes[2].bar(x + width/2, beta_means, width, label='Beta', color='goldenrod')
    axes[2].set_title(r'$\hat{\beta}$ (BatchNorm Coefficients)', fontsize=12)
    axes[2].set_xlabel('Layer Index')
    axes[2].set_ylabel('Mean Value')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.4)
else:
    axes[2].text(0.5, 0.5, 'No batch norm layers found', ha='center', va='center')

plt.tight_layout()
plt.savefig(f"{save_path}/parameter_estimates.png", dpi=300)
plt.show()

# ============================================
# SAVE ALL RESULTS
# ============================================
# Create output directories
os.makedirs(f"{save_path}/est_mu", exist_ok=True)
os.makedirs(f"{save_path}/est_theta", exist_ok=True)
os.makedirs(f"{save_path}/est_beta", exist_ok=True)
os.makedirs(f"{save_path}/predictions", exist_ok=True)

# Save estimates
est_mu_df.to_csv(f"{save_path}/est_mu/est_mu.csv", index=False)
if theta_params:
    theta_df.to_csv(f"{save_path}/est_theta/est_theta.csv", index=False)
if beta_params:
    beta_df.to_csv(f"{save_path}/est_beta/est_beta.csv", index=False)
y_pred_df.to_csv(f"{save_path}/predictions/predictions.csv", index=False)

# Save metrics
metrics_df = pd.DataFrame({
    'training_time_minutes': [training_time],
    'mse_test': [eval_metrics[0]],
    'mae_test': [eval_metrics[1]],
    'final_train_loss': [history.history['loss'][-1]],
    'final_val_loss': [history.history['val_loss'][-1]]
})
metrics_df.to_csv(f"{save_path}/evaluation_metrics.csv", index=False)

print("\n" + "="*60)
print("RESULTS SAVED SUCCESSFULLY!")
print("="*60)
print(f"  Location: {save_path}")
print(f"  - Model: {save_path}/model/")
print(f"  - Mu estimates: {save_path}/est_mu/")
print(f"  - Theta estimates: {save_path}/est_theta/")
print(f"  - Beta estimates: {save_path}/est_beta/")
print(f"  - Predictions: {save_path}/predictions/")
print(f"  - Metrics: {save_path}/evaluation_metrics.csv")
print("="*60)

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"Training time: {training_time:.2f} minutes")
print(f"Test MSE: {eval_metrics[0]:.8f}")
print(f"Test MAE: {eval_metrics[1]:.8f}")
print(f"Final Training Loss: {history.history['loss'][-1]:.8f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.8f}")
print(f"Number of trainable parameters: {model.count_params():,}")

# ============================================
# PREDICTED INTENSITY MAPS 
# ============================================

from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

# Treatment labels for 9 combinations
treatment_labels = [
    '(a) Frother 5ppm, Airflow 5 L/min',
    '(b) Frother 5ppm, Airflow 8 L/min', 
    '(c) Frother 5ppm, Airflow 10 L/min',
    '(d) Frother 10ppm, Airflow 5 L/min',
    '(e) Frother 10ppm, Airflow 8 L/min',
    '(f) Frother 10ppm, Airflow 10 L/min',
    '(g) Frother 15ppm, Airflow 5 L/min',
    '(h) Frother 15ppm, Airflow 8 L/min',
    '(i) Frother 15ppm, Airflow 10 L/min'
]

# Get test data (from your main code)
test_cells = cell_centers_test.values
y_pred = y_pred  # from your prediction

# Create grid for heatmap
grid_res = 50
xi = np.linspace(test_cells[:, 0].min(), test_cells[:, 0].max(), grid_res)
yi = np.linspace(test_cells[:, 1].min(), test_cells[:, 1].max(), grid_res)
xi_grid, yi_grid = np.meshgrid(xi, yi)

# Create figure 3x3 for predicted maps
fig_pred, axes_pred = plt.subplots(3, 3, figsize=(18, 15))
axes_pred = axes_pred.flatten()

for idx, sp in enumerate(species_cols):
    sp_idx = species_cols.index(sp)
    pred_data = y_pred[:, sp_idx]
    
    # Use log1p transformation for better visualization
    pred_log = np.log1p(pred_data)
    zi_pred = griddata((test_cells[:, 0], test_cells[:, 1]), pred_log, 
                       (xi_grid, yi_grid), method='nearest', fill_value=0)
    
    # Plot heatmap
    im = axes_pred[idx].imshow(zi_pred.T, extent=[xi.min(), xi.max(), yi.min(), yi.max()], 
                               origin='lower', cmap='hot', aspect='auto')
    axes_pred[idx].set_title(treatment_labels[idx], fontsize=11, fontweight='bold')
    axes_pred[idx].set_xlabel('X Coordinate (mm)', fontsize=10)
    axes_pred[idx].set_ylabel('Y Coordinate (mm)', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes_pred[idx], shrink=0.7)
    cbar.set_label('log(1 + Predicted Intensity)', fontsize=9)

# Main title
fig_pred.suptitle('Predicted Bubble Intensity Maps from PDL Model\nfor Nine Treatment Combinations', 
                  fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

plt.show()
# Save figure
save_path = "/content/drive/My Drive/code Review Paper for NN -desy/Multivariate LGCP/result"
plt.savefig(f"{save_path}/Predicted_Intensity_Maps.png", dpi=300, bbox_inches='tight')