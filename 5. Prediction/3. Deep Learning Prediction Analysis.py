# =====================================================
# DEEP LEARNING MODEL FOR JAVA ISLAND
# Spatial Grid: 2x7 = 14 zones | Resolution: 1.5° x 1.5°
# Period: 2005-2025
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Nadam

# =====================================================
# LOAD DATA
# NOTE: Adjust this path to match your local data directory
# =====================================================
from google.colab import drive
drive.mount('/content/drive')

BASE_PATH = "/content/drive/MyDrive/penelitian/paper 1/-HIMAWAN-Prediction of intensity and location of seismic events using deep learning"
PATH_LIST = f"{BASE_PATH}/max_values_per_TT_java_2x7_2005_2025.csv"
PATH_MATRIX = f"{BASE_PATH}/matrix_data_CNN_java_2x7_2005_2025.csv"

data_list = pd.read_csv(PATH_LIST)
data_matrix = pd.read_csv(PATH_MATRIX)
data_matrix = np.log(data_matrix + 1e-8)

# =====================================================
# LSTM DATASET PREPARATION
# =====================================================
def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps_lstm = 30
X, y = create_dataset(data_list["log_MaxValue"].values, time_steps_lstm)

split_idx_lstm = int(len(X) * 0.75)
X_train_lstm, X_test_lstm = X[:split_idx_lstm], X[split_idx_lstm:]
y_train_lstm, y_test_lstm = y[:split_idx_lstm], y[split_idx_lstm:]
X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

# =====================================================
# CNN DATASET PREPARATION
# =====================================================
def create_sequences(features, targets, time_steps):
    X, y = [], []
    for i in range(len(features) - time_steps):
        sequence = np.array([features[i+j].reshape(2, 7) for j in range(time_steps)])
        X.append(sequence.transpose((1, 2, 0)))
        y.append(targets[i + time_steps])
    return np.array(X), np.array(y)

features = data_matrix.iloc[:, :14].values
targets = data_matrix.iloc[:, 14:].values

time_steps_cnn = 20
X, y = create_sequences(features, targets, time_steps_cnn)

split_idx_cnn = int(len(X) * 0.75)
train_features, test_features = X[:split_idx_cnn], X[split_idx_cnn:]
train_targets, test_targets = y[:split_idx_cnn], y[split_idx_cnn:]

data_matrix['MaxValue'] = data_matrix.iloc[:, :14].max(axis=1)

# =====================================================
# LSTM 10x RUNS
# =====================================================
losses = []
val_losses = []
r2_scores = []
maes = []
best_epochs = []

for i in range(10):
    print(f"Training run {i+1}/10")
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(time_steps_lstm, 1)))
    model.add(Dense(1))
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=128,
                        validation_data=(X_test_lstm, y_test_lstm), verbose=0, callbacks=[es])
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    best_epoch = np.argmin(history.history['val_loss']) + 1
    predictions_model = model.predict(X_test_lstm).flatten()
    r2_model = r2_score(y_test_lstm, predictions_model)
    mae_model = mean_absolute_error(y_test_lstm, predictions_model)
    losses.append(final_train_loss)
    val_losses.append(final_val_loss)
    r2_scores.append(r2_model)
    maes.append(mae_model)
    best_epochs.append(best_epoch)
    print(f"Training Loss: {final_train_loss:.4f}")
    print(f"Validation Loss: {final_val_loss:.4f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"R^2: {r2_model:.4f}")
    print(f"MAE: {mae_model:.4f}")

avg_loss = np.mean(losses)
std_loss = np.std(losses)
avg_val_loss = np.mean(val_losses)
std_val_loss = np.std(val_losses)
avg_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
avg_mae = np.mean(maes)
std_mae = np.std(maes)
avg_best_epoch = np.mean(best_epochs)
std_best_epoch = np.std(best_epochs)

print(f"\nAverage Training Loss: {avg_loss:.4f}, Std: {std_loss:.4f}")
print(f"Average Validation Loss: {avg_val_loss:.4f}, Std: {std_val_loss:.4f}")
print(f"Average R^2: {avg_r2:.4f}, Std: {std_r2:.4f}")
print(f"Average MAE: {avg_mae:.4f}, Std: {std_mae:.4f}")
print(f"Average Best Epoch: {avg_best_epoch:.2f}, Std: {std_best_epoch:.2f}")

# =====================================================
# LSTM BASELINE VS OPTIMIZED
# =====================================================
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(time_steps_lstm, 1)))
model.add(Dense(1))
optimizer = Adam()
model.compile(optimizer=optimizer, loss='mean_squared_error')
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=128, validation_data=(X_test_lstm, y_test_lstm), verbose=0, callbacks=[es])

model_optimized = Sequential()
model_optimized.add(LSTM(100, activation='tanh', input_shape=(time_steps_lstm, 1), recurrent_initializer='random_normal'))
model_optimized.add(Dense(1))
optimizer = Nadam()
model_optimized.compile(optimizer=optimizer, loss='mean_squared_error')
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history_opt = model_optimized.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=128, validation_data=(X_test_lstm, y_test_lstm), verbose=0, callbacks=[es])

val_loss_model_1 = history.history['val_loss']
val_loss_model_2 = history_opt.history['val_loss']
len1 = len(val_loss_model_1)
len2 = len(val_loss_model_2)
min_epochs = min(len1, len2)
val_loss_model_1 = val_loss_model_1[:min_epochs]
val_loss_model_2 = val_loss_model_2[:min_epochs]
epochs = range(1, min_epochs + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, val_loss_model_1, label='LSTM-Baseline', marker='o', linewidth=1.5, markersize=4)
plt.plot(epochs, val_loss_model_2, label='LSTM-Optimized', marker='x', linewidth=1.5, markersize=4)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Validation Loss (MSE)', fontsize=14)
plt.title('LSTM Validation Loss Comparison (2×7 Grid)', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================================
# LSTM PREDICTIONS VS ACTUAL
# =====================================================
predictions_baseline = model.predict(X_test_lstm).flatten()
predictions_optimized = model_optimized.predict(X_test_lstm).flatten()

mse_baseline = mean_squared_error(y_test_lstm, predictions_baseline)
r2_baseline = r2_score(y_test_lstm, predictions_baseline)
mse_optimized = mean_squared_error(y_test_lstm, predictions_optimized)
r2_optimized = r2_score(y_test_lstm, predictions_optimized)

plt.figure(figsize=(14, 6))
plt.plot(range(len(y_test_lstm)), y_test_lstm, color='black', label='Actual Values', linewidth=1.5)
plt.plot(range(len(predictions_baseline)), predictions_baseline, color='blue', label=f'LSTM_Baseline (MSE: {mse_baseline:.2f}, R2: {r2_baseline:.2f})', linewidth=1.5, alpha=0.7)
plt.plot(range(len(predictions_optimized)), predictions_optimized, color='orange', label=f'LSTM_Optimized (MSE: {mse_optimized:.2f}, R2: {r2_optimized:.2f})', linewidth=1.5, alpha=0.7)
plt.xlabel('Time Point/Index', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.title('Performance of LSTM Models vs Actual Values', fontsize=16)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================================
# CNN CLASSIFICATION FOR JAVA ISLAND (10x RUNS)
# =====================================================
data_matrix = data_matrix.fillna(-1e9).astype(float)
zone_columns = [f"Zone_{i}_{j}" for i in range(2) for j in range(7)]
features_raw = data_matrix[zone_columns].values

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_raw)

max_zone_indices = np.argmax(features_scaled, axis=1)
targets_oh = tf.keras.utils.to_categorical(max_zone_indices, num_classes=14)

unique, counts = np.unique(max_zone_indices, return_counts=True)
print("\nClass distribution (14 zones):")
for i in range(14):
    cnt = counts[unique == i][0] if i in unique else 0
    print(f"  Zone {i}: {cnt} samples")

time_steps = 20
def create_sequences(features, targets, time_steps):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:i+time_steps].reshape(time_steps, 2, 7))
        y.append(targets[i + time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(features_scaled, targets_oh, time_steps)
X = np.expand_dims(X, axis=-1)

split_idx = int(0.75 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_features_static = np.transpose(X_train.squeeze(-1), (0, 2, 3, 1))
test_features_static = np.transpose(X_test.squeeze(-1), (0, 2, 3, 1))

class_weights = {}
total = len(y_train)
for i in range(14):
    if i in unique:
        class_weights[i] = total / (len(unique) * counts[np.where(unique == i)[0][0]])
    else:
        class_weights[i] = 0.0

losses, val_losses, accs, precisions, recalls, f1_scores, best_epochs = [], [], [], [], [], [], []

for run in range(10):
    print(f"\nTraining run {run+1}/10")
    model = Sequential([
        Conv2D(32, (2, 3), activation='relu', padding='same', input_shape=(2, 7, time_steps)),
        MaxPooling2D((1, 2), padding='same'),
        Dropout(0.3),
        Conv2D(64, (2, 2), activation='relu', padding='same'),
        MaxPooling2D((1, 2), padding='same'),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(14, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    history = model.fit(train_features_static, y_train, validation_data=(test_features_static, y_test), epochs=100, batch_size=32, class_weight=class_weights, callbacks=[es], verbose=0)
    preds = model.predict(test_features_static, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_acc = history.history['val_accuracy'][-1]
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    precision = precision_score(true_classes, pred_classes, average='weighted', zero_division=0)
    recall = recall_score(true_classes, pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(true_classes, pred_classes, average='weighted', zero_division=0)
    losses.append(final_train_loss)
    val_losses.append(final_val_loss)
    accs.append(final_acc)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    best_epochs.append(best_epoch)
    print(f"Val Acc: {final_acc:.4f}, F1: {f1:.4f}")

print("\n=== FINAL SUMMARY (10 Runs) ===")
print(f"Average Accuracy:     {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Average F1 Score:     {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Average Precision:    {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Average Recall:       {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"Average Best Epoch:   {np.mean(best_epochs):.2f} ± {np.std(best_epochs):.2f}")

# =====================================================
# CNN BASELINE VS OPTIMIZED
# =====================================================
model_cnn_base = Sequential([
    Conv2D(32, (2, 3), activation='relu', padding='same', input_shape=(2, 7, time_steps)),
    MaxPooling2D((1, 2), padding='same'),
    Dropout(0.3),
    Conv2D(64, (2, 2), activation='relu', padding='same'),
    MaxPooling2D((1, 2), padding='same'),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(14, activation='softmax')
])
model_cnn_base.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
hist_base = model_cnn_base.fit(train_features_static, y_train, validation_data=(test_features_static, y_test), epochs=100, batch_size=32, callbacks=[es], verbose=0)

model_cnn_opt = Sequential([
    Conv2D(64, (2, 3), activation='relu', padding='same', input_shape=(2, 7, time_steps)),
    MaxPooling2D((1, 2), padding='same'),
    Dropout(0.25),
    Conv2D(128, (1, 2), activation='relu', padding='same'),
    MaxPooling2D((1, 2), padding='same'),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(14, activation='softmax')
])
model_cnn_opt.compile(optimizer=Nadam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
hist_opt = model_cnn_opt.fit(train_features_static, y_train, validation_data=(test_features_static, y_test), epochs=100, batch_size=32, callbacks=[es], verbose=0)

val_acc_base = hist_base.history['val_accuracy']
val_acc_opt = hist_opt.history['val_accuracy']
min_epochs = min(len(val_acc_base), len(val_acc_opt))
val_acc_base = val_acc_base[:min_epochs]
val_acc_opt = val_acc_opt[:min_epochs]
epochs = range(1, min_epochs + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, val_acc_base, label='Baseline CNN', marker='o', linewidth=1.5, markersize=4)
plt.plot(epochs, val_acc_opt, label='Optimized CNN', marker='x', linewidth=1.5, markersize=4)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Validation Accuracy', fontsize=14)
plt.title('CNN Validation Accuracy Comparison (2×7 Grid = 14 Zones)', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

val_loss_base = hist_base.history['val_loss']
val_loss_opt = hist_opt.history['val_loss']
min_epochs = min(len(val_loss_base), len(val_loss_opt))
val_loss_base = val_loss_base[:min_epochs]
val_loss_opt = val_loss_opt[:min_epochs]
epochs = range(1, min_epochs + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, val_loss_base, label='Baseline CNN', marker='o', linewidth=1.5, markersize=4)
plt.plot(epochs, val_loss_opt, label='Optimized CNN', marker='x', linewidth=1.5, markersize=4)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Validation Loss', fontsize=14)
plt.title('CNN Validation Loss Comparison (2×7 Grid = 14 Zones)', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================================
# CNN ZONE PROBABILITIES VISUALIZATION
# =====================================================
predictions = model_cnn_base.predict(test_features_static, verbose=0)
predictions_optimized = model_cnn_opt.predict(test_features_static, verbose=0)
true_classes = np.argmax(y_test, axis=1)
pred_classes = np.argmax(predictions, axis=1)
pred_classes_opt = np.argmax(predictions_optimized, axis=1)

acc_base = accuracy_score(true_classes, pred_classes)
acc_opt = accuracy_score(true_classes, pred_classes_opt)
print(f"\nBaseline CNN Accuracy: {acc_base:.4f}")
print(f"Optimized CNN Accuracy: {acc_opt:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for zone in range(14):
    axes[0].plot(predictions[-100:, zone], label=f'Zone{zone+1}', alpha=0.7, linewidth=1.5)
axes[0].set_ylabel('Probability', fontsize=14)
axes[0].set_title('CNN Baseline - Zone Probabilities (Last 100 Days)', fontsize=16)
axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

for zone in range(14):
    axes[1].plot(predictions_optimized[-100:, zone], label=f'Zone{zone+1}', alpha=0.7, linewidth=1.5)
axes[1].set_xlabel('Days', fontsize=14)
axes[1].set_ylabel('Probability', fontsize=14)
axes[1].set_title('CNN Optimized - Zone Probabilities (Last 100 Days)', fontsize=16)
axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
plt.tight_layout()
plt.show()

# =====================================================
# ENAS - JAVA ISLAND 2x7 GRID
# Neural Architecture Search for LSTM, CNN, and Multi-output Models
# =====================================================

import random

# =====================================================
# PART 1: ENAS FOR LSTM (REGRESSION)
# =====================================================

seed = 22
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

search_space_lstm = {
    'num_neurons': [10, 20, 30, 40, 50, 64, 80, 100],
    'num_layer': [1, 2],
    'optimizer': ['adam', 'nadam', 'rmsprop'],
    'recurrent_init': ['random_normal', 'glorot_uniform', 'orthogonal']
}

def create_random_architecture_lstm(search_space):
    return {
        'num_neurons_1': random.choice(search_space['num_neurons']),
        'num_neurons_2': random.choice(search_space['num_neurons']),
        'optimizer': random.choice(search_space['optimizer']),
        'num_layer': random.choice(search_space['num_layer']),
        'recurrent_init': random.choice(search_space['recurrent_init'])
    }

def evaluate_architecture_lstm(architecture, X_train, y_train, X_val, y_val):
    model = Sequential()
    if architecture['num_layer'] == 1:
        model.add(LSTM(architecture['num_neurons_1'], activation='tanh',
                       input_shape=(time_steps_lstm, 1),
                       recurrent_initializer=architecture['recurrent_init']))
    else:
        model.add(LSTM(architecture['num_neurons_1'], activation='tanh',
                       input_shape=(time_steps_lstm, 1),
                       recurrent_initializer=architecture['recurrent_init'],
                       return_sequences=True))
        model.add(LSTM(architecture['num_neurons_2'], activation='tanh',
                       recurrent_initializer=architecture['recurrent_init']))
    model.add(Dense(1))

    optimizer_map = {'adam': Adam(), 'nadam': Nadam(), 'rmsprop': RMSprop()}
    model.compile(optimizer=optimizer_map[architecture['optimizer']], loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0,
              callbacks=[es], validation_data=(X_val, y_val))

    y_pred = model.predict(X_val, verbose=0)
    return mean_squared_error(y_val, y_pred)

population_size = 10
num_generations = 10
population = [create_random_architecture_lstm(search_space_lstm) for _ in range(population_size)]

for generation in range(num_generations):
    print(f"LSTM ENAS Generation {generation + 1}")
    fitnesses = [evaluate_architecture_lstm(arch, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm) for arch in population]
    best_idx = np.argmin(fitnesses)
    parents = [population[best_idx]] * population_size
    population = [create_random_architecture_lstm(search_space_lstm) for _ in range(population_size)]
    population[0] = parents[0]

best_lstm = min(population, key=lambda arch: evaluate_architecture_lstm(arch, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm))
print("Best LSTM Architecture from ENAS:", best_lstm)

# =====================================================
# PART 2: ENAS FOR CNN (CLASSIFICATION)
# =====================================================

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

search_space_cnn = {
    'num_filters': [(32, 64), (64, 128), (32, 64, 128)],
    'dropout_rate': [0.2, 0.3, 0.4],
    'optimizer': ['adam', 'nadam'],
    'dense_units': [64, 128, 256]
}

def create_random_architecture_cnn(search_space):
    return {
        'num_filters': random.choice(search_space['num_filters']),
        'dropout_rate': random.choice(search_space['dropout_rate']),
        'optimizer': random.choice(search_space['optimizer']),
        'dense_units': random.choice(search_space['dense_units'])
    }

def evaluate_architecture_cnn(architecture, X_train, y_train, X_val, y_val):
    model = Sequential()
    for i, num_filters in enumerate(architecture['num_filters']):
        if i == 0:
            model.add(Conv2D(num_filters, (2, 3),
                            activation='relu', padding='same',
                            input_shape=(2, 7, time_steps_cnn)))
        else:
            model.add(Conv2D(num_filters, (1, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D((1, 2) if num_filters > 32 else (1, 1), padding='same'))
        model.add(Dropout(architecture['dropout_rate']))

    model.add(Flatten())
    model.add(Dense(architecture['dense_units'], activation='relu'))
    model.add(Dropout(architecture['dropout_rate']))
    model.add(Dense(14, activation='softmax'))

    optimizer_map = {'adam': Adam(), 'nadam': Nadam()}
    model.compile(optimizer=optimizer_map[architecture['optimizer']],
                  loss='categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0,
              callbacks=[es], validation_data=(X_val, y_val))

    _, acc = model.evaluate(X_val, y_val, verbose=0)
    return acc

population = [create_random_architecture_cnn(search_space_cnn) for _ in range(population_size)]

for generation in range(num_generations):
    print(f"CNN ENAS Generation {generation + 1}")
    fitnesses = [evaluate_architecture_cnn(arch, train_features_static, y_train, test_features_static, y_test) for arch in population]
    best_idx = np.argmax(fitnesses)
    parents = [population[best_idx]] * population_size
    population = [create_random_architecture_cnn(search_space_cnn) for _ in range(population_size)]
    population[0] = parents[0]

best_cnn = max(population, key=lambda arch: evaluate_architecture_cnn(arch, train_features_static, y_train, test_features_static, y_test))
print("Best CNN Architecture from ENAS:", best_cnn)

print("\n ENAS completed for Java Island (2×7 Grid)")