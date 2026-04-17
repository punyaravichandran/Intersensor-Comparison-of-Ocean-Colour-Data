# -*- coding: utf-8 -*-
"""
PACE vs In-situ chlorophyll validation (LOOCV)
Author: Punya. P
"""

# Imports the necessary python packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut

from scipy.stats import pearsonr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load Data
data_path = "data/PACE_MODIS_CHL_5_31March2024_Sir_send_New2.csv"
data = pd.read_csv(data_path)

# Select required columns
data1 = data[['PaceChl', 'latitude.1', 'longitude.1', 'Insitu_PACE']].dropna()

# Rename for coloumns
data1 = data1.rename(columns={
    'latitude.1': 'latitude',
    'longitude.1': 'longitude'
})

# Features & Target
X = data1[['PaceChl']].values
y = data1['Insitu_PACE'].values

# Scaling 
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()


# Reshape for CNN
X = X.reshape((X.shape[0], X.shape[1], 1))

# Model
def create_cnn_model(input_shape):
    model = Sequential([
        Reshape((1, input_shape[1]), input_shape=input_shape),
        Conv1D(32, kernel_size=1, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# LOOCV
loo = LeaveOneOut()

mse_scores, r2_scores, mae_scores, corr_scores = [], [], [], []
all_y_true, all_y_pred = [], []

# Training Loop
for train_idx, val_idx in loo.split(X):

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = create_cnn_model(X.shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    y_pred = model.predict(X_val).flatten()

    # Store predictions
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)

    mse_scores.append(mean_squared_error(y_val, y_pred))
    mae_scores.append(mean_absolute_error(y_val, y_pred))

# Final Metrics
final_r2 = r2_score(all_y_true, all_y_pred)
final_mse = np.mean(mse_scores)
final_mae = np.mean(mae_scores)

# Pearson correlation
corr, _ = pearsonr(all_y_true, all_y_pred)

# Inverse scaling for plotting
y_true_inv = scaler_y.inverse_transform(np.array(all_y_true).reshape(-1, 1)).flatten()
y_pred_inv = scaler_y.inverse_transform(np.array(all_y_pred).reshape(-1, 1)).flatten()

# Plot 
plt.figure(figsize=(5, 5))

sns.scatterplot(x=y_true_inv, y=y_pred_inv, s=60, alpha=0.7)

# 1:1 line
min_val = min(y_true_inv.min(), y_pred_inv.min())
max_val = max(y_true_inv.max(), y_pred_inv.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.xlabel('In-situ Chlorophyll (mg/m³)')
plt.ylabel('PACE Chlorophyll (mg/m³)')

# Metrics
plt.text(0.05, 0.95, f'R² = {final_r2:.3f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.90, f'MSE = {final_mse:.3f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f'MAE = {final_mae:.3f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.80, f'R = {corr:.3f}', transform=plt.gca().transAxes)

plt.grid(True)
plt.tight_layout()
plt.show()
