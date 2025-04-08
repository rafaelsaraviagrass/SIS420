import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

# Cargar los datos desde el archivo JSON
with open('datos/stations.json', 'r') as file:
    data = json.load(file)

# Extraer características relevantes
features = []
target = []

for station in data['features']:
    props = station['properties']
    try:
        bikes_available = props['bikesAvailable']
        docks_available = props['docksAvailable']
        total_docks = props['totalDocks']
        latitude = props['latitude']
        longitude = props['longitude']

        if None not in (bikes_available, docks_available, total_docks, latitude, longitude):
            features.append([docks_available, total_docks, latitude, longitude])
            target.append(bikes_available)
    except KeyError:
        continue

# Convertir a arrays de NumPy
X = np.array(features)
y = np.array(target).reshape(-1, 1)

# Normalizar los datos con scalers separados
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Separar los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# En lugar de una búsqueda de hiperparámetros completa, usaremos una configuración mejorada
# pero fija (basada en buenas prácticas)

# Configuración del modelo mejorado
neurons = 64
layers = 2
dropout_rate = 0.2
learning_rate = 0.001
regularization = 0.001

# Crear modelo con normalización por lotes y regularización
modelo_mlp = Sequential()

# Primera capa con normalización por lotes
modelo_mlp.add(Dense(neurons, input_dim=X_train.shape[1], kernel_regularizer=l2(regularization)))
modelo_mlp.add(BatchNormalization())
modelo_mlp.add(Activation('relu'))
modelo_mlp.add(Dropout(dropout_rate))

# Segunda capa oculta
if layers >= 2:
    modelo_mlp.add(Dense(neurons, kernel_regularizer=l2(regularization)))
    modelo_mlp.add(BatchNormalization())
    modelo_mlp.add(Activation('relu'))
    modelo_mlp.add(Dropout(dropout_rate))

# Tercera capa oculta (opcional)
if layers >= 3:
    modelo_mlp.add(Dense(neurons // 2, kernel_regularizer=l2(regularization)))
    modelo_mlp.add(BatchNormalization())
    modelo_mlp.add(Activation('relu'))
    modelo_mlp.add(Dropout(dropout_rate))

# Capa de salida
modelo_mlp.add(Dense(1))

# Compilar el modelo
modelo_mlp.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

# Definir callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'mejor_modelo_mlp.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Entrenar el modelo
print("\nEntrenando modelo MLP mejorado...")
history_mlp = modelo_mlp.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Predicciones
y_pred_mlp = modelo_mlp.predict(X_test)

# Evaluación
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

print("\n--- Resultados Red Neuronal MLP Mejorada ---")
print(f"Error cuadrático medio (MSE): {mse_mlp:.4f}")
print(f"Coeficiente de determinación (R2): {r2_mlp:.4f}")

# Desnormalizar para mostrar resultados reales
y_test_real_mlp = scaler_y.inverse_transform(y_test)[:, 0]
y_pred_real_mlp = scaler_y.inverse_transform(y_pred_mlp)[:, 0]

print("\nEjemplos de predicción (MLP):")
for i in range(10):
    print(f"Real: {y_test_real_mlp[i]:.2f} - Predicción: {y_pred_real_mlp[i]:.2f}")

# Gráfico de la curva de aprendizaje
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_mlp.history['loss'], label='Entrenamiento')
plt.plot(history_mlp.history['val_loss'], label='Validación')
plt.title('Curva de Aprendizaje: Loss')
plt.xlabel('Época')
plt.ylabel('Error (MSE)')
plt.legend()
plt.grid(True)

# Gráfico comparativo de predicciones
plt.subplot(1, 2, 2)
plt.scatter(y_test_real_mlp, y_pred_real_mlp, alpha=0.5)
plt.plot([min(y_test_real_mlp), max(y_test_real_mlp)], 
         [min(y_test_real_mlp), max(y_test_real_mlp)], 
         'r--')
plt.title('Predicción vs Real')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico de comparación de valores reales vs predichos
plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test_real_mlp)), y_test_real_mlp, label='Real', color='blue', alpha=0.7)
plt.scatter(range(len(y_pred_real_mlp)), y_pred_real_mlp, label='Predicción MLP', color='red', alpha=0.7)
plt.title('Comparación: Valores Reales vs Predicción Red Neuronal MLP')
plt.xlabel('Muestra')
plt.ylabel('Bikes Available (Desnormalizado)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualización de residuos
residuos = y_test_real_mlp - y_pred_real_mlp
plt.figure(figsize=(12, 6))
plt.scatter(y_pred_real_mlp, residuos, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Gráfico de Residuos')
plt.xlabel('Predicciones')
plt.ylabel('Residuos (Real - Predicho)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Predicción con nueva estación
nueva_muestra = np.array([[10, 15, 39.9522, -75.1639]])  # docksAvailable, totalDocks, lat, lon
nueva_muestra_norm = scaler_X.transform(nueva_muestra)

prediccion_norm_mlp = modelo_mlp.predict(nueva_muestra_norm)
prediccion_final_mlp = scaler_y.inverse_transform(prediccion_norm_mlp)[:, 0]

print(f"\nPredicción para estación con 10 docks disponibles, 15 en total y coordenadas (lat, lon) (MLP): {prediccion_final_mlp[0]:.2f} bicicletas disponibles")

# Guardar el modelo entrenado
modelo_mlp.save('modelo_mlp_final_bicicletas.h5')
print("\nModelo guardado como 'modelo_mlp_final_bicicletas.h5'")