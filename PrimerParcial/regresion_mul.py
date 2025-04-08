import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Cargar datos desde el archivo JSON
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

        # Asegurarse de que no haya valores nulos
        if None not in (bikes_available, docks_available, total_docks, latitude, longitude):
            features.append([docks_available, total_docks, latitude, longitude])
            target.append(bikes_available)
    except KeyError:
        continue

# Convertir a arrays de NumPy
X = np.array(features)
y = np.array(target).reshape(-1, 1)

# Normalizar entradas y salidas por separado
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Crear y entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Realizar predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Resultados Regresión Multivariable ---")
print(f"Error cuadrático medio (MSE): {mse:.4f}")
print(f"Coeficiente de determinación (R2): {r2:.4f}")

# Desnormalizar para mostrar resultados reales
y_test_real = scaler_y.inverse_transform(y_test).flatten()
y_pred_real = scaler_y.inverse_transform(y_pred).flatten()

print("\nEjemplos de predicción:")
for i in range(10):
    print(f"Real: {y_test_real[i]:.2f} - Predicción: {y_pred_real[i]:.2f}")

# Gráfico de comparación
plt.scatter(range(len(y_test_real)), y_test_real, label='Real')
plt.scatter(range(len(y_pred_real)), y_pred_real, label='Predicción')
plt.title('Comparación: Valores Reales vs Predichos')
plt.xlabel('Muestra')
plt.ylabel('Bikes Available (Desnormalizado)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Predicción de una nueva estación
nueva_muestra = np.array([[10, 15, 39.9522, -75.1639]])  # docksAvailable, totalDocks, lat, lon
nueva_muestra_norm = scaler_X.transform(nueva_muestra)

prediccion_norm = modelo.predict(nueva_muestra_norm)
prediccion_final = scaler_y.inverse_transform(prediccion_norm)[0, 0]

print(f"\nPredicción para estación con 10 docks disponibles, 15 en total y coordenadas (lat, lon): {prediccion_final:.2f} bicicletas disponibles")
