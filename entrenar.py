import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans


# DESCARGAR DATASET MNIST

print("Cargando dataset MNIST...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("\nDataset cargado correctamente")

print("Imagenes de entrenamiento:", x_train.shape)
print("Etiquetas de entrenamiento:", y_train.shape)

print("Imagenes de prueba:", x_test.shape)
print("Etiquetas de prueba:", y_test.shape)


# NORMALIZACION DE IMAGENES

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# CONVERTIR IMAGENES A FORMATO CNN

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("\nNueva forma de X:", x_train.shape)
print("Forma de una imagen:", x_train[0].shape)

# SEPARAR ENTRENAMIENTO Y VALIDACION

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.2, random_state=42, stratify=y_train)

print("\nDATOS FINALES")

print("Entrenamiento:", x_train.shape)
print("Validacion:", x_val.shape)
print("Prueba:", x_test.shape)


# VISUALIZAR DIGITOS

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.title(f"Etiqueta: {y_train[i]}")
    plt.axis('off')

plt.suptitle("Dataset MNIST - TecnoForms")
plt.tight_layout()

plt.savefig('resultados/digitos_mnist.png', dpi=300)

plt.close()

print("\nPREPROCESAMIENTO COMPLETADO")

# CONSTRUCCION DE LA RED NEURONAL CNN

print("\nConstruyendo modelo CNN...")

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# COMPILACION DEL MODELO

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Mostrar arquitectura

model.summary()

# ENTRENAMIENTO DEL MODELO

print("\nIniciando entrenamiento de la CNN...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_val, y_val)
)

print("\nENTRENAMIENTO COMPLETADO")

# EVALUACION DEL MODELO

print("\nEvaluando modelo con datos de prueba...")

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nRESULTADOS DEL MODELO")
print("Exactitud (Accuracy):", test_accuracy)
print("Perdida (Loss):", test_loss)

# GUARDAR MODELO ENTRENADO

model.save("modelos/modelo_mnist.keras")
print("\nModelo guardado correctamente")

# GRAFICAS DEL ENTRENAMIENTO

plt.figure(figsize=(12, 5))

# GRAFICA DE EXACTITUD

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validacion')
plt.title('Exactitud del modelo')
plt.xlabel('Epocas')
plt.ylabel('Accuracy')
plt.legend()

# GRAFICA DE PERDIDA

plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validacion')

plt.title('Perdida del modelo')
plt.xlabel('Epocas')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('resultados/graficas_entrenamiento.png', dpi=300)
plt.close()

print("\nEVALUACION Y GRAFICAS COMPLETADAS")



# COMPARACION ENTRE CNN Y KNN

print("\nINICIANDO COMPARACION ENTRE CNN Y KNN...")


# PREPARAR DATOS PARA EL MODELO KNN

print("\nPreparando datos para KNN...")

x_train_knn = x_train.reshape(-1, 784)
x_val_knn = x_val.reshape(-1, 784)


# UTILIZAR UNA MUESTRA PARA REDUCIR EL TIEMPO

x_train_knn, _, y_train_knn, _ = train_test_split(x_train_knn,y_train,train_size=0.2,random_state=42,stratify=y_train)
x_val_knn, _, y_val_knn, _ = train_test_split(x_val_knn,y_val,train_size=0.2,random_state=42,stratify=y_val)


# ENTRENAMIENTO DEL MODELO KNN

print("\nEntrenando modelo KNN...")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_knn, y_train_knn)


# PREDICCIONES KNN

print("\nRealizando predicciones...")
pred_knn = knn.predict(x_val_knn)


# EVALUACION DEL MODELO KNN

accuracy_knn = accuracy_score(y_val_knn, pred_knn)
print("\nRESULTADOS KNN")
print("Exactitud KNN:", accuracy_knn)


# COMPARACION DE RESULTADOS

print("\nCOMPARACION DE ALGORITMOS")
print("CNN:", test_accuracy)
print("KNN:", accuracy_knn)

# GUARDAR RESULTADOS

with open('resultados/comparacion_algoritmos.txt', 'w') as archivo:
    archivo.write("COMPARACION DE ALGORITMOS - TECNOFORMS\n\n")
    archivo.write(f"CNN: {test_accuracy:.4f}\n")
    archivo.write(f"KNN: {accuracy_knn:.4f}\n")

print("\nCOMPARACION COMPLETADA")



# IMPLEMENTACION DEL ALGORITMO K-MEANS

print("\nINICIANDO MODELO K-MEANS...")

# PREPARAR IMAGENES PARA K-MEANS

print("\nPreparando datos para K-Means...")
x_train_kmeans = x_train.reshape(-1, 784)


# UTILIZAR UNA MUESTRA PARA REDUCIR EL TIEMPO

x_kmeans, _, y_kmeans, _ = train_test_split(
    x_train_kmeans,
    y_train,
    train_size=0.1,
    random_state=42,
    stratify=y_train
)


# CONSTRUCCION DEL MODELO K-MEANS

print("\nConstruyendo modelo K-Means...")

kmeans = KMeans(n_clusters=10,random_state=42,n_init=5,max_iter=100)


# ENTRENAMIENTO DEL MODELO

print("\nEntrenando modelo K-Means...")
clusters = kmeans.fit_predict(x_kmeans)


# EVALUACION DE LOS GRUPOS

from sklearn.metrics import adjusted_rand_score
resultado_kmeans = adjusted_rand_score(y_kmeans, clusters)

print("\nRESULTADOS K-MEANS")
print("Cantidad de grupos:", kmeans.n_clusters)
print("Cantidad de imagenes:", len(x_kmeans))
print("Indice Adjusted Rand:", resultado_kmeans)


# GUARDAR RESULTADOS

with open('resultados/resultados_kmeans.txt', 'w') as archivo:
    archivo.write("RESULTADOS K-MEANS - TECNOFORMS\n\n")
    archivo.write(f"Cantidad de grupos: {kmeans.n_clusters}\n")
    archivo.write(f"Cantidad de imagenes: {len(x_kmeans)}\n")
    archivo.write(f"Adjusted Rand Index: {resultado_kmeans:.4f}\n")

print("\nMODELO K-MEANS COMPLETADO")
