import numpy as np
import tensorflow as tf
from rgb_a_grises import RGBToGray

# Cargar MNIST
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

# Normalizar y simular imágenes RGB
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)          # (N,28,28,1)
x_train_rgb = np.repeat(x_train, 3, axis=-1)   # (N,28,28,3)

print("Shape original RGB:", x_train_rgb.shape)

# Aplicar la capa de conversión a gris
gray_layer = RGBToGray()
x_gray = gray_layer(x_train_rgb)
print("Shape después de la capa:", x_gray.shape)

# Modelo simple usando la capa
model = tf.keras.Sequential([
    RGBToGray(input_shape=(28,28,3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

