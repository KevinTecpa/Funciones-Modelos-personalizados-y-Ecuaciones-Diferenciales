import tensorflow as tf
from tensorflow.keras.layers import Layer

class RGBToGray(Layer):
    """Capa Keras que convierte imágenes RGB en escala de grises."""
    def __init__(self, **kwargs):
        super(RGBToGray, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.image.rgb_to_grayscale(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)
