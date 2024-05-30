# Custom L1 Distance layer module

# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer


# Custom L1 Distance Layer from Jupyter - needed to load custom model 
class L1Dist(Layer):
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
        
    # Similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)