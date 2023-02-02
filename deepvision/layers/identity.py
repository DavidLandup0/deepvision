import tensorflow as tf
from tensorflow.keras.layers import Layer


class Identity(Layer):
    """Identity layer taken from tf-nightly.

    This layer is only temporarily hosted here, to avoid a tf-nightly dependency
    just for this one layer.

    This layer should be used as a placeholder when no operation is to be
    performed. The layer is argument insensitive, and returns its `inputs`
    argument as output.
    Args:
        name: Optional name for the layer instance.
    """

    def call(self, inputs):
        return tf.identity(inputs)
