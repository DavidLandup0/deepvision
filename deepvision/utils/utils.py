from tensorflow import keras
from tensorflow.keras import layers


def same_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def parse_model_inputs(backend, input_shape, input_tensor):
    if backend == "tensorflow":
        if input_tensor is None:
            return layers.Input(shape=input_shape)
        else:
            if not keras.backend.is_keras_tensor(input_tensor):
                return layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                return input_tensor
    elif backend == "pytorch":
        return input_tensor
    else:
        raise ValueError(f"Backend not supported: {backend}")
