from keras import layers, ops


class Split(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        x1, x2 = ops.split(x, 2, axis=-1)
        return x1, x2

    def reverse(self, x1, x2):
        return ops.concatenate([x1, x2], axis=-1)
