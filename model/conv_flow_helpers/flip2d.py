from keras import ops, layers


class Flip2d(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x, ldj, reverse=False):
        x = ops.flip(x, axis=1)
        x = ops.flip(x, axis=2)
        return x, ldj

    def reverse(self, z, ldj):
        return self.call(z, ldj, reverse=True)
