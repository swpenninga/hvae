from keras import layers, ops


class ActNorm(layers.Layer):
    def __init__(self, n_channels):
        super().__init__()
        self.translation = self.add_weight(
            shape=(1, 1, 1, n_channels), initializer="zeros", trainable=True
        )
        self.log_scale = self.add_weight(
            shape=(1, 1, 1, n_channels), initializer="zeros", trainable=True
        )
        self.initialized = False

    def call(self, x, ldj, reverse=False):
        _, H, W, _ = ops.shape(x)
        d_ldj = -H * W * ops.sum(self.log_scale[0, 0, 0, :])

        if not reverse:
            z = (x - self.translation) * ops.exp(-self.log_scale)
            ldj += d_ldj
            return z, ldj
        else:
            z = x * ops.exp(self.log_scale) + self.translation

            ldj -= d_ldj
            return z, ldj

    def reverse(self, z, ldj):
        return self.call(z, ldj, reverse=True)
