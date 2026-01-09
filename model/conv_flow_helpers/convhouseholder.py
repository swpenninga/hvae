from keras import layers, ops


class Conv1x1Householder(layers.Layer):
    def __init__(self, n_channels, n_reflections):
        super().__init__()
        self.n_channels = n_channels
        self.n_reflections = n_reflections

        self.V = self.add_weight(
            shape=(n_reflections, n_channels),
            initializer="random_normal",
            trainable=True,
        )

    def construct_Q(self):
        I = ops.eye(self.n_channels)
        Q = I

        for i in range(self.n_reflections):
            v = ops.reshape(self.V[i], [self.n_channels, 1])

            vvT = ops.matmul(v, ops.transpose(v))
            vTv = ops.matmul(ops.transpose(v), v)
            Q = ops.matmul(Q, I - 2 * vvT / vTv)

        return Q

    def call(self, x, ldj, context, reverse=False):
        _, H, W, _ = ops.shape(x)

        Q = self.construct_Q()

        if not reverse:
            Q = ops.reshape(Q, [1, 1, self.n_channels, self.n_channels])

            z = ops.conv(x, Q, strides=1, padding="valid")

            return z, ldj
        else:
            Q_inv = ops.transpose(Q)
            Q_inv = ops.reshape(Q_inv, [1, 1, self.n_channels, self.n_channels])

            z = ops.conv(x, Q_inv, strides=1, padding="valid")

            return z, ldj

    def reverse(self, z, ldj, context):
        return self.call(z, ldj, context, reverse=True)
