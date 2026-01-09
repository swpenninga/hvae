from keras import ops, layers


def space_to_depth_ops(x):
    xs = ops.shape(x)  # batch, height, width, channels
    x = ops.reshape(
        x, (xs[0], xs[1] // 2, 2, xs[2] // 2, 2, xs[3])
    )  # batch, height/2, 2, width/2, 2, channels
    x = layers.Permute((1, 3, 5, 2, 4))(
        x
    )  # batch, height, width, channels, channels, 2, 2
    x = ops.reshape(x, (xs[0], xs[1] // 2, xs[2] // 2, xs[3] * 4))
    return x


def depth_to_space_ops(x):
    xs = ops.shape(x)  # batch, height, width, channels
    x = ops.reshape(
        x, (xs[0], xs[1], xs[2], xs[3] // 4, 2, 2)
    )  # batch, height, width, channels/4, 2, 2
    x = layers.Permute((1, 4, 2, 5, 3))(x)  # batch, height, 2, width, 2, channels
    x = ops.reshape(x, (xs[0], xs[1] * 2, xs[2] * 2, xs[3] // 4))
    return x


class Squeeze(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x, ldj, reverse=False):
        if not reverse:
            z = space_to_depth_ops(x)
        else:
            z = depth_to_space_ops(x)
        return z, ldj

    def reverse(self, z, ldj):
        return self.call(z, ldj, reverse=True)
