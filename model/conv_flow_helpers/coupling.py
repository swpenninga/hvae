from keras import ops, layers, random, regularizers
from keras.src import activations
import numpy as np


class Coupling(layers.Layer):
    def __init__(self, input_size, n_channels):
        super().__init__()
        self.n_channels = n_channels

        in_channels = n_channels // 2

        self.nn = Block_copy(
            input_size=input_size,
            in_width=in_channels,
            middle_width=in_channels // 2,
            out_width=in_channels,
            activation=layers.ReLU(),
            bn=False,
            residual=True,
            zero_last=False,
            model_depth=1,
            depthwise=False,
        )
        self.last = layers.Conv2D(
            n_channels,
            kernel_size=1,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )

    def call(self, x, ldj, reverse=False):
        x1 = x[:, :, :, : self.n_channels // 2]
        x2 = x[:, :, :, self.n_channels // 2 :]
        h = self.last(activations.relu(self.nn(x1)))
        h_s, t = h[:, :, :, ::2], h[:, :, :, 1::2]

        logs_range = 2.0
        log_s = logs_range * ops.tanh(h_s / logs_range)

        if not reverse:
            z2 = x2 * ops.exp(log_s) + t
            ldj = ldj + ops.sum(log_s, axis=(1, 2, 3))
        else:
            z2 = (x2 - t) * ops.exp(-log_s)
            ldj = ldj - ops.sum(log_s, axis=(1, 2, 3))

        z = ops.concatenate([x1, z2], axis=3)

        return z, ldj

    def reverse(self, z, ldj):
        return self.call(z, ldj, reverse=True)


# A copy of block is created to avoid circular imports (building_blocks -> decoder_block -> conv_flow -> coupling -> building_blocks)


class Block_copy(layers.Layer):
    def __init__(
        self,
        input_size,
        in_width,
        middle_width,
        out_width,
        activation,
        bn,
        residual,
        zero_last,
        model_depth,
        depthwise,
    ):
        super().__init__()
        # Parameters used for building
        self.input_size = input_size
        self.in_width = in_width
        self.middle_width = middle_width
        self.out_width = out_width
        self.zero_last = zero_last
        self.model_depth = model_depth
        self.depthwise = depthwise

        # Parameters used for calling
        self.activation = activation
        self.residual = residual

        # Batchnorm Layers, regularization for gradient smoothing (NVAE)
        self.bn1 = (
            layers.BatchNormalization(gamma_regularizer=regularizers.L1(0.01))
            if bn
            else layers.Identity()
        )
        self.bn2 = (
            layers.BatchNormalization(gamma_regularizer=regularizers.L1(0.01))
            if bn
            else layers.Identity()
        )
        self.bn3 = (
            layers.BatchNormalization(gamma_regularizer=regularizers.L1(0.01))
            if bn
            else layers.Identity()
        )
        self.bn4 = (
            layers.BatchNormalization(gamma_regularizer=regularizers.L1(0.01))
            if bn
            else layers.Identity()
        )

        # Convolutional layers
        self.c1 = layers.Conv2D(middle_width, kernel_size=1)
        if depthwise:
            # Depthwise 5x5 convs as used in NVAE
            self.c2 = (
                layers.DepthwiseConv2D(kernel_size=5, padding="same")
                if (self.input_size > 4)
                else layers.Conv2D(middle_width, kernel_size=1)
            )
            self.c3 = (
                layers.DepthwiseConv2D(kernel_size=5, padding="same")
                if (self.input_size > 4)
                else layers.Conv2D(middle_width, kernel_size=1)
            )
        else:
            # 3x3 convs as used in (Efficient)-VDVAE
            self.c2 = (
                layers.Conv2D(middle_width, kernel_size=3, padding="same")
                if (self.input_size > 2)
                else layers.Conv2D(middle_width, kernel_size=1)
            )
            self.c3 = (
                layers.Conv2D(middle_width, kernel_size=3, padding="same")
                if (self.input_size > 2)
                else layers.Conv2D(middle_width, kernel_size=1)
            )
        kern_init = "zeros" if zero_last else "glorot_uniform"
        self.c4 = layers.Conv2D(out_width, kernel_size=1, kernel_initializer=kern_init)

    def build(self):
        _ = self.call(
            random.normal([1, self.input_size, self.input_size, self.in_width])
        )

        # Initialize last layer with 1/sqrt(n)
        weights = self.c4.get_weights()
        weights[0] *= np.sqrt(1 / self.model_depth)
        self.c4.set_weights(weights)

    def call(self, x):
        xhat = self.c1(self.activation(self.bn1(x)))
        xhat = self.c2(self.activation(self.bn2(xhat)))
        xhat = self.c3(self.activation(self.bn3(xhat)))
        xhat = self.c4(self.activation(self.bn4(xhat)))
        out = ops.add(x, xhat) if self.residual else xhat
        return out


if __name__ == "__main__":
    x = random.normal((128, 8, 8, 4))
    ldj = ops.zeros(128)
    coupling = Coupling(8, 4)

    z, _ = coupling(x, ldj)

    x_recon, _ = coupling(z, ldj, reverse=True)

    print(ops.mean((x - x_recon) ** 2))
