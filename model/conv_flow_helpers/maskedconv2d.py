from keras import ops, layers
import keras
import numpy as np


def triu_conv_mask(n_in, n_out, kernel_size, diagonal_zeros):
    assert n_out % n_in == 0 or n_in % n_out == 0, "%d - %d" % (n_in, n_out)

    # Build autoregressive mask
    l = (kernel_size[0] - 1) // 2
    m = (kernel_size[1] - 1) // 2
    mask = np.ones((n_out, n_in, kernel_size[0], kernel_size[1]), dtype=np.float32)
    mask[:, :, :l, :] = 0
    mask[:, :, l, :m] = 0

    if n_out >= n_in:
        k = n_out // n_in
        for i in range(n_in):
            mask[i * k : (i + 1) * k, i + 1 :, l, m] = 0
            if diagonal_zeros:
                mask[i * k : (i + 1) * k, i : i + 1, l, m] = 0
    else:
        k = n_in // n_out
        for i in range(n_out):
            mask[i : i + 1, (i + 1) * k :, l, m] = 0
            if diagonal_zeros:
                mask[i : i + 1, i * k : (i + 1) * k :, l, m] = 0
    return mask


class MaskedConv2d(layers.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        size_kernel=(3, 3),
        diagonal_zeros=False,
        bias=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.size_kernel = size_kernel
        self.stride = (1, 1)
        self.diagonal_zeros = diagonal_zeros

        if keras.dtype_policies.DTypePolicy().name == "mixed_bfloat16":
            self.dt = "bfloat16"
        else:
            self.dt = "float32"

        self.kernel = self.add_weight(
            shape=(*self.size_kernel, self.in_features, self.out_features),
            trainable=True,
            dtype=self.dt,
        )
        if bias:
            self.bias = self.add_weight(
                shape=(1, 1, 1, out_features),
                initializer="zeros",
                trainable=True,
                dtype=self.dt,
            )
        else:
            self.bias = None

        mask = triu_conv_mask(
            self.in_features, self.out_features, size_kernel, diagonal_zeros
        )
        mask = np.transpose(mask, (2, 3, 1, 0))
        self.mask = ops.convert_to_tensor(mask, dtype=self.dt)

    def call(self, x):
        out = ops.conv(x, self.mask * self.kernel, strides=self.stride, padding="same")
        out = ops.add(out, self.bias)
        return out
