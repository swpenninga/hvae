from keras import layers, ops, Sequential
import keras

from conv_flow_helpers.convhouseholder import Conv1x1Householder
from conv_flow_helpers.maskedconv2d import MaskedConv2d, triu_conv_mask
from conv_flow_helpers.spectralnorm import MaskedSpectralNormalization
import conv_flow_helpers.convexp as F_convexp


class GeneralizedSylvester(layers.Layer):
    def __init__(self, in_channels, n_channels, spectral_normalize=True):
        super().__init__()
        self.n_channels = n_channels
        kernel_size = [3, 3, in_channels, in_channels]
        self.spectral_normalize = spectral_normalize

        if keras.dtype_policies.DTypePolicy().name == "mixed_bfloat16":
            self.dt = "bfloat16"
        else:
            self.dt = "float32"

        self.kernel = self.add_weight(
            shape=kernel_size,
            initializer="glorot_normal",
            trainable=True,
            dtype=self.dt,
        )
        # This initialization is not the same as the original implementation in pytorch.
        self.kernel.assign(self.kernel / ops.cast(ops.prod(kernel_size), self.dt))

        self.stride = (1, 1)
        self.padding = (1, 1)

        self.conv1x1 = Conv1x1Householder(in_channels, in_channels)
        self.nn = Sequential()
        mask = triu_conv_mask(in_channels, n_channels, (3, 3), diagonal_zeros=False)
        if self.spectral_normalize:
            self.nn.add(
                MaskedSpectralNormalization(
                    MaskedConv2d(
                        in_channels, n_channels, (3, 3), diagonal_zeros=False, bias=True
                    ),
                    mask=mask,
                    power_iterations=1,
                )
            )
        else:
            self.nn.add(
                MaskedConv2d(
                    in_channels, n_channels, (3, 3), diagonal_zeros=False, bias=True
                )
            )
        self.nn.add(layers.ReLU())
        mask = triu_conv_mask(n_channels, n_channels, (1, 1), diagonal_zeros=False)
        if self.spectral_normalize:
            self.nn.add(
                MaskedSpectralNormalization(
                    MaskedConv2d(
                        n_channels, n_channels, (1, 1), diagonal_zeros=False, bias=True
                    ),
                    mask=mask,
                    power_iterations=1,
                )
            )
        else:
            self.nn.add(
                MaskedConv2d(
                    n_channels, n_channels, (1, 1), diagonal_zeros=False, bias=True
                )
            )
        self.nn.add(layers.ReLU())

        self.linear_t1 = MaskedConv2d(
            n_channels, in_channels, size_kernel=(3, 3), diagonal_zeros=True, bias=True
        )
        self.linear_s1 = MaskedConv2d(
            n_channels, in_channels, size_kernel=(3, 3), diagonal_zeros=True, bias=True
        )
        self.linear_t2 = MaskedConv2d(
            n_channels, in_channels, size_kernel=(3, 3), diagonal_zeros=True, bias=True
        )
        self.linear_s2 = MaskedConv2d(
            n_channels, in_channels, size_kernel=(3, 3), diagonal_zeros=True, bias=True
        )

        weights = self.linear_t1.get_weights()
        weights[0] *= 0
        self.linear_t1.set_weights(weights)

        weights = self.linear_s1.get_weights()
        weights[0] *= 0
        self.linear_s1.set_weights(weights)

        if self.spectral_normalize:
            mask = triu_conv_mask(n_channels, in_channels, (3, 3), diagonal_zeros=True)
            self.linear_t1 = MaskedSpectralNormalization(
                self.linear_t1, mask=mask, power_iterations=1
            )
            self.linear_s1 = MaskedSpectralNormalization(
                self.linear_s1, mask=mask, power_iterations=1
            )
            self.linear_t2 = MaskedSpectralNormalization(
                self.linear_t2, mask=mask, power_iterations=1
            )
            self.linear_s2 = MaskedSpectralNormalization(
                self.linear_s2, mask=mask, power_iterations=1
            )

        self.trucation = 0
        self.terms = 9

    def der_tanh(self, a):
        return 1.0 - ops.tanh(a) ** 2

    def f_AR_nn(self, x):
        h = self.nn(x)

        h_t1 = self.linear_t1(h)
        h_s1 = self.linear_s1(h)
        h_t2 = self.linear_t2(h)
        h_s2 = self.linear_s2(h)

        t1 = h_t1
        s1 = ops.tanh(h_s1)

        t2 = h_t2
        s2 = 0.5 * ops.tanh(h_s2)

        return t1, s1, t2, s2

    def f_AR(self, x):
        t1, s1, t2, s2 = self.f_AR_nn(x)

        # alternative transform.
        s1_x = s1 * x + t1
        out = s2 * ops.tanh(s1_x) + t2
        diag = s2 * self.der_tanh(s1_x) * s1

        return out, diag

    def fw_call(self, x, ldj):
        z, _ = self.conv1x1(x, ops.zeros_like(ldj), None)
        Mz = F_convexp.conv_exp(
            z, self.kernel, terms=self.terms, dynamic_truncation=self.trucation
        )

        f_ar_Mz, diagonal = self.f_AR(Mz)

        Minv_f_ar_Mz = F_convexp.inv_conv_exp(
            f_ar_Mz, self.kernel, terms=self.terms, dynamic_truncation=self.trucation
        )

        Minv_f_ar_Mz, _ = self.conv1x1(
            Minv_f_ar_Mz, ops.zeros_like(ldj), None, reverse=True
        )

        out = x + Minv_f_ar_Mz

        delta_ldj = ops.sum(ops.log(1.0 + diagonal), axis=(1, 2, 3))

        ldj += delta_ldj

        return out, ldj

    def rv_call(self, x, ldj):
        v, _ = self.conv1x1(x, ops.zeros_like(ldj), None)
        v = F_convexp.conv_exp(
            v, self.kernel, terms=self.terms, dynamic_truncation=self.trucation
        )

        u = self.autoregressive_fixed_point_iteration(v)

        u = F_convexp.inv_conv_exp(
            u, self.kernel, terms=self.terms, dynamic_truncation=self.trucation
        )

        out, _ = self.conv1x1(u, ops.zeros_like(ldj), None, reverse=True)
        _, n_ldj = self.fw_call(out, -ldj)
        ldj = n_ldj
        return out, ldj

    def autoregressive_fixed_point_iteration(self, y):
        x = ops.copy(y)
        n_iterations = int(ops.prod(ops.shape(y)[1:]))

        converged_at = None

        atol = 1e-4
        oldx = x
        for iteration in range(1, n_iterations):
            y_, _ = self.f_AR(x)

            # Solve 1d case exact if conditioning checks out.
            newx = y - y_
            # y = x + f(x)
            # diff = y - (x * (1 + scale) + t)
            diff = newx - x
            if ops.max(ops.abs(diff)) < atol:
                print(ops.max(ops.abs(diff)))
                converged_at = iteration
                break
            oldx = x
            x = newx

        if converged_at is None:
            message = "Did not converge in {} iterations".format(n_iterations)
        else:
            message = "Converged at iteration {}".format(iteration)
        print(message)

        diff = ops.abs(x - oldx)
        factor = ops.sum((diff < atol)) / ops.prod(ops.shape(x))
        meandiff = ops.mean(diff)
        maxdiff = ops.max(diff)

        print(
            "{:.2f} dimensions converged, meandiff {} / maxdiff {}".format(
                factor, meandiff, maxdiff
            )
        )

        return x

    def call(self, inputs):
        x, ldj, reverse = inputs

        (out, ldj) = ops.cond(
            reverse, lambda: self.rv_call(x, ldj), lambda: self.fw_call(x, ldj)
        )
        return out, ldj


def get_GeneralizedSylvester(in_channels, hidden_channels, spectral_normalize=True):
    if spectral_normalize:
        return MaskedSpectralNormalization(
            GeneralizedSylvester(in_channels, hidden_channels, spectral_normalize),
            mask=None,
        )
    else:
        return GeneralizedSylvester(in_channels, hidden_channels, spectral_normalize)
