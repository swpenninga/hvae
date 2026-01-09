from keras import layers, ops, random, Sequential

from conv_flow_helpers.actnorm import ActNorm
from conv_flow_helpers.flip2d import Flip2d
from conv_flow_helpers.squeeze import Squeeze
from conv_flow_helpers.split import Split
from conv_flow_helpers.coupling import Coupling
from conv_flow_helpers.generalizedsylvester import (
    get_GeneralizedSylvester,
    GeneralizedSylvester,
)
from conv_flow_helpers.spectralnorm import MaskedSpectralNormalization


class ConvSylvester(layers.Layer):
    def __init__(
        self,
        input_size,
        flows_per_level,
        spectral_norm,
        sylv_channels,
        split_first,
        stage_limit,
    ):
        super().__init__()
        self.input_size = input_size  # (H, W, C)
        self.flows_per_level = flows_per_level
        self.spectral_norm = spectral_norm
        self.sylv_channels = sylv_channels
        if stage_limit == 0:
            stage_limit = 100

        self.n_levels = min(int(ops.log2(input_size[0]) + 1), stage_limit)
        self.split_first = split_first

        self.flow = Sequential()
        current_size = input_size
        if self.split_first:
            self.flow.add(Split())
            current_size = (current_size[0], current_size[1], current_size[2] // 2)

        for level in range(self.n_levels):
            # Add all the layers at this level
            for _ in range(flows_per_level):
                # bruh this actnorm homie is fuckign it up and killing
                # my brother on batch 1

                self.flow.add(ActNorm(current_size[-1]))

                self.flow.add(
                    get_GeneralizedSylvester(
                        current_size[-1],
                        self.sylv_channels,
                        spectral_normalize=spectral_norm,
                    )
                )
                if current_size[0] > 1:
                    self.flow.add(Flip2d())

            # if we are not yet at the last level
            if level < self.n_levels - 1:
                if level >= 1:
                    self.flow.add(
                        Coupling(
                            input_size=current_size[0], n_channels=current_size[-1]
                        )
                    )
                    self.flow.add(Split())

                    current_size = (
                        current_size[0],
                        current_size[1],
                        current_size[2] // 2,
                    )

                self.flow.add(Squeeze())
                current_size = (
                    current_size[0] // 2,
                    current_size[1] // 2,
                    current_size[2] * 4,
                )

    def build(self):
        _ = self.call(z=random.normal((1, *self.input_size)))

    def reconstruct_z(self, z, z_splits):
        for layer in reversed(self.flow.layers):
            if isinstance(layer, Squeeze):
                z, _ = layer.reverse(z, ops.convert_to_tensor(0))
            elif isinstance(layer, Split):
                z = layer.reverse(z, z_splits.pop(-1))
            elif isinstance(layer, Flip2d):
                z, _ = layer(z, ops.convert_to_tensor(0))
        return z

    def call(self, z, h=None):
        ldj = ops.zeros(ops.shape(z)[0])

        z_splits = []
        for layer in self.flow.layers:
            if isinstance(layer, Split):
                z, split = layer(z)
                z_splits.append(split)
            elif isinstance(layer, MaskedSpectralNormalization) or isinstance(
                layer, GeneralizedSylvester
            ):
                z, ldj = layer((z, ldj, ops.convert_to_tensor(False)))
            else:
                z, ldj = layer(z, ldj)

        z = self.reconstruct_z(z, z_splits)
        return z, ldj
