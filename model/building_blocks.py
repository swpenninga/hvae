import os.path
import sys
import random as pythonrandom
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from attention.attention import AttentionUnpool, attention_block
from conv_sylvester_flow import ConvSylvester
from keras import Sequential, Variable, layers, ops, random
from sylvester_flow import OrthogonalSylvesterStack
from utils.distributions import SoftPlus, flow_kl, gaussian_analytical_kl


class DecoderStage(layers.Layer):
    # Everything that happens at every scale (resolution) of the decoder, from input to output and upsampling
    def __init__(self, params, stage_num):
        super().__init__()
        self.zdim = params.zdim[stage_num]
        self.input_size = params.dec_input_size[stage_num]
        self.in_width = params.dec_in_width[stage_num]
        self.middle_width = params.dec_middle_width[stage_num]
        self.pool_width = params.dec_pool_width[stage_num]
        self.num_blocks = params.dec_num_blocks[stage_num]

        # attention
        self.use_depthwise_attention = params.use_depthwise_attention

        # sylvester flow params
        self.num_flows = params.num_flows[stage_num]
        self.flow_in_ch = params.flow_in_ch[stage_num] if self.num_flows > 0 else 0
        self.num_ortho_vecs = (
            params.num_ortho_vecs[stage_num] if self.num_flows > 0 else 0
        )

        # conv_sylvester flow param
        self.convsylv_channels = params.convsylv_channels[stage_num]
        self.convsylv_flows_per_stage = params.convsylv_flows_per_stage[stage_num]
        self.convsylv_splitfirst = params.convsylv_splitfirst[stage_num]
        self.convsylv_stage_limit = params.convsylv_stage_limit[stage_num]

        self.z_out = params.z_out

        # For the decoder, a block is a DecBlock
        self.blocks = Sequential()
        for i in range(self.num_blocks):
            first_block = False
            last_block = False
            if i == 0 and self.input_size == min(params.dec_input_size):
                first_block = True
            if i == self.num_blocks - 1 and self.input_size == max(
                params.dec_input_size
            ):
                last_block = True

            self.blocks.add(
                DecBlock(
                    input_size=self.input_size,
                    in_width=self.in_width,
                    middle_width=self.middle_width,
                    zdim=self.zdim,
                    z_out=params.z_out,
                    kernelsize=params.kernelsizes[str(self.input_size)],
                    block_activation=params.block_activation,
                    block_bn=params.block_bn,
                    model_depth=params.model_depth,
                    depthwise=params.depthwise,
                    num_flows=self.num_flows,
                    flow_in_ch=self.flow_in_ch,
                    num_ortho_vecs=self.num_ortho_vecs,
                    flow_type=params.flow_type,
                    spectral_norm=params.spectral_norm,
                    convsylv_channels=self.convsylv_channels,
                    convsylv_flows_per_stage=self.convsylv_flows_per_stage,
                    convsylv_splitfirst=self.convsylv_splitfirst,
                    convsylv_stage_limit=self.convsylv_stage_limit,
                    gradient_smoothing=params.gradient_smoothing,
                    cr_vae=params.cr_vae,
                    cr_duplicate_noise=params.cr_duplicate_noise,
                    use_depthwise_attention=self.use_depthwise_attention,
                    query_width=params.query_width,
                    num_queries=params.num_queries,
                    first_block=first_block,
                    last_block=last_block,
                )
            )
        self.pool = PoolLayer(
            input_size=self.input_size,
            in_width=self.in_width,
            out_width=self.pool_width,
            pool_activation=params.pool_activation,
            unpool=True,
            data_size=params.enc_input_size[0],
        )
        if self.use_depthwise_attention:
            self.attn_pool = AttentionUnpool(
                input_size=self.input_size,
                data_size=params.enc_input_size[0],
            )

    def build(self):
        for dec_block in self.blocks.layers:
            dec_block.build()
        self.pool.build()

    def call(self, x, act, training):
        # Through latent aggregation, all latents are summed,
        # so we create the output shape and sum to it for all blocks in the stage.
        B = ops.shape(x[0])[0] if self.use_depthwise_attention else ops.shape(x)[0]
        z_blocks = ops.tile(ops.zeros([1, *self.z_out]), (B, 1, 1, 1))
        kl_blocks = []
        for dec_block in self.blocks.layers:
            x, z, kl = dec_block.call(x, act, training=training)
            z_blocks += z
            kl_blocks.append(kl)

        if not self.use_depthwise_attention:
            x = self.pool(x)
        else:
            dec, vp, kp = x
            x = (self.pool(dec), self.attn_pool(vp), self.attn_pool(kp))
        return x, z_blocks, kl_blocks

    def call_uncond(self, x, t=1):
        B = ops.shape(x[0])[0] if self.use_depthwise_attention else ops.shape(x)[0]
        z_blocks = ops.tile(ops.zeros([1, *self.z_out]), (B, 1, 1, 1))
        for dec_block in self.blocks.layers:
            x, z = dec_block.call_uncond(x, t)
            z_blocks += z

        if not self.use_depthwise_attention:
            x = self.pool(x)
        else:
            dec, vp, kp = x
            x = (self.pool(dec), self.attn_pool(vp), self.attn_pool(kp))
        return x, z_blocks


class EncoderStage(layers.Layer):
    # Everything that happens at every scale (resolution) of the encoder, from input to output and downsampling
    def __init__(self, params, stage_num):
        super().__init__()
        self.stage_num = stage_num

        self.input_size = params.enc_input_size[stage_num]
        self.in_width = params.enc_in_width[stage_num]
        self.middle_width = params.enc_middle_width[stage_num]
        self.pool_width = params.enc_pool_width[stage_num]
        self.num_blocks = params.enc_num_blocks[stage_num]

        self.use_depthwise_attention = params.use_depthwise_attention
        if self.use_depthwise_attention:
            self.key_width = params.query_width
            self.lw_v = layers.LayerNormalization(axis=[1, 2, 3])

        out_width = self.in_width
        self.blocks = Sequential()
        for i in range(self.num_blocks):
            out_width = self.in_width
            # If this is the final block of the stage, we also output the keys k^q_l
            if i == self.num_blocks - 1 and self.use_depthwise_attention:
                out_width += self.key_width

            in_width = self.in_width

            # For the encoder, a block is just a ResNet Block
            self.blocks.add(
                Block(
                    input_size=self.input_size,
                    in_width=in_width,
                    middle_width=self.middle_width,
                    out_width=out_width,
                    kernelsize=params.kernelsizes[str(self.input_size)],
                    activation=params.block_activation,
                    bn=params.block_bn,
                    residual=True,
                    zero_last=False,
                    model_depth=params.model_depth,
                    depthwise=False,
                    dwa_enc=self.use_depthwise_attention,
                )
            )

        self.pool = PoolLayer(
            input_size=self.input_size,
            in_width=out_width,
            out_width=self.pool_width,
            pool_activation=params.pool_activation,
            unpool=False,
        )

    def build(self):
        for enc_block in self.blocks.layers:
            enc_block.build()
        self.pool.build()

        _, _ = self.call(
            random.normal([1, self.input_size, self.input_size, self.in_width])
        )

    def call(self, x):
        for enc_block in self.blocks.layers:
            # Only the final block will have x with more channels
            x = enc_block(x)

        # return pooled x for the next stage and the activations for the decoder
        # For depthwise attention, x is split into values and keys
        if self.use_depthwise_attention:
            if self.num_blocks > 0:
                vq = x[:, :, :, : self.in_width]  # (B, H, W, in_width)
                vq += ops.gelu(self.lw_v(vq))
                kq = x[:, :, :, self.in_width :]  # (B, H, W, key_width)
            else:
                vq = None
                kq = None
            return self.pool(x), [vq, kq]
        else:
            return self.pool(x), x


class DecBlock(layers.Layer):
    # Definition of a single block in a decoder stage
    def __init__(
        self,
        input_size,
        in_width,
        middle_width,
        zdim,
        z_out,
        kernelsize,
        block_activation,
        block_bn,
        model_depth,
        depthwise,
        num_flows,
        flow_in_ch,
        num_ortho_vecs,
        flow_type,
        spectral_norm,
        convsylv_channels,
        convsylv_flows_per_stage,
        convsylv_splitfirst,
        convsylv_stage_limit,
        gradient_smoothing,
        cr_vae,
        cr_duplicate_noise,
        use_depthwise_attention,
        query_width,
        num_queries,
        first_block,
        last_block,
    ):
        super().__init__()
        self.first_block = first_block
        self.last_block = last_block
        # The blocks cannot have an identical seedgenerator, they also cannot share a single one,
        # this is necessary because of stateless mirrorstrategy
        seed = pythonrandom.randint(0, 1_000_000)
        self.seed_gen = random.SeedGenerator(1337 + seed)

        # Parameters used for building
        self.input_size = input_size
        self.in_width = in_width
        self.middle_width = middle_width
        self.zdim = zdim
        self.model_depth = model_depth

        self.cr_vae = cr_vae
        self.cr_duplicate_noise = cr_duplicate_noise
        self.use_depthwise_attention = use_depthwise_attention

        self.num_flows = num_flows
        self.flow_in_ch = flow_in_ch
        self.num_ortho_vecs = num_ortho_vecs
        self.flow_type = flow_type

        if self.flow_type != "none" and self.num_flows > 0:
            self.use_flow = True
        else:
            self.use_flow = False

        # output of prior block is mu_p, std_p, and residual
        p_out_width = 2 * self.zdim + self.in_width
        if self.use_depthwise_attention:
            self.query_width = query_width
            self.num_queries = num_queries
            # Also queries if depthwise attention is used
            p_out_width += self.query_width

            # If we have more than one query, we combine them after attention
            self.combine_queries = self.num_queries > 1
            if self.combine_queries:
                self.queries_comb_q = layers.Conv2D(self.in_width, kernel_size=1)
                if not self.first_block:
                    self.queries_comb_p = layers.Conv2D(self.in_width, kernel_size=1)

            queries_out = self.query_width * num_queries
            self.gamma_q = Variable(0, trainable=True, dtype="float32")

            if not self.first_block:
                # Also output prior queries
                queries_out *= 2
                self.gamma_p = Variable(0, trainable=True, dtype="float32")

            self.queries = Block(
                input_size=self.input_size,
                in_width=self.in_width,
                middle_width=self.middle_width,
                out_width=queries_out,
                kernelsize=kernelsize,
                activation=block_activation,
                bn=block_bn,
                residual=False,
                zero_last=False,
                model_depth=model_depth,
                depthwise=depthwise,
            )

        q_out_width = self.zdim * 2
        # For the sylvester flow, we also output h (flow parameters)
        if self.flow_type == "sylvester":
            q_out_width += self.flow_in_ch

        # Block that takes activations from encoder
        self.q = Block(
            input_size=self.input_size,
            in_width=2 * self.in_width,
            middle_width=self.middle_width,
            out_width=q_out_width,
            kernelsize=kernelsize,
            activation=block_activation,
            bn=block_bn,
            residual=False,
            zero_last=False,
            model_depth=model_depth,
            depthwise=depthwise,
        )

        if self.use_flow:
            if self.flow_type == "sylvester":
                # Only implemented orthogonal sylvester flows
                self.flows = OrthogonalSylvesterStack(
                    num_flows=self.num_flows,
                    num_ortho_vecs=self.num_ortho_vecs,
                    z_ch=self.zdim,
                    build_shape=[1, self.input_size, self.input_size, self.flow_in_ch],
                    model_depth=model_depth,
                )
            elif self.flow_type == "conv_sylvester":
                self.flows = ConvSylvester(
                    input_size=(self.input_size, self.input_size, self.zdim),
                    flows_per_level=convsylv_flows_per_stage,
                    spectral_norm=spectral_norm,
                    sylv_channels=convsylv_channels,
                    split_first=convsylv_splitfirst,
                    stage_limit=convsylv_stage_limit,
                )
            self.kl = flow_kl()
        else:
            self.kl = gaussian_analytical_kl()

        # Prior block
        self.p = Block(
            input_size=self.input_size,
            in_width=self.in_width,
            middle_width=self.middle_width,
            out_width=p_out_width,
            kernelsize=kernelsize,
            activation=block_activation,
            bn=block_bn,
            residual=False,
            zero_last=True,
            model_depth=model_depth,
            depthwise=depthwise,
        )

        # Residual block for after adding Z
        if not self.last_block:
            self.res = Block(
                input_size=self.input_size,
                in_width=self.in_width,
                middle_width=self.middle_width,
                out_width=self.in_width,
                kernelsize=kernelsize,
                activation=block_activation,
                bn=block_bn,
                residual=True,
                zero_last=False,
                model_depth=model_depth,
                depthwise=depthwise,
            )
            self.z_proj = layers.Conv2D(in_width, kernel_size=1)

        self.sp = SoftPlus(gradient_smoothing)
        self.z_out_f = layers.Conv2D(z_out[-1], kernel_size=1)
        if z_out[0] > input_size:
            self.z_out_up = layers.UpSampling2D(
                size=z_out[0] // input_size, interpolation="nearest"
            )
        else:
            self.z_out_up = layers.Identity()

    def build(self):
        self.q.build()
        self.p.build()
        if self.use_flow:
            self.flows.build()

        if not self.last_block:
            # Last block does not have these layers, since the chain stops and latents are collected.
            self.res.build()
            _ = self.z_proj(
                random.normal([1, self.input_size, self.input_size, self.zdim])
            )
            weights = self.z_proj.get_weights()
            weights[0] *= np.sqrt(1 / self.model_depth)
            self.z_proj.set_weights(weights)

    def sample(self, x, act, training):
        # Calculate all attention outputs
        if self.use_depthwise_attention:
            x, vp, kp = x
            vq, kq = act

            queries = self.queries(x)

            if not self.first_block:
                qq, qp = ops.split(queries, 2, axis=3)
                if self.combine_queries:
                    for i, query in enumerate(ops.split(qp, self.num_queries, axis=-1)):
                        attn = attention_block(query, kp, vp)
                        if i == 0:
                            attn_p = attn
                        else:
                            attn_p = ops.concatenate([attn_p, attn], axis=-1)
                    attn_p = self.queries_comb_p(attn_p)
                else:
                    attn_p = attention_block(qp, kp, vp)
                attn_p += ops.gelu(attn_p)
            else:
                qq, qp = queries, None

            if ops.shape(vq)[-1] > 1:
                if self.combine_queries:
                    for i, query in enumerate(ops.split(qq, self.num_queries, axis=-1)):
                        attn = attention_block(
                            query, kq[:, :, :, :, :-1], vq[:, :, :, :, :-1]
                        )
                        if i == 0:
                            attn_q = attn
                        else:
                            attn_q = ops.concatenate([attn_q, attn], axis=-1)
                    attn_q = self.queries_comb_q(attn_q)
                else:
                    attn_q = attention_block(
                        qq, kq[:, :, :, :, :-1], vq[:, :, :, :, :-1]
                    )
                attn_q += ops.gelu(attn_q)
            else:
                attn_q = ops.zeros_like(x)
            q_out = self.q(
                ops.concatenate(
                    [x, vq[:, :, :, :, -1] + self.gamma_q * attn_q], axis=-1
                )
            )
        else:
            q_out = self.q(ops.concatenate([x, act], axis=-1))

        if self.flow_type == "sylvester":
            qm, q_std, h = (
                q_out[:, :, :, : self.zdim],
                q_out[:, :, :, self.zdim : self.zdim * 2],
                q_out[:, :, :, self.zdim * 2 :],
            )
        else:
            qm, q_std = ops.split(q_out, 2, axis=3)
            h = None

        # calculate prior (mu_p, variance_p, and residual)
        if self.use_depthwise_attention:
            if self.first_block:
                prior = self.p(x)
            else:
                prior = self.p(x + self.gamma_p * attn_p)
            pm, p_std, vpl, kpl = (
                prior[:, :, :, : self.zdim],
                prior[:, :, :, self.zdim : self.zdim * 2],
                prior[:, :, :, self.zdim * 2 : self.zdim * 2 + self.in_width],
                prior[:, :, :, self.zdim * 2 + self.in_width :],
            )

            x = ops.add(x, vpl)  # (B, H, W, in_width)

            if self.first_block:
                vp = ops.expand_dims(vpl, axis=-1)
                kp = ops.expand_dims(kpl, axis=-1)
            else:
                vp = ops.concatenate([vp, ops.expand_dims(vpl, axis=-1)], axis=-1)
                kp = ops.concatenate([kp, ops.expand_dims(kpl, axis=-1)], axis=-1)
            x = (x, vp, kp)

        else:
            prior = self.p(x)
            pm, p_std, vpl = (
                prior[:, :, :, : self.zdim],
                prior[:, :, :, self.zdim : self.zdim * 2],
                prior[:, :, :, self.zdim * 2 :],
            )
            x = ops.add(x, vpl)

        # Softplus for smoothing
        q_std = self.sp(q_std)
        p_std = self.sp(p_std)

        noise = random.normal(ops.shape(q_std), seed=self.seed_gen)
        if self.cr_vae and self.cr_duplicate_noise and training:
            # Copy noise to be the same for both normal and transformed batch for CR-VAE
            noise = ops.repeat(noise[ops.shape(q_std)[0] // 2 :], 2, axis=0)

        z0 = ops.add(qm, ops.multiply(q_std, noise))

        if self.use_flow:
            zk, log_det_j = self.flows.call(z0, h)
            kl = self.kl.call(qm, q_std, pm, p_std, z0, zk)
            if self.cr_vae and training:
                R_loss = self.kl.cr_loss(zk)  # This loss is 0!
                kl = (kl, R_loss)  # we package the R_loss together with kl
            return x, zk, (kl, log_det_j)
        else:
            kl = self.kl.call(qm, q_std, pm, p_std)
            if self.cr_vae and training:
                R_loss = self.kl.cr_loss(qm, q_std)
                kl = (kl, R_loss)  # we package the R_loss together with kl
            return x, z0, kl

    def sample_uncond(self, x, t=1):
        if self.use_depthwise_attention:
            x, vp, kp = x
            if not self.first_block:
                queries = self.queries(x)
                _, qp = ops.split(queries, 2, axis=3)
                if self.combine_queries:
                    for i, query in enumerate(ops.split(qp, self.num_queries, axis=-1)):
                        attn = attention_block(query, kp, vp)
                        if i == 0:
                            attn_p = attn
                        else:
                            attn_p = ops.concatenate([attn_p, attn], axis=-1)
                    attn_p = self.queries_comb_p(attn_p)
                else:
                    attn_p = attention_block(qp, kp, vp)
                attn_p += ops.gelu(attn_p)

                prior = self.p(x + self.gamma_p * attn_p)
            else:
                prior = self.p(x)

            pm, p_std, vpl, kpl = (
                prior[:, :, :, : self.zdim],
                prior[:, :, :, self.zdim : self.zdim * 2],
                prior[:, :, :, self.zdim * 2 : self.zdim * 2 + self.in_width],
                prior[:, :, :, self.zdim * 2 + self.in_width :],
            )

            x = ops.add(x, vpl)  # (B, H, W, in_width)

            if self.first_block:
                vp = ops.expand_dims(vpl, axis=-1)
                kp = ops.expand_dims(kpl, axis=-1)
            else:
                vp = ops.concatenate([vp, ops.expand_dims(vpl, axis=-1)], axis=-1)
                kp = ops.concatenate([kp, ops.expand_dims(kpl, axis=-1)], axis=-1)
            x = (x, vp, kp)
        else:
            prior = self.p(x)
            pm, p_std, vpl = (
                prior[:, :, :, : self.zdim],
                prior[:, :, :, self.zdim : self.zdim * 2],
                prior[:, :, :, self.zdim * 2 :],
            )
            x = ops.add(x, vpl)

        p_std = self.sp(p_std)
        noise = random.normal(ops.shape(p_std), seed=self.seed_gen)
        z = pm + p_std * noise * t
        # Flows are not used during unconditional sampling
        return x, z

    def call(self, x, act, training):
        x, z, kl = self.sample(x, act, training=training)
        if self.use_depthwise_attention:
            x, vp, kp = x
        if not self.last_block:
            x = ops.add(x, self.z_proj(z))
            x = self.res(x)
        if self.use_depthwise_attention:
            x = (x, vp, kp)

        z = self.z_out_f(z)
        z = self.z_out_up(z)
        return x, z, kl

    def call_uncond(self, x, t=1):
        x, z = self.sample_uncond(x, t)
        if self.use_depthwise_attention:
            x, vp, kp = x
        if not self.last_block:
            x = ops.add(x, self.z_proj(z))
            x = self.res(x)
        if self.use_depthwise_attention:
            x = (x, vp, kp)

        z = self.z_out_f(z)
        z = self.z_out_up(z)
        return x, z


class Block(layers.Layer):
    # Represents a single ResNet block
    def __init__(
        self,
        input_size,
        in_width,
        middle_width,
        out_width,
        kernelsize,
        activation,
        bn,
        residual,
        zero_last,
        model_depth,
        depthwise,
        dwa_enc=False,
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
        self.dwa_enc = dwa_enc

        # Groupnorm layers
        self.gn1 = (
            layers.GroupNormalization(groups=in_width // 8) if bn else layers.Identity()
        )
        self.gn2 = (
            layers.GroupNormalization(groups=middle_width // 8)
            if bn
            else layers.Identity()
        )
        self.gn3 = (
            layers.GroupNormalization(groups=middle_width // 8)
            if bn
            else layers.Identity()
        )
        self.gn4 = (
            layers.GroupNormalization(groups=middle_width // 8)
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
            self.c2 = layers.Conv2D(
                middle_width, kernel_size=kernelsize, padding="same"
            )
            self.c3 = layers.Conv2D(
                middle_width, kernel_size=kernelsize, padding="same"
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
        xhat = self.c1(self.activation(self.gn1(x)))
        xhat = self.c2(self.activation(self.gn2(xhat)))
        xhat = self.c3(self.activation(self.gn3(xhat)))
        xhat = self.c4(self.activation(self.gn4(xhat)))

        if self.dwa_enc and self.in_width < self.out_width:
            out = ops.concatenate(
                [
                    ops.add(xhat[:, :, :, : self.in_width], x),  # vq
                    xhat[:, :, :, self.in_width :],  # kq
                ],
                axis=-1,
            )
        elif self.dwa_enc and self.in_width > self.out_width:
            out = ops.add(xhat, x[:, :, :, : self.out_width])  # vq
        else:
            out = ops.add(x, xhat) if self.residual else xhat
        return out


class PoolLayer(layers.Layer):
    # Pooling / Unpooling layer, comes with optional activation and 1x1 conv to change channel width
    def __init__(
        self, input_size, in_width, out_width, pool_activation, unpool, data_size=None
    ):
        super().__init__()
        # Building
        self.input_size = input_size
        self.in_width = in_width
        self.out_width = out_width
        self.activation = pool_activation

        # Layers
        if in_width == out_width:
            self.c1 = layers.Identity()
        else:
            self.c1 = layers.Conv2D(out_width, kernel_size=1)

        if unpool:
            self.pool = (
                layers.UpSampling2D(2)
                if (input_size < data_size)
                else layers.Identity()
            )
        else:
            self.pool = (
                layers.AveragePooling2D(2) if (input_size > 1) else layers.Identity()
            )

    def build(self):
        _ = self.call(
            random.normal([1, self.input_size, self.input_size, self.in_width])
        )

    def call(self, x):
        return self.pool(self.activation(self.c1(x)))
