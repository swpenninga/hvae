from attention.attention import prepare_activations
from utils.distributions import SoftPlus
from building_blocks import DecoderStage, EncoderStage, Block
from keras import Model, Sequential, Variable, layers, ops, random


class VAE(Model):
    # HVAE model, containing encoder and decoder with some helper functions
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        self.beta_max = params.beta
        self.beta_warmup_epochs = params.beta_warmup_epochs

        beta_start = 0 if self.beta_warmup_epochs > 0 else params.beta
        self.beta = Variable(beta_start, trainable=False, dtype="float32")
        self.cr_vae = params.cr_vae

        self.loss_fn = params.loss_fn
        self.ndims = params.enc_input_size[0] ** 2 * params.data_width
        self.gradient_smoothing = params.gradient_smoothing
        self.min_mol_logscale = -250
        self.softplus = SoftPlus(0.69314718056)

    def build(self):
        print("Building HVAE")
        self.encoder.build()
        self.decoder.build()
        _ = self.call(
            random.normal(
                [
                    2,  # Batch_size is at least 2 in case of cr-vae
                    self.params.enc_input_size[0],
                    self.params.enc_input_size[0],
                    self.params.data_width,
                ]
            ),
            training=True,
        )
        _ = self.call(
            random.normal(
                [
                    2,  # Batch_size is at least 2 in case of cr-vae
                    self.params.enc_input_size[0],
                    self.params.enc_input_size[0],
                    self.params.data_width,
                ]
            ),
            training=False,
        )

    def get_elbo(self, x, px_z, kl, training):
        recon_total = self.loss_fn(x, px_z)
        kl_total = ops.zeros(ops.shape(x)[0])

        r_total = ops.zeros(ops.shape(x)[0] // 2)

        for kls_stage in kl:
            for kl_block in kls_stage:
                # Check which scenario we are in
                if self.cr_vae and training:
                    if isinstance(kl_block[0], (list, tuple)) and len(kl_block[0]) == 2:
                        # Flow, cr-vae; kl_block = ((kl, R_loss), log_det_j)
                        kl_total += ops.sum(kl_block[0][0], axis=[1, 2, 3])
                        kl_total -= kl_block[1]
                        # r for a flow stage is always 0, no need to add.
                    else:
                        # Flow, no cr-vae; kl_block = (kl, R_loss)
                        kl_total += ops.sum(kl_block[0], axis=[1, 2, 3])
                        r_total += ops.sum(kl_block[1], axis=[1, 2, 3])
                else:
                    if isinstance(kl_block, (list, tuple)) and len(kl_block) == 2:
                        # Flow, no cr-vae; kl_block = (kl, log_det_j)
                        kl_total += ops.sum(kl_block[0], axis=[1, 2, 3])
                        kl_total -= kl_block[1]
                    else:
                        # No Flow, no cr-vae; kl_block = kl
                        kl_total += ops.sum(kl_block, axis=[1, 2, 3])

        if self.params.channels_out > 1:
            # bits/dim for RGB
            recon_total /= self.ndims * ops.log(2)
            kl_total /= self.ndims * ops.log(2)
            r_total /= self.ndims * ops.log(2)

        recon_total = ops.mean(recon_total)
        kl_total = ops.mean(kl_total)
        r_total = ops.mean(r_total) * self.params.cr_lambda
        elbo = recon_total + self.beta * (kl_total + r_total)
        return elbo, recon_total, kl_total

    def call(self, x, training):
        # The encoder gives [vq] per stage for normal vdvae
        # and gives [vq, kq] per stage for attentative vdvae
        activations = self.encoder(x)
        px_z, z, kl = self.decoder(activations, training=training)
        return px_z, z, kl

    # # # # MR MOL # # # # #
    #       _.-=-._        #
    #    o~`  '  > `.      #
    #    `.  ,       :     #
    #     `"-.__/    `.    #
    #        /       ::    #
    #       / .:    .:|    #
    #      :       .::!.   #
    #     /'| ::  :::'!!   #
    #   .:  :/' .::/  !!   #
    #   :::/   :::'   !!   #
    #   `:"::'''!!    !!   #
    #     /          :!!.  #
    #    /     .-~-:  !!!  #
    #   /:   :'        !!. #
    #  :::  :'          !! #
    #  |::  |        :!!!! #
    #  `::  :        !!!!' #
    #   |:. `:    .  '!!!  #
    #   `::.  \   `::. !'  #
    #    _.`::.\     ::    #
    # .-~_____:~~    :     #
    # ~~~  .-'__..-~'      #
    #      ~~~~            #
    # # # # MR MOL # # # # #
    def sample_from_mol(self, logits, t=1.0):
        # Sampling from the mixture of logistic functions
        B, H, W, _ = ops.shape(logits)

        # Unpack parameters
        logit_probs = logits[:, :, :, : self.params.num_output_mixtures]  # B, H, W, M
        l = logits[:, :, :, self.params.num_output_mixtures :]  # B, H, W, M*C*3
        l = layers.Reshape(
            [H, W, self.params.data_width, 3 * self.params.num_output_mixtures]
        )(l)  # B, H, W, C, 3*M

        # sample mixture indicator from softmax
        gumbel_noise = -ops.log(
            -ops.log(
                random.uniform(
                    (B, H, W, self.params.num_output_mixtures),
                    minval=1e-5,
                    maxval=1.0 - 1e-5,
                )
            )
        )  # B, H, W, M
        amax = ops.argmax(logit_probs / t + gumbel_noise, axis=-1)  # B, H, W
        lambda_ = ops.cast(
            ops.one_hot(amax, self.params.num_output_mixtures, axis=-1), "float32"
        )  # B, H, W, M
        lambda_ = ops.expand_dims(lambda_, axis=3)  # B, H, W, 1, M

        means = ops.sum(
            l[:, :, :, :, : self.params.num_output_mixtures] * lambda_, axis=-1
        )  # B, H, W, C

        log_scales = self.min_mol_logscale + self.softplus(
            l[
                :,
                :,
                :,
                :,
                self.params.num_output_mixtures : 2 * self.params.num_output_mixtures,
            ]
            - self.min_mol_logscale
        )  # B, H, W, C
        log_scales = ops.sum(log_scales * lambda_, axis=-1)  # B, H, W, C

        coeffs = ops.sum(
            ops.tanh(
                l[
                    :,
                    :,
                    :,
                    :,
                    2 * self.params.num_output_mixtures : 3
                    * self.params.num_output_mixtures,
                ]
            )
            * lambda_,
            axis=-1,
        )  # B, H, W, C

        # Sample from logistic
        u = random.uniform(
            (B, H, W, self.params.data_width), minval=1e-5, maxval=1.0 - 1e-5
        )
        x = means + ops.exp(log_scales) * t * (
            ops.log(u) - ops.log(1.0 - u)
        )  # B, H, W, C

        # Auto-regressive sampling RGB and clip
        if self.params.data_width == 3:
            x0 = ops.clip(x[:, :, :, 0:1], -1, 1)
            x1 = ops.clip(x[:, :, :, 1:2] + coeffs[:, :, :, 0:1] * x0, -1, 1)
            x2 = ops.clip(
                x[:, :, :, 2:3] + coeffs[:, :, :, 1:2] * x0 + coeffs[:, :, :, 2:3] * x1,
                -1,
                1,
            )
            x = ops.concatenate([x0, x1, x2], axis=-1)  # B, H, W, C
        else:
            x = ops.clip(x, -1, 1)

        return x

    def print_model(self):
        print("------ Encoder ------")
        print(
            (
                f"first 1x1 conv: [None, {self.params.enc_input_size[0], self.params.enc_input_size[0], self.params.data_width}]"
                f" -> [None, {self.params.enc_input_size[0], self.params.enc_input_size[0], self.encoder.first_conv.weights[0].shape[-1]}]"
            )
        )
        for stage_num, stage in enumerate(self.encoder.stages.layers):
            kernel_size = self.params.kernelsizes[str(stage.input_size)]
            stage_params = 0
            for block in stage.blocks.layers:
                stage_params += block.count_params()
            stage_params += stage.pool.count_params()
            print(
                (
                    f"stage:{stage_num:2} - input_size:{stage.input_size:3} - in_width:{stage.in_width:3}"
                    f" - middle_width:{stage.middle_width:3} - #blocks:{stage.num_blocks:2} - ksize:{kernel_size:2} - #params:{stage_params:,d}"
                )
            )
        print("------ Decoder ------")
        for stage_num, stage in enumerate(self.decoder.stages.layers):
            stage_params = 0
            for block in stage.blocks.layers:
                if self.params.use_depthwise_attention:
                    stage_params += block.queries.count_params()
                    if block.combine_queries:
                        stage_params += block.queries_comb_q.count_params()
                        if not block.first_block:
                            stage_params += block.queries_comb_p.count_params()
                stage_params += block.q.count_params()

                stage_params += block.p.count_params()
                stage_params += block.z_out_f.count_params()
                if not block.last_block:
                    stage_params += block.res.count_params()
                    stage_params += block.z_proj.count_params()
            stage_params += stage.pool.count_params()

            print(
                (
                    f"stage:{self.params.num_stages - stage_num - 1:2} - input_size:{stage.input_size:3} - in_width:{stage.in_width:3}"
                    f" - middle_width:{stage.middle_width:3} - #layers:{stage.num_blocks:2} - z_ch:{stage.zdim:3} - #params:{stage_params:,d}"
                )
            )
        output_blocks_params = 0
        for block in self.decoder.output_blocks.layers:
            output_blocks_params += block.count_params()
        print(
            (
                f"--- Output ---\n"
                f"blocks   - input_size:{self.decoder.output_shape[1]:3} - in_width:{self.params.z_out_width:3} - middle_width:{self.params.z_out_middle_width:3} - #blocks:{self.decoder.num_output_blocks:2} - ksize:{self.params.kernelsizes[str(self.params.dec_input_size[-1])]:2} - #params:{output_blocks_params:,d}\n"
                f"last 3x3 conv: [None, {stage.input_size, stage.input_size, self.params.z_out_width}]"
                f" -> [None, {stage.input_size, stage.input_size, self.decoder.last_conv.weights[0].shape[-1]}]"
            )
        )
        print("------  Flows  ------")
        if self.params.flow_type == "none":
            print(f"        {None}")
        else:
            for stage_num, stage in enumerate(self.decoder.stages.layers):
                stage_params = 0
                for block in stage.blocks.layers:
                    if block.use_flow:
                        stage_params += block.flows.count_params()
                if self.params.flow_type == "sylvester":
                    print(
                        f"stage:{self.params.num_stages - stage_num - 1:2} - input_size:{stage.input_size:3} - eff_zdim:{stage.zdim * stage.input_size * stage.input_size:5} - #flows:{stage.num_flows:2} - num_ortho_vecs:{stage.num_ortho_vecs:2} - flow_in_ch:{stage.flow_in_ch:2} - #params:{stage_params:,d}"
                    )
                else:
                    channels = 0
                    flows_per_level = 0
                    split_first = False
                    n_levels = 0
                    use_flow = False
                    if len(stage.blocks.layers) > 0:
                        if stage.blocks.layers[0].use_flow:
                            channels = stage.blocks.layers[0].flows.sylv_channels
                            flows_per_level = stage.blocks.layers[
                                0
                            ].flows.flows_per_level
                            split_first = stage.blocks.layers[0].flows.split_first
                            n_levels = stage.blocks.layers[0].flows.n_levels
                            use_flow = True
                    print(
                        f"stage:{self.params.num_stages - stage_num - 1:2} - input_size:{stage.input_size:3} - eff_zdim:{stage.zdim * stage.input_size * stage.input_size:5} - flow:{use_flow:1} - width: {channels:3} - #flows_per_level:{flows_per_level} - splitfirst: {split_first:1} - n_levels: {n_levels:2} - #params:{stage_params:,d}"
                    )
            print(f"flow_type:      {self.params.flow_type}")
            if self.params.flow_type == "conv_sylvester":
                print(f"spectral_norm:  {self.params.spectral_norm}")

        print("---- Attention parameters ----")
        print(f"depthwise_attention: {self.params.use_depthwise_attention}")
        if self.params.use_depthwise_attention:
            print(f"query_width:         {self.params.query_width}")
            print(f"num_queries:         {self.params.num_queries}\n")

        print("---- Model settings ----")
        print(
            (
                f"init_zeros:           {self.params.init_zeros}\n"
                f"block_act:            {self.params.b_act_name}\n"
                f"pool_act:             {self.params.p_act_name}\n"
                f"group_norm:           {self.params.block_bn}\n"
                f"depthwise_conv:       {self.params.depthwise}\n"
                f"model_depth:          {self.params.model_depth}\n"
                f"num_enc_blocks:       {sum(self.params.enc_num_blocks)}\n"
                f"num_output_mix:       {self.params.num_output_mixtures}\n"
                f"grad_smoothing:       {self.params.gradient_smoothing}"
            )
        )

        print("---- Training parameters ----")
        print(
            f"dataset:              {self.params.dataset}\n"
            f"batch_size:           {self.params.batch_size * self.params.batch_size_div}\n"
            f"batch_size_div:       {self.params.batch_size_div}\n"
            f"cr_vae:               {self.params.cr_vae}"
        )
        if self.params.cr_vae:
            print(
                f"  cr_lambda:          {self.params.cr_lambda}\n"
                f"  cr_duplicate_noise: {self.params.cr_duplicate_noise}"
            )
        print(
            f"epochs:               {self.params.epochs}\n"
            f"beta_max:             {self.params.beta}\n"
            f"  beta_warm_up:       {self.params.beta_warmup_epochs}\n"
            f"  beta_cyclic:        {self.params.cyclic_beta}"
        )
        if self.params.cyclic_beta:
            print(f"  beta_#_cycles:      {self.params.num_cycles}")
        print(
            f"learning_rate:        {self.params.learning_rate}\n"
            f"  lr_end:             {self.params.learning_rate_end}\n"
            f"  lr_warmup:          {self.params.lr_warmup_epochs}\n"
            f"optimizer:            {self.params.optimizer}\n"
            f"  ema:                {self.params.use_ema}\n"
            f"  weight_decay:       {self.params.weight_decay}\n"
            f"  scheduler:          {self.params.scheduler}\n"
            f"  grad_clipnorm:      {self.params.gradient_clipnorm}\n"
            f"  grad_skipnorm:      {self.params.gradient_skipnorm}\n"
            f"save:                 {self.params.save}\n"
            f"jit:                  {self.params.jit}\n"
            f"gpu:                  {self.params.gpu}"
        )

        if self.params.save:
            print(f"train_folder:          {self.params.train_folder}")


class Decoder(layers.Layer):
    # Decoder, starts with bias or 0, moves through stages, sums z's and puts it in output blocks + conv
    def __init__(self, params):
        super().__init__()
        self.output_shape = [
            1,
            params.dec_input_size[-1],
            params.dec_input_size[-1],
            params.dec_in_width[-1],
        ]
        self.z_features_shape = (
            1,
            params.z_out[0],
            params.z_out[1],
            params.z_out[2],
        )
        self.num_stages = params.num_stages
        self.z_to_features = layers.Conv2D(
            params.z_out_width, kernel_size=1, padding="same"
        )
        self.model_depth = params.model_depth
        self.num_output_blocks = params.output_blocks
        self.init_shape = [
            params.dec_input_size[0],
            params.dec_input_size[0],
            params.dec_in_width[0],
        ]
        self.init_zeros = params.init_zeros
        if not self.init_zeros:
            self.init_bias = self.add_weight(
                shape=(1, self.init_shape[0], self.init_shape[1], self.init_shape[2]),
                initializer="zeros",
                trainable=True,
            )

        self.use_depthwise_attention = params.use_depthwise_attention

        self.stages = Sequential()
        for num in range(self.num_stages):
            self.stages.add(DecoderStage(params, num))

        self.activation = params.block_activation
        self.output_blocks = Sequential()
        for num in range(self.num_output_blocks):
            self.output_blocks.add(
                Block(
                    input_size=self.output_shape[1],
                    in_width=params.z_out_width,
                    middle_width=params.z_out_middle_width,
                    out_width=params.z_out_width,
                    kernelsize=params.kernelsizes[str(params.dec_input_size[-1])],
                    activation=params.block_activation,
                    bn=params.block_bn,
                    residual=True,
                    zero_last=False,
                    model_depth=params.model_depth,
                    depthwise=params.depthwise,
                )
            )
        self.last_conv = layers.Conv2D(
            params.channels_out,
            kernel_size=params.kernelsizes[str(params.dec_input_size[-1])],
            padding="same",
        )

    def build(self):
        for dec_stage in self.stages.layers:
            dec_stage.build()
        for out_block in self.output_blocks.layers:
            out_block.build()

    def depthwise_attention_call(self, activations, training):
        z_stages = []
        kl_stages = []

        if self.init_zeros:
            x = ops.zeros(
                shape=(
                    ops.shape(activations[0][0])[0],
                    self.init_shape[0],
                    self.init_shape[1],
                    self.init_shape[2],
                )
            )
        else:
            b = ops.shape(activations[0][0])[0]
            x = ops.repeat(self.init_bias, b, axis=0)

        # vp and kp get overwritten in first call, theyre placeholders
        vp = ops.zeros(1)
        kp = ops.zeros(1)

        # pack top-down stream as x, vp, kp
        x = (x, vp, kp)

        # We give multiple activations to each decoder stage, to have
        # v^q_>l and k^q_>l
        # activations are 32x32, 16x16, 8x8, etc.
        # we convert a list with different resolutions to a
        # list containing tensors of same resolution with dimension (layers)
        activations = prepare_activations(activations)

        for dec_stage, act in zip(self.stages.layers, reversed(activations)):
            # First stage gets [32x32, 16x16, ..., 1x1], last stage gets [32x32]
            x, z, kl = dec_stage(x, act, training=training)
            z_stages += [z]
            kl_stages += [kl]
        return x, z_stages, kl_stages

    def call(self, activations, training):
        if self.use_depthwise_attention:
            x, z_stages, kl_stages = self.depthwise_attention_call(
                activations, training=training
            )
        else:
            z_stages = []
            kl_stages = []

            if self.init_zeros:
                x = ops.zeros_like(activations[-1])
            else:
                b = ops.shape(activations[-1])[0]
                x = ops.repeat(self.init_bias, b, axis=0)

            for dec_stage, act in zip(self.stages.layers, reversed(activations)):
                x, z, kl = dec_stage(x, act, training=training)
                z_stages.append(z)
                kl_stages.append(kl)

        # Sum all the latents of all stages
        z_out = sum(z_stages) / ops.sqrt(self.model_depth)

        # Create output
        px_z = self.activation(self.z_to_features(z_out))
        for out_block in self.output_blocks.layers:
            px_z = out_block(px_z)
        px_z = self.last_conv(px_z)

        return px_z, z_stages, kl_stages

    def call_uncond(self, num_images=16, t=1):
        if self.init_zeros:
            x = ops.zeros(
                (num_images, self.init_shape[0], self.init_shape[1], self.init_shape[2])
            )
        else:
            x = ops.repeat(self.init_bias, num_images, axis=0)

        if self.use_depthwise_attention:
            # vp and kp get overwritten in first call, these are placeholders
            vp = ops.zeros(1)
            kp = ops.zeros(1)
            # pack top-down stream as x, vp, kp
            x = (x, vp, kp)

        z_stages = []
        for dec_stage in self.stages.layers:
            x, z = dec_stage.call_uncond(x, t)
            z_stages.append(z)

        z_out = sum(z_stages) / ops.sqrt(self.model_depth)

        px_z = self.activation(self.z_to_features(z_out))
        for out_block in self.output_blocks.layers:
            px_z = out_block(px_z)
        px_z = self.last_conv(px_z)

        return px_z


class Encoder(layers.Layer):
    # Encoder of the HVAE
    def __init__(self, params):
        super().__init__()
        self.input_shape = [
            1,
            params.enc_input_size[0],
            params.enc_input_size[0],
            params.data_width,
        ]
        self.num_stages = params.num_stages
        self.first_conv = layers.Conv2D(
            params.enc_in_width[0],
            kernel_size=params.kernelsizes[str(params.enc_input_size[0])],
            padding="same",
        )

        self.stages = Sequential()
        for num in range(self.num_stages):
            self.stages.add(EncoderStage(params, num))

    def build(self):
        for enc_stage in self.stages.layers:
            enc_stage.build()
        _ = self.call(random.normal(self.input_shape))

    def call(self, x):
        x = self.first_conv(x)
        activations = []
        for enc_stage in self.stages.layers:
            # If there is no depth_wise attention, activations are passed to decoder
            # If there is depth_wise attention, act[0]=v^q, act[1]=k^q
            # every v^q and k^q is at a different resolution,
            # pooling/upsampling happens at attention block
            x, act = enc_stage(x)
            activations += [act]
        return activations
