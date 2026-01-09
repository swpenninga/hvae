import datetime
import os
import pickle

import numpy as np
from keras import layers, ops, optimizers
from losses import BinaryCE, DiscMixLogistic


# Here we store and convert the input arguments to training parameters
class Parameters:
    def __init__(self, args):
        super().__init__()
        self.save = args.save
        self.save_dir = args.save_dir
        self.dataset = args.dataset
        self.batch_size_div = max(1, args.batch_size_div)
        self.batch_size = int(args.batch_size // self.batch_size_div)
        self.cr_vae = args.cr_vae
        self.cr_lambda = args.cr_lambda
        self.cr_duplicate_noise = args.cr_duplicate_noise

        self.jit = "auto" if args.jit else False
        self.gpu = args.gpu

        # Model parameters
        self.b_act_name = args.b_act
        self.p_act_name = args.p_act
        self.block_activation = self.get_activation(args.b_act)
        self.pool_activation = self.get_activation(args.p_act)
        self.init_zeros = args.init_zeros
        self.block_bn = args.block_gn
        self.depthwise = args.depthwise
        self.num_output_mixtures = args.num_output_mixtures

        # Attention
        self.use_depthwise_attention = args.use_depthwise_attention
        self.query_width = args.query_width
        self.num_queries = args.num_queries

        # Gradient smoothing and skipping
        self.gradient_smoothing = args.gradient_smoothing
        self.gradient_clipnorm = (
            args.gradient_clipnorm if args.gradient_clipnorm > 0 else None
        )
        self.gradient_skipnorm = (
            args.gradient_skipnorm if args.gradient_skipnorm > 0 else 1e9
        )  # If it is 0, we make the threshold very large to effectively turn it off

        # Flow parameters
        self.flow_type = args.flow_type
        self.spectral_norm = args.spectral_norm
        self.add_dataset_context()  # Writes new params
        self.set_stage_parameters(args)  # Writes new params

        # Number of stochastic layers
        self.model_depth = sum(self.dec_num_blocks)

        # Hyper parameters
        self.epochs = int(args.epochs)
        self.early_stopping = int(args.early_stopping)
        self.beta = float(args.beta)
        self.beta_warmup_epochs = int(args.beta_warmup_epochs)
        self.cyclic_beta = bool(args.cyclic_beta)
        self.num_cycles = int(args.number_cycles)

        self.learning_rate = float(args.learning_rate)
        self.learning_rate_end = float(args.learning_rate_end)
        self.lr_warmup_epochs = int(args.lr_warmup_epochs)
        self.weight_decay = float(args.weight_decay)
        self.use_ema = bool(args.use_ema)

        self.optimizer = str(args.optimizer)
        self.scheduler = str(args.scheduler)

        self.get_kernelsizes(args.increase_kernelsize)

        # Verify that configuration meets requirements
        self.verify_parameters()

        # Create a save folder
        if self.save:
            self.create_savefolder()
            with open(os.path.join(self.save_path, "args.pkl"), "wb") as f:
                pickle.dump(args, f)
        else:
            self.save_path = None

    def get_activation(self, name):
        # Turns string into keras activation
        return layers.Activation(name)

    def add_dataset_context(self):
        # Sets loss function and data related parameters based on dataset
        if self.dataset == "cifar10":
            self.enc_input_size = [32, 16, 8, 4, 2, 1]
            self.data_width = 3
            self.n_bits = 8
            self.loss_fn = DiscMixLogistic(
                self.n_bits,
                self.num_output_mixtures,
                self.data_width,
            )
            self.channels_out = (self.data_width * 3 + 1) * self.num_output_mixtures
            train_len = 50000

        elif self.dataset == "mnist":
            self.enc_input_size = [32, 16, 8, 4, 2, 1]
            self.data_width = 1
            self.n_bits = 8
            self.loss_fn = BinaryCE(self.dataset)
            self.channels_out = 1
            train_len = 60000

        elif self.dataset == "imagenet32":
            self.enc_input_size = [32, 16, 8, 4, 2, 1]
            self.data_width = 3
            self.n_bits = 8
            self.loss_fn = DiscMixLogistic(
                self.n_bits,
                self.num_output_mixtures,
                self.data_width,
            )
            self.channels_out = (self.data_width * 3 + 1) * self.num_output_mixtures
            train_len = 1281167  # 1281167
        else:
            raise ValueError("No valid dataset was selected")

        self.step_per_epoch = int(
            np.floor(train_len / self.batch_size)
        )  # We floor since drop_remainder=True in the dataloaders

        self.dec_input_size = list(reversed(self.enc_input_size))
        self.num_stages = len(self.enc_input_size)

    def set_stage_parameters(self, args):
        # Repeats the parameters to the number of stages, if only a single input is given.
        # Reverses the decoder params to go from top-down to bottom-up.
        # Determines the pool widths for the convs inbetween stages.
        self.zdim = (
            list(reversed(args.z_width))
            if (len(args.z_width) == self.num_stages)
            else args.z_width * self.num_stages
        )
        self.num_flows = (
            list(reversed(args.num_flows))
            if (len(args.num_flows) == self.num_stages)
            else args.num_flows * self.num_stages
        )
        self.flow_in_ch = (
            list(reversed(args.flow_in_ch))
            if (len(args.flow_in_ch) == self.num_stages)
            else args.flow_in_ch * self.num_stages
        )
        self.num_ortho_vecs = (
            list(reversed(args.num_ortho_vecs))
            if (len(args.num_ortho_vecs) == self.num_stages)
            else args.num_ortho_vecs * self.num_stages
        )

        self.convsylv_channels = (
            list(reversed(args.convsylv_channels))
            if (len(args.convsylv_channels) == self.num_stages)
            else args.convsylv_channels * self.num_stages
        )
        self.convsylv_flows_per_stage = (
            list(reversed(args.convsylv_flows_per_stage))
            if (len(args.convsylv_flows_per_stage) == self.num_stages)
            else args.convsylv_flows_per_stage * self.num_stages
        )

        self.convsylv_splitfirst = (
            list(reversed(args.convsylv_splitfirst))
            if (len(args.convsylv_splitfirst) == self.num_stages)
            else args.convsylv_splitfirst * self.num_stages
        )

        self.convsylv_stage_limit = (
            list(reversed(args.convsylv_stage_limit))
            if (len(args.convsylv_stage_limit) == self.num_stages)
            else args.convsylv_stage_limit * self.num_stages
        )

        self.enc_in_width = (
            args.stage_in_width
            if (len(args.stage_in_width) == self.num_stages)
            else args.stage_in_width * self.num_stages
        )
        self.enc_middle_width = (
            args.enc_middle_width
            if (len(args.enc_middle_width) == self.num_stages)
            else args.enc_middle_width * self.num_stages
        )
        self.enc_num_blocks = (
            args.enc_num_blocks
            if (len(args.enc_num_blocks) == self.num_stages)
            else args.enc_num_blocks * self.num_stages
        )
        self.enc_pool_width = np.roll(self.enc_in_width, -1)
        self.enc_pool_width[-1] = self.enc_pool_width[-2]

        self.dec_in_width = list(reversed(self.enc_in_width))
        self.dec_middle_width = (
            list(reversed(args.dec_middle_width))
            if (len(args.dec_middle_width) == self.num_stages)
            else args.dec_middle_width * self.num_stages
        )
        self.dec_num_blocks = (
            list(reversed(args.dec_num_blocks))
            if (len(args.dec_num_blocks) == self.num_stages)
            else args.dec_num_blocks * self.num_stages
        )
        self.dec_pool_width = np.roll(self.dec_in_width, -1)
        self.dec_pool_width[-1] = self.dec_pool_width[-2]
        self.output_blocks = args.output_blocks

        self.z_out = (
            self.dec_input_size[-1],
            self.dec_input_size[-1],
            self.dec_in_width[-1],
        )
        self.z_out_width = args.z_out_width
        self.z_out_middle_width = args.z_out_middle_width

    def get_optimizer(self):
        # Returns the optimizer and scheduler based on the input arguments
        if self.scheduler == "none":
            lr_scheduler = self.learning_rate
        elif self.scheduler == "exp":
            lr_scheduler = optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=self.step_per_epoch,
                decay_rate=0.977,  # 100 epochs for 10x decrease
            )
        elif self.scheduler == "cosd":
            w_steps = self.lr_warmup_epochs * self.step_per_epoch
            lr_scheduler = optimizers.schedules.CosineDecay(
                initial_learning_rate=1e-6,
                decay_steps=self.step_per_epoch * self.epochs,
                alpha=self.learning_rate_end,
                warmup_target=self.learning_rate,
                warmup_steps=w_steps,
            )
        elif self.scheduler == "cosdr":
            lr_scheduler = optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=self.learning_rate,
                first_decay_steps=self.step_per_epoch * 2,
                t_mul=2.0,
                m_mul=1.0,
                alpha=0.0,
            )
        else:
            raise ValueError("invalid lr scheduler")

        grad_acc = self.batch_size_div if self.batch_size_div > 1 else None
        if self.optimizer == "adamax":
            opt = optimizers.Adamax(
                learning_rate=lr_scheduler,
                epsilon=1e-7,
                beta_1=0.9,
                beta_2=0.999,
                global_clipnorm=self.gradient_clipnorm,
                use_ema=self.use_ema,
                ema_momentum=0.99,
                weight_decay=self.weight_decay,
                gradient_accumulation_steps=grad_acc,
            )
        elif self.optimizer == "adamw":
            opt = optimizers.AdamW(
                learning_rate=lr_scheduler,
                global_clipnorm=self.gradient_clipnorm,
                use_ema=self.use_ema,
                ema_momentum=0.9999,
                weight_decay=self.weight_decay,
                gradient_accumulation_steps=grad_acc,
            )
        elif self.optimizer == "sgd":
            opt = optimizers.SGD(
                learning_rate=lr_scheduler,
                global_clipnorm=self.gradient_clipnorm,
                use_ema=self.use_ema,
                ema_momentum=0.9999,
                weight_decay=self.weight_decay,
                gradient_accumulation_steps=grad_acc,
            )
        else:
            raise ValueError("invalid optimizer")
        return opt

    def create_savefolder(self):
        # Creates a savefolder for epoch logging, configuration, weights and images
        date = (
            str(datetime.datetime.now())[4:16]
            .replace(" ", "_")
            .replace("-", "_")
            .replace(":", "_")
        )
        save_path = os.path.join(self.save_dir, self.dataset + date)

        if not os.path.exists(save_path):
            self.save_path = save_path
            self.train_folder = self.dataset + date
        else:
            for i in range(1, 10000):
                if not os.path.exists(save_path + f"_{i}"):
                    self.save_path = save_path + f"_{i}"
                    self.train_folder = self.dataset + date + f"_{i}"
                    break
        os.makedirs(self.save_path)
        os.makedirs(os.path.join(self.save_path, "reconstructions"))

    def get_kernelsizes(self, increase_kernelsize):
        if increase_kernelsize:
            # Growing kernelsizes with resolution
            self.kernelsizes = {
                "1024": 13,
                "512": 11,
                "256": 7,
                "128": 5,
                "64": 3,
                "32": 3,
                "16": 3,
                "8": 3,
                "4": 3,
                "2": 1,
                "1": 1,
            }
        else:
            # kernelsize is always 3 or 1
            self.kernelsizes = {
                "1024": 3,
                "512": 3,
                "256": 3,
                "128": 3,
                "64": 3,
                "32": 3,
                "16": 3,
                "8": 3,
                "4": 3,
                "2": 1,
                "1": 1,
            }

    def verify_parameters(self):
        err = "Length of this parameter must equal number of stages"
        assert len(self.zdim) == self.num_stages, err
        assert len(self.enc_in_width) == self.num_stages, err
        assert len(self.enc_middle_width) == self.num_stages, err
        assert len(self.enc_num_blocks) == self.num_stages, err
        assert len(self.dec_in_width) == self.num_stages, err
        assert len(self.dec_middle_width) == self.num_stages, err
        assert len(self.dec_num_blocks) == self.num_stages, err

        assert self.beta_warmup_epochs == int(self.beta_warmup_epochs), (
            "beta_warmup_epochs should be integer"
        )
        assert self.num_output_mixtures == int(self.num_output_mixtures), (
            "num_output_mixtures should be integer"
        )

        if self.cyclic_beta:
            assert self.num_cycles * self.beta_warmup_epochs < self.epochs, (
                "Not enough epochs to complete beta scheduler cycles"
            )

        assert self.beta_warmup_epochs >= 0, "No negative warmup epochs"

        for i in range(len(self.enc_input_size) - 1):
            assert self.enc_input_size[i] // 2 == self.enc_input_size[i + 1], (
                "Every input_size should be a decreasing power of 2"
            )
            assert ops.log2(self.enc_input_size[i]) == int(
                ops.log2(self.enc_input_size[i])
            ), "Every input_size should be an integer power of 2"

        if self.flow_type != "none":
            for vec, z, num, input_shape in zip(
                self.num_ortho_vecs, self.zdim, self.num_flows, self.enc_input_size
            ):
                if num > 0:
                    assert vec <= z * (input_shape**2), (
                        "Number of orthogonal vectors cannot be larger than z_dimensions"
                    )

        if self.cr_vae:
            assert self.batch_size % 2 == 0, "Batch size must be even for CR-VAE"
