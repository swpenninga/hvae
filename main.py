import argparse
import warnings
import yaml

warnings.filterwarnings("ignore", category=UserWarning)

# General parser arguments
parser = argparse.ArgumentParser(description="Keras3 HierarchicalVAE")
parser.add_argument("--backend", type=str, default="tensorflow")
parser.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    choices=["mnist", "cifar10", "echonet", "imagenet32", "celeba64"],
)
parser.add_argument("--gpu", type=str, default="0")  # comma separated list of gpu ids
parser.add_argument("--jit", action="store_false", default=True)
parser.add_argument("--save", action="store_false", default=True)
parser.add_argument("--save_dir", type=str, default="out_folder/")

parser.add_argument("--b_act", type=str, default="silu")
parser.add_argument("--p_act", type=str, default="silu")
parser.add_argument(
    "--block_gn", action="store_true", default=False
)  # channels must be multiple of 8
parser.add_argument("--depthwise", action="store_true", default=False)
parser.add_argument("--num_output_mixtures", type=int, default=10)
parser.add_argument("--init_zeros", action="store_true", default=False)
parser.add_argument("--increase_kernelsize", action="store_true", default=False)

# Model parameters, ordering of list follows encoder stages: (0: 32x32, 1: 16x16, 2: 8x8, etc.)
# You can give 1 number (same for all stages) or a list of numbers (1 per stage)

# If using depth-wise attention, stage_in_width should be equal at all stages!
parser.add_argument("--stage_in_width", type=list, default=[64])
parser.add_argument("--enc_middle_width", type=list, default=[32])
parser.add_argument("--dec_middle_width", type=list, default=[32])
parser.add_argument("--z_width", type=list, default=[16])
parser.add_argument("--enc_num_blocks", type=list, default=[3])
parser.add_argument("--dec_num_blocks", type=list, default=[3])

# Parameters of reconstruction head (latent aggregation)
parser.add_argument("--output_blocks", type=int, default=3)
parser.add_argument("--z_out_width", type=int, default=64)
parser.add_argument("--z_out_middle_width", type=int, default=32)

# Flow parameters
parser.add_argument(
    "--flow_type",
    type=str,
    default="none",
    choices=["none", "sylvester", "conv_sylvester"],
)
# >0 is boolean for conv_sylvester and number of flows for sylvester
parser.add_argument("--num_flows", type=list, default=[0, 0, 2, 8, 12, 16])

##### Parameters for sylvester flow #####
parser.add_argument("--flow_in_ch", type=list, default=[0, 0, 1, 2, 4, 64])
parser.add_argument("--num_ortho_vecs", type=list, default=[0, 0, 8, 8, 8, 8])

##### Parameters for conv_sylvester flow #####
parser.add_argument(
    "--spectral_norm", action="store_false", default=True
)  # will cause problems if turned off
parser.add_argument("--convsylv_channels", type=list, default=[128])

# how many blocks of actnorm->sylv->flip in every stage of flow before split
parser.add_argument("--convsylv_flows_per_stage", type=list, default=[1])

# boolean value per stage, split latents before flow or not
parser.add_argument("--convsylv_splitfirst", type=list, default=[1, 1, 0, 0, 0, 0])

# 0 means no limit, otherwise this gives the maximum number of stages (splits) in the flow
parser.add_argument("--convsylv_stage_limit", type=list, default=[3])

# Attention parameters
parser.add_argument("--use_depthwise_attention", action="store_true", default=False)
parser.add_argument("--query_width", type=int, default=20)
parser.add_argument("--num_queries", type=int, default=1)
# key width matches query width
# value width is stage_width

# Training parameters
# (batch_size is divided by div and gradients accumulate to batch_size)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--batch_size_div", type=int, default=1)

# cr-vae divides training_batch in half, and concatenates another transformed version of the batch
parser.add_argument("--cr_vae", action="store_true", default=False)
parser.add_argument("--cr_lambda", type=float, default=0.1)
parser.add_argument("--cr_duplicate_noise", action="store_false", default=True)


parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--early_stopping", type=int, default=3000)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--learning_rate_end", type=float, default=1e-6)
parser.add_argument("--lr_warmup_epochs", type=int, default=1)
parser.add_argument(
    "--scheduler", type=str, default="cosd", choices=["none", "exp", "cosd", "cosdr"]
)
parser.add_argument(
    "--optimizer", type=str, default="adamax", choices=["adamax", "adamw", "sgd"]
)  # In keras3 for adamax, weight decay is decoupled from the momentum/lr updates just like AdamW!

parser.add_argument(
    "--weight_decay", type=float, default=1e-4
)  # weight decay is related to batch size, lr, #params, etc. the weights are SUMMED
parser.add_argument("--use_ema", action="store_false", default=True)

parser.add_argument("--gradient_smoothing", type=float, default=0.69314718056)  # Ln(2)
parser.add_argument("--gradient_clipnorm", type=float, default=0)  # 0 gets set to None
parser.add_argument(
    "--gradient_skipnorm", type=float, default=300
)  # Threshold for max(grad_norm) skipping batch update, starts after epoch 2

parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument(
    "--beta_warmup_epochs", type=int, default=0
)  # total warmup period, but if cyclic multiplied with number_cycles
parser.add_argument("--cyclic_beta", action="store_true", default=False)
parser.add_argument("--number_cycles", type=int, default=3)

parser.add_argument(
    "--config", type=str, default="None"
)  # Name of config file in cfg folder, overrides all parameters!
args = parser.parse_args()

# First check if config file is defined to overwrite args
import os

if args.config != "None":
    with open(os.path.join("cfg", args.config + ".yaml"), "r") as f:
        config = yaml.safe_load(f)
        args = argparse.Namespace(**config)

os.environ["KERAS_BACKEND"] = args.backend
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import tensorflow as tf

# Add all folders to the current path, the imports should happen after keras backend is set
import sys

sys.path.append("model")
sys.path.append("utils")
sys.path.append("training")
sys.path.append("data")

from config_params import Parameters
from dataloader import get_data
from plot_history import plt_train_history
from plot_hierarchy import plt_hierarchy
from vae_trainer import VAETrainer


def run(arguments):
    params = Parameters(arguments)

    # Multi-GPU sometimes leads to NaN loss (BUG)
    strategy = (
        tf.distribute.MirroredStrategy()
        if len(params.gpu.split(",")) > 1
        else tf.distribute.get_strategy()
    )
    print("Number of devices: ", strategy.num_replicas_in_sync)
    assert params.batch_size % strategy.num_replicas_in_sync == 0, (
        "Batch size must be divisible by number of devices"
    )

    # Get dataloaders
    train_set, val_set, _ = get_data(params.dataset, params.batch_size, params.cr_vae)

    with strategy.scope():
        opt = params.get_optimizer()
        trainer = VAETrainer(params)
        trainer.build()
        trainer.compile(optimizer=opt, jit_compile=params.jit)

    # Save the configuration and print to terminal
    if params.save:
        std_out = sys.stdout
        with open(
            os.path.join(params.save_path, "training_params.txt"), "w", encoding="utf-8"
        ) as f:
            sys.stdout = f
            trainer.summary()
            trainer.model.print_model()
        sys.stdout = std_out

    trainer.summary()
    trainer.model.print_model()

    # Make sure that the correct device is used, and not altered by some code
    assert os.environ["CUDA_VISIBLE_DEVICES"] == args.gpu, "GPU not set correctly"
    print("-------------------")
    print("Trainer built: ", trainer.built)
    print("VAE built:     ", trainer.model.built)
    print("-------------------")

    # Train the model
    train_history = trainer.fit(
        train_set,
        epochs=params.epochs,
        validation_data=val_set,
        callbacks=trainer.get_callbacks(),
        steps_per_epoch=params.step_per_epoch,
    )

    # Plot the training history
    if params.save:
        plt_train_history(train_history, params.dataset, params.save_path)
        plt_hierarchy(trainer.model, val_set, params.save_path)


if __name__ == "__main__":
    run(args)
