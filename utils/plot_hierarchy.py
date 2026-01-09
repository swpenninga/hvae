import os
import matplotlib.pyplot as plt
from keras import ops
import tensorflow as tf


# This function is called at the end of training to plot hierarchy timings and stage outputs
@tf.function
def get_enc_timings(model, image):
    time_stamps = [tf.timestamp()]
    x = model.encoder.first_conv(image)
    for enc_stage in model.encoder.stages.layers:
        x, _ = enc_stage(x)
        time_stamps.append(tf.timestamp())

    diff = []
    for i in range(len(time_stamps) - 1):
        diff += [time_stamps[i + 1] - time_stamps[i]]
    return diff


def run_enc_timings(model, image, num_runs=10):
    time_stamps = []
    for _ in range(num_runs + 3):
        time_stamps.append(get_enc_timings(model, image))
    time_stamps = tf.stack(time_stamps[3:], axis=0)
    mean_time = tf.reduce_mean(time_stamps, axis=0)
    return mean_time


@tf.function
def get_dec_timings(model, activations):
    num_blocks_per_stage = []
    for dec_stage in model.decoder.stages.layers:
        num_blocks_per_stage.append(dec_stage.num_blocks)

    # conditional timings per layer
    time_stamps = [tf.timestamp()]
    if model.decoder.init_zeros:
        x = ops.zeros_like(activations[-1])
    else:
        b = ops.shape(activations[-1])[0]
        x = ops.repeat(model.decoder.init_bias, b, axis=0)

    for dec_stage, act in zip(model.decoder.stages.layers, reversed(activations)):
        x, _, _ = dec_stage(x, act, training=False)
        time_stamps.append(tf.timestamp())

    diff_cond = []
    for i in range(len(time_stamps) - 1):
        diff_cond += [(time_stamps[i + 1] - time_stamps[i]) / num_blocks_per_stage[i]]

    # unconditional timings per layer
    time_stamps = [tf.timestamp()]
    if model.decoder.init_zeros:
        x = ops.zeros(
            (
                1,
                model.decoder.init_shape[0],
                model.decoder.init_shape[1],
                model.decoder.init_shape[2],
            )
        )
    else:
        x = ops.repeat(model.decoder.init_bias, 1, axis=0)

    for dec_stage in model.decoder.stages.layers:
        x, z = dec_stage.call_uncond(x, 1)
        time_stamps.append(tf.timestamp())
    diff_uncond = []
    for i in range(len(time_stamps) - 1):
        diff_uncond += [(time_stamps[i + 1] - time_stamps[i]) / num_blocks_per_stage[i]]
    return diff_cond, diff_uncond, num_blocks_per_stage


def run_dec_timings(model, image, num_runs=10):
    activations = model.encoder(image)
    time_stamps_cond = []
    time_stamps_uncond = []
    for _ in range(num_runs + 3):
        diff_cond, diff_uncond, num_blocks_per_stage = get_dec_timings(
            model, activations
        )
        time_stamps_cond.append(diff_cond)
        time_stamps_uncond.append(diff_uncond)

    time_stamps_cond = tf.stack(time_stamps_cond[3:], axis=0)
    time_stamps_uncond = tf.stack(time_stamps_uncond[3:], axis=0)
    mean_time_cond = tf.reduce_mean(time_stamps_cond, axis=0)
    mean_time_uncond = tf.reduce_mean(time_stamps_uncond, axis=0)

    mean_time_cond = tf.reverse(mean_time_cond, axis=[0])
    mean_time_uncond = tf.reverse(mean_time_uncond, axis=[0])

    return mean_time_cond, mean_time_uncond, num_blocks_per_stage


def run_timings(model, image, num_runs=20, save_path=None):
    timing_folder = os.path.join(save_path, "timings")
    os.makedirs(timing_folder, exist_ok=True)

    # 1 number per stage
    enc_time = run_enc_timings(model, image, num_runs)
    # 2 numbers per stage, conditional and unconditional
    # this is the number per block that is repeated num_block times per stage
    dec_time_cond, dec_time_uncond, num_blocks = run_dec_timings(model, image, num_runs)

    # Encoder
    enc_stretched = []
    cumulated = 0
    enc_cumulative = []
    num_stages = len(num_blocks)
    for i in range(num_stages):
        cumulated += enc_time[i]
        for _ in range(num_blocks[i]):
            enc_stretched.append(enc_time[i])
            enc_cumulative.append(cumulated)
    enc_stretched = tf.stack(enc_stretched)
    enc_cumulative = tf.stack(enc_cumulative)

    plt.figure(figsize=(8, 4))
    plt.suptitle("Encoder pass")
    plt.subplot(1, 2, 1)
    plt.bar(
        range(1, len(enc_stretched) + 1),
        enc_stretched,
        color="blue",
        label="Layer time",
    )
    plt.legend()
    plt.xlabel("Depth (#layers)")
    plt.ylabel("Time (s)")
    plt.subplot(1, 2, 2)
    plt.bar(
        range(1, len(enc_cumulative) + 1),
        enc_cumulative,
        color="orange",
        label="Cumulative time",
    )
    plt.legend()
    plt.xlabel("Depth (#layers)")
    plt.ylabel("Time (s)")
    plt.savefig(
        os.path.join(timing_folder, "enc_pass.pdf"),
        bbox_inches="tight",
    )

    # Decoder conditional
    dec_stretched = []
    cumulated = 0
    dec_cumulative = []
    for i in range(num_stages):
        for _ in range(num_blocks[i]):
            cumulated += dec_time_cond[i]
            dec_stretched.append(dec_time_cond[i])
            dec_cumulative.append(cumulated)
    dec_stretched = tf.stack(dec_stretched)
    dec_cumulative = tf.stack(dec_cumulative)

    plt.figure(figsize=(8, 4))
    plt.suptitle("Decoder conditional pass")
    plt.subplot(1, 2, 1)
    plt.bar(
        range(1, len(dec_stretched) + 1),
        dec_stretched,
        color="blue",
        label="Layer time",
    )
    plt.legend()
    plt.xlabel("Depth (#layers)")
    plt.ylabel("Time (s)")
    plt.subplot(1, 2, 2)
    plt.bar(
        range(1, len(dec_cumulative) + 1),
        dec_cumulative,
        color="orange",
        label="Cumulative time",
    )
    plt.legend()
    plt.xlabel("Depth (#layers)")
    plt.ylabel("Time (s)")
    plt.savefig(
        os.path.join(timing_folder, "dec_conditional_pass.pdf"),
        bbox_inches="tight",
    )

    # Decoder unconditional
    dec_stretched = []
    cumulated = 0
    dec_cumulative = []
    for i in range(num_stages):
        for _ in range(num_blocks[i]):
            cumulated += dec_time_uncond[i]
            dec_stretched.append(dec_time_uncond[i])
            dec_cumulative.append(cumulated)
    dec_stretched = tf.stack(dec_stretched)
    dec_cumulative = tf.stack(dec_cumulative)

    plt.figure(figsize=(8, 4))
    plt.suptitle("Decoder unconditional pass")
    plt.subplot(1, 2, 1)
    plt.bar(
        range(1, len(dec_stretched) + 1),
        dec_stretched,
        color="blue",
        label="Layer time",
    )
    plt.legend()
    plt.xlabel("Depth (#layers)")
    plt.ylabel("Time (s)")
    plt.subplot(1, 2, 2)
    plt.bar(
        range(1, len(dec_cumulative) + 1),
        dec_cumulative,
        color="orange",
        label="Cumulative time",
    )
    plt.legend()
    plt.xlabel("Depth (#layers)")
    plt.ylabel("Time (s)")
    plt.savefig(
        os.path.join(timing_folder, "dec_unconditional_pass.pdf"),
        bbox_inches="tight",
    )

    # Total cumulative conditional time
    tot_cumulative = []
    cumulated = 0
    for i in range(num_stages):
        cumulated += enc_time[i]
        for _ in range(num_blocks[i]):
            cumulated += dec_time_cond[i]
            tot_cumulative.append(cumulated)
    tot_cumulative = tf.stack(tot_cumulative)
    plt.figure(figsize=(4, 4))
    plt.suptitle("Total cumulative conditional time")
    plt.bar(
        range(1, len(tot_cumulative) + 1),
        tot_cumulative,
        color="blue",
        label="Layer time",
    )
    plt.legend()
    plt.xlabel("Depth (#layers)")
    plt.ylabel("Time (s)")
    plt.savefig(
        os.path.join(timing_folder, "total_conditional_pass.pdf"),
        bbox_inches="tight",
    )
    return


def plot_stages(model, image, save_path, grid=6):
    stages_folder = os.path.join(save_path, "stages")
    os.makedirs(stages_folder, exist_ok=True)

    plt.figure(figsize=(8, 8), facecolor="black")
    plt.suptitle("Input image")
    if model.params.dataset == "mnist":
        plt.imshow(image[0, 2:30, 2:30, 0], cmap="gray")
    if model.params.dataset == "echonet":
        plt.imshow((image[0, 8:120, 8:120, 0] + 1) / 2, cmap="gray")
    else:
        plt.imshow((image[0] + 1) / 2)
    plt.axis("off")
    plt.savefig(
        os.path.join(stages_folder, "input_image.pdf"),
        bbox_inches="tight",
    )
    plt.close()

    image = ops.repeat(image, grid**2, axis=0)
    activations = model.encoder(image)
    num_blocks = model.params.model_depth

    entropy_img_stack = []

    for j in range(num_blocks + 1):
        # initialize image
        if model.decoder.init_zeros:
            x = ops.zeros_like(activations[-1])
        else:
            b = ops.shape(activations[-1])[0]
            x = ops.repeat(model.decoder.init_bias, b, axis=0)

        curr_block = 0
        z_out = ops.zeros_like(activations[0])
        # perform either inference or unconditional sampling
        for _, (dec_stage, act) in enumerate(
            zip(model.decoder.stages.layers, reversed(activations))
        ):
            for dec_block in dec_stage.blocks.layers:
                if curr_block < j:
                    x, z, _ = dec_block(x, act, training=False)
                else:
                    x, z = dec_block.call_uncond(x, 1)
                curr_block += 1
                z_out += z
            x = dec_stage.pool(x)

        z_out /= ops.sqrt(model.decoder.model_depth)
        px_z = model.decoder.activation(model.decoder.z_to_features(z_out))
        for out_block in model.decoder.output_blocks.layers:
            px_z = out_block(px_z)
        x = model.decoder.last_conv(px_z)

        if model.params.dataset == "mnist":
            x = x[:, 2:30, 2:30]
            x = ops.sigmoid(x)
        elif model.params.dataset == "echonet":
            x = model.sample_from_mol(x)[:, 8:120, 8:120]
            x = (x + 1) / 2
        else:
            x = model.sample_from_mol(x)
            x = (x + 1) / 2

        plt.figure(figsize=(16, 16))
        plt.suptitle(f"Conditioned until layer {j}")
        for p in range(grid**2):
            plt.subplot(grid, grid, p + 1, facecolor="black")
            if model.params.dataset == "mnist" or model.params.dataset == "echonet":
                plt.imshow(x[p, :, :, 0], vmin=0, vmax=1, cmap="gray")
            else:
                plt.imshow(x[p], vmin=0, vmax=1)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(stages_folder, f"cond_until_layer_{j}.pdf"),
            bbox_inches="tight",
        )
        plt.close()

        entropy_img = ops.var(x, axis=0)
        if model.params.dataset not in ["mnist", "echonet"]:
            entropy_img = (
                entropy_img[:, :, 0] ** 2
                + entropy_img[:, :, 1] ** 2
                + entropy_img[:, :, 2] ** 2
            ) ** (1 / 3)
        entropy_img_stack.append(entropy_img)

    entropy_img_stack = ops.stack(entropy_img_stack, axis=0)
    entropy_img_stack /= ops.max(entropy_img_stack)

    num_images = len(entropy_img_stack)
    num_columns = 8
    num_rows = num_images // num_columns + (num_images % num_columns > 0)

    plt.figure(figsize=(32, 32), facecolor="black")
    plt.suptitle("Normalized Entropy conditioned until each layer")
    for j in range(num_images):
        plt.subplot(num_rows, num_columns, j + 1, facecolor="black")
        plt.imshow(entropy_img_stack[j], vmin=0, vmax=1, cmap="inferno")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(stages_folder, "entropy_maps.pdf"),
        bbox_inches="tight",
    )
    plt.close()

    return


def plt_hierarchy(
    model,
    data_set,
    save_path,
):
    # grab random GT image
    im = next(iter(data_set))
    im = im[0:1]

    # calculate and plot timings
    run_timings(model, im, num_runs=25, save_path=save_path)
    plot_stages(model, im, save_path)

    return
