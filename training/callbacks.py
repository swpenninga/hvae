import os

import matplotlib.pyplot as plt
from dataloader import get_data
from keras import callbacks, ops, random


class GradSkipNorm(callbacks.Callback):
    # At epoch 3, sets the grad_skipnorm threshold to the given value
    def __init__(self, threshold):
        super(GradSkipNorm, self).__init__()
        self.threshold = threshold

    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 2:
            self.model.grad_skipnorm.assign(self.threshold)


class KLStagePrint(callbacks.Callback):
    # Prints the KL and AU per stage of the model at every epoch, on a small batch of the validation set
    def __init__(self, dataset, save, savepath):
        _, val_set, _ = get_data(dataset, 50)
        for _ in random.randint(shape=[1], minval=1, maxval=10):
            im = next(iter(val_set))
        self.in_images = im
        self.au_delta = 0.01
        self.save = save
        if save:
            self.save_path = os.path.join(savepath, "kl_log.txt")
            self.au_path = os.path.join(savepath, "au_log.txt")

    def on_epoch_end(self, epoch, logs=None):
        # retrieve KL for every stage and print it
        _, _, kl = self.model.model.call(self.in_images, training=False)
        string = "KL - "
        au_string = "AU - "
        au_total = 0
        dims_total = 0
        for i, kls_stage in enumerate(kl):
            if kls_stage:
                kl_at_stage = 0.0
                AU_at_stage = 0
                for kls in kls_stage:
                    kl_at_stage += ops.sum(kls, axis=[1, 2, 3])
                    dimensions = ops.cast(ops.prod(ops.shape(kls)[1:]), "float32")
                    dims_total += dimensions
                    curr_au = ops.mean(ops.sum(kls > self.au_delta, axis=[1, 2, 3]))
                    au_total += curr_au
                    AU_at_stage += curr_au / dimensions
                kl_at_stage = ops.mean(kl_at_stage)
                AU_at_stage /= len(kls_stage)
                if self.model.model.params.channels_out > 1:
                    kl_at_stage /= self.model.model.ndims * ops.log(2)
                string += f"stage {2**i}x{2**i}: {kl_at_stage.numpy():.3f} - "
                au_string += f"stage {2**i}x{2**i}: {AU_at_stage.numpy():.1%} - "
        au_string += f"total: {au_total.numpy() / dims_total:.1%} - "
        print("\n")
        print(string)
        print(au_string)
        print("\n")
        if self.save:
            with open(self.save_path, "a") as f:
                f.write(f"epoch {epoch + 1} - " + string + "\n")
            with open(self.au_path, "a") as f:
                f.write(f"epoch {epoch + 1} - " + au_string + "\n")


class CyclicBetaScheduler(callbacks.Callback):
    # Linearly schedules beta, in multiple iterations of warmup epochs
    def __init__(self, number_cycles, beta_warmup_epochs):
        self.batch_count = 0
        self.number_cycles = number_cycles
        self.beta_warmup_epochs = beta_warmup_epochs

    def on_train_batch_begin(self, batch, logs=None):
        # half of the warmup epochs are annealing, the other half refinement, we repeat this for number_cycles then fix beta at 1
        if (
            self.batch_count
            < self.beta_warmup_epochs
            * self.number_cycles
            * self.model.model.params.step_per_epoch
        ):
            step = self.batch_count % (
                self.beta_warmup_epochs * self.model.model.params.step_per_epoch
            )
            if (
                step
                > self.beta_warmup_epochs * self.model.model.params.step_per_epoch / 2
            ):
                self.model.model.beta.assign(self.model.model.beta_max)
            else:
                beta_frac = step / (
                    self.beta_warmup_epochs * self.model.model.params.step_per_epoch / 2
                )
                self.model.model.beta.assign(beta_frac * self.model.model.beta_max)
        else:
            self.model.model.beta.assign(self.model.model.beta_max)
        self.batch_count += 1


class BetaScheduler(callbacks.Callback):
    # Linear beta scheduler over beta_warmup_epochs
    def __init__(self):
        self.batch_count = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.model.model.beta.assign(
            min(
                1,
                self.batch_count
                / (
                    self.model.model.beta_warmup_epochs
                    * self.model.model.params.step_per_epoch
                ),
            )
            * self.model.model.beta_max
        )
        self.batch_count += 1


class SavePlots(callbacks.Callback):
    # Saves a couple of reconstructions every epoch and some unconditional samples at different temperatures
    def __init__(self, dataset):
        _, val_set, _ = get_data(dataset, 100)
        for _ in random.randint(shape=[1], minval=1, maxval=10):
            im = next(iter(val_set))
        self.in_images = ops.cast(im[:8], "float32")
        self.t_set = [0.9, 1.0, 1.1]

    def on_epoch_end(self, epoch, logs=None):
        # Reconstructions
        px_z, _, _ = self.model.model.call(self.in_images)
        if self.model.model.params.dataset == "mnist":
            plot_images = self.in_images[:, 2:30, 2:30]
            px_z = px_z[:, 2:30, 2:30]
            px_z = ops.sigmoid(px_z)
        else:
            plot_images = (self.in_images + 1) / 2  # [0, 1]
            px_z = self.model.model.sample_from_mol(px_z)
            px_z = (px_z + 1) / 2  # [0, 1]

        # Unconditional generation
        pz0 = self.model.model.decoder.call_uncond(num_images=16, t=self.t_set[0])
        pz1 = self.model.model.decoder.call_uncond(num_images=16, t=self.t_set[1])
        pz2 = self.model.model.decoder.call_uncond(num_images=16, t=self.t_set[2])
        pz = ops.concatenate([pz0, pz1, pz2])

        if self.model.model.params.dataset == "mnist":
            pz = pz[:, 2:30, 2:30]
            pz = ops.sigmoid(pz)
        else:
            pz = self.model.model.sample_from_mol(pz)
            pz = (pz + 1) / 2  # [0, 1]

        px_z = ops.cast(px_z, "float32")
        pz = ops.cast(pz, "float32")

        # Plots
        statistics = self.model.metrics
        elbo = statistics[0].result()
        recon = statistics[1].result()
        kl = statistics[2].result()

        plt.figure(figsize=(12, 3))
        plt.suptitle(
            f"elbo:{elbo.numpy():.3f}, recon:{recon.numpy():.3f}, kl:{kl.numpy():.3f}"
        )
        for i in range(16):
            plt.subplot(2, 8, i + 1)
            plt.axis("off")
            if i < 8:
                if self.model.model.params.data_width > 1:
                    plt.imshow(plot_images[i], vmin=0, vmax=1)
                else:
                    plt.imshow(plot_images[i], vmin=0, vmax=1, cmap="gray")
            else:
                if self.model.model.params.data_width > 1:
                    plt.imshow(px_z[i - 8], vmin=0, vmax=1)
                else:
                    plt.imshow(px_z[i - 8], vmin=0, vmax=1, cmap="gray")
        plt.savefig(
            os.path.join(
                self.model.model.params.save_path,
                "reconstructions",
                f"{epoch + 1}_elbo{elbo:.3f}_recon.pdf",
            )
        )
        plt.close()

        plt.figure(figsize=(12, 9))
        plt.suptitle(
            f"t={self.t_set[0]} [0,1], t={self.t_set[1]} [2,3], t={self.t_set[2]} [4,5]"
        )
        for i in range(16 * 3):
            plt.subplot(6, 8, i + 1)
            plt.axis("off")
            if self.model.model.params.data_width > 1:
                plt.imshow(pz[i], vmin=0, vmax=1)
            else:
                plt.imshow(pz[i], vmin=0, vmax=1, cmap="gray")

        plt.savefig(
            os.path.join(
                self.model.model.params.save_path,
                "reconstructions",
                f"{epoch + 1}_unconditional.pdf",
            )
        )
        plt.close()
