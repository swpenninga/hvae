import os

import matplotlib.pyplot as plt

# Plots the training history


def get_ylims(dataset_name):
    if dataset_name == "mnist":
        elbo_lims = (20, 120)
        recon_lims = (40, 80)
        kl_lims = (10, 50)

    elif dataset_name == "cifar10":
        elbo_lims = (2, 4)
        recon_lims = (1, 3)
        kl_lims = (0.4, 2)

    else:
        elbo_lims = (0, 7)
        recon_lims = (0, 5)
        kl_lims = (0, 3)

    return elbo_lims, recon_lims, kl_lims


def plt_train_history(history, dataset_name, path):
    elbo_lims, recon_lims, kl_lims = get_ylims(dataset_name)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(history.history["loss"], label="Elbo")
    plt.plot(history.history["val_loss"], label="val_Elbo")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Elbo")
    plt.ylim(elbo_lims)

    plt.subplot(2, 3, 2)
    plt.plot(history.history["recon"], label="Reconstruction")
    plt.plot(history.history["val_recon"], label="val_Reconstruction")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction")
    plt.ylim(recon_lims)

    plt.subplot(2, 3, 3)
    plt.plot(history.history["kl"], label="KL")
    plt.plot(history.history["val_kl"], label="val_KL")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("KL")
    plt.ylim(kl_lims)

    plt.subplot(2, 3, 4)
    plt.plot(history.history["beta"], label="Beta")
    plt.xlabel("Epoch")
    plt.ylabel("Beta")

    plt.subplot(2, 3, 5)
    plt.plot(history.history["curr_lr"], label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()

    plt.subplot(2, 3, 6)
    grad_norms = history.history["grad_norm"]
    plt.plot(grad_norms, label="Gradient Norms")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norms")
    plt.legend()

    plt.savefig(os.path.join(path, "train_history.pdf"))

    val_elbo = min(history.history["val_loss"])
    open(os.path.join(path, f"min_val_elbo{val_elbo:.3f}.txt"), "w").close()
