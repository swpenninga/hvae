import os
import pickle
import sys

sys.path.append("model")
sys.path.append("utils")
sys.path.append("training")
sys.path.append("data")

from model.vae import VAE
from config_params import Parameters


def loadmodel(path):
    with open(os.path.join(path, "args.pkl"), "rb") as f:
        args = pickle.load(f)
    args.save = False  # Do not save the model again
    vae = VAE(Parameters(args))
    vae.build()
    vae.load_weights(os.path.join(path, "model.weights.h5"))
    return vae
