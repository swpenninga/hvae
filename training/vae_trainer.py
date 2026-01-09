import tensorflow as tf
from callbacks import (
    BetaScheduler,
    SavePlots,
    KLStagePrint,
    CyclicBetaScheduler,
    GradSkipNorm,
)
from keras import Model, callbacks, metrics, ops, Variable
from metrics import GradientNorms
from vae import VAE


class VAETrainer(Model):
    # Trainer class for the VAE model
    def __init__(self, params, *args, **kwargs):
        super().__init__(**kwargs)

        self.params = params
        self.lr = params.learning_rate
        self.batch_size = params.batch_size

        self.model = VAE(params)
        self.bits_dim_scale = ops.multiply(self.model.ndims, ops.log(2))

        self.elbo_tracker = metrics.Mean(name="loss")
        self.recon_tracker = metrics.Mean(name="recon")
        self.kl_tracker = metrics.Mean(name="kl")
        self.grad_tracker = GradientNorms(name="grad_norm")
        self.skip_tracker = metrics.Sum(name="skips")

        # grad_skipnorm gets set by a callback after n epochs
        self.grad_skipnorm = Variable(1e9, trainable=False, dtype="float32")

        self.strategy = tf.distribute.get_strategy()

    def build(self):
        shape_tuple = (
            None,  # batch
            self.params.enc_input_size[0],  # height
            self.params.enc_input_size[0],  # width
            self.params.data_width,  # channels
        )
        super().build(shape_tuple)
        self.model.build()

    def skip(self):
        self.skip_tracker.update_state(1)
        return 1

    def apply(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return 0

    def call(self, x, training):
        px_z, _, kl = self.model.call(x, training=training)
        elbo, recon, kl = self.model.get_elbo(x, px_z, kl, training=training)
        return elbo, recon, kl

    def train_step(self, data):
        with tf.GradientTape() as tape:
            elbo, recon, kl = self.call(data, training=True)

        gradients = tape.gradient(elbo, self.model.trainable_variables)

        # Check grads
        grad_norm = tf.linalg.global_norm([g for g in gradients if g is not None])
        _ = ops.cond(
            ops.logical_and(
                grad_norm < self.grad_skipnorm, tf.math.is_finite(grad_norm)
            ),
            true_fn=lambda: self.apply(gradients),
            false_fn=lambda: self.skip(),
        )

        self.elbo_tracker.update_state(elbo)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        self.grad_tracker.update_state(grad_norm)

        # Keras prints dictionary in alphabetical order regardless
        # of the order proposed here
        return {
            "loss": self.elbo_tracker.result(),
            "recon": self.recon_tracker.result(),
            "kl": self.kl_tracker.result(),
            "grad_norm": self.grad_tracker.result(),
            "skips": self.skip_tracker.result(),
            "curr_lr": self.optimizer.learning_rate,
            "beta": self.model.beta,
        }

    def test_step(self, data):
        _, recon, kl = self.call(data, training=False)
        elbo = recon + self.model.beta_max * kl

        self.elbo_tracker.update_state(elbo)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)

        return {
            "loss": self.elbo_tracker.result(),
            "recon": self.recon_tracker.result(),
            "kl": self.kl_tracker.result(),
        }

    @property
    def metrics(self):
        return [self.elbo_tracker, self.recon_tracker, self.kl_tracker]

    def get_callbacks(self):
        callbacks_set = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=self.model.params.early_stopping,
                restore_best_weights=True,
                start_from_epoch=self.model.beta_warmup_epochs,
            ),
            callbacks.TerminateOnNaN(),
        ]

        if self.grad_skipnorm > 0:
            callbacks_set += [GradSkipNorm(self.model.params.gradient_skipnorm)]

        if self.model.params.flow_type == "none":
            callbacks_set += [
                KLStagePrint(
                    self.model.params.dataset,
                    self.model.params.save,
                    self.model.params.save_path,
                )
            ]

        if self.model.beta_warmup_epochs > 0:
            if self.model.params.cyclic_beta:
                callbacks_set += [
                    CyclicBetaScheduler(
                        self.model.params.num_cycles,
                        self.model.params.beta_warmup_epochs,
                    )
                ]
            else:
                callbacks_set += [BetaScheduler()]

        if self.model.params.save:
            callbacks_set += [
                SavePlots(self.model.params.dataset),
                callbacks.ModelCheckpoint(
                    filepath=self.model.params.save_path + "/model.weights.h5",
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True,
                    save_weights_only=True,
                    save_freq="epoch",
                ),
            ]
        return callbacks_set
