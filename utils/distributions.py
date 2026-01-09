from keras import layers, ops


class SoftPlus(layers.Layer):
    # Custom softplus as in pytorch, with beta:
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html
    def __init__(self, gradient_smoothing, min=-10):
        super().__init__(trainable=False)
        assert gradient_smoothing > 0, "gradient_smoothing must be greater than 0"
        self.gradient_smoothing = gradient_smoothing
        self.min = min

    def call(self, std):
        std = (1 / self.gradient_smoothing) * ops.logaddexp(
            0, self.gradient_smoothing * std
        )
        return ops.maximum(std, self.min)


class gaussian_analytical_kl:
    # Kl divergence between two diagonal Gaussians
    def __init__(self):
        super().__init__()

    def call(self, q_mu, q_std, p_mu, p_std):
        term1 = (q_mu - p_mu) * ops.reciprocal(p_std)
        term2 = q_std * ops.reciprocal(p_std)
        loss = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - ops.log(term2)
        return loss

    def cr_loss(self, q_mu, q_std):
        # Consistency-regularization loss according to:
        # https://github.com/sinhasam/CRVAE/blob/main/CRVAE/losses.py
        mu, mu_tilde = ops.split(q_mu, 2, axis=0)
        std, std_tilde = ops.split(q_std, 2, axis=0)

        cr_loss = (
            ops.log(std / std_tilde)
            - 0.5
            + 0.5 * (std_tilde**2 + (mu_tilde - mu) ** 2) / std**2
        )
        return cr_loss


class flow_kl:
    # KL-divergence function that is compatible with flow layers, based on:
    # https://github.com/riannevdberg/sylvester-flows/blob/master/optimization/loss.py
    def __init__(self):
        super().__init__()

    def log_normal_diag(self, z, mu, std):
        var = ops.square(std)
        logvar = ops.log(var)
        log_norm = -0.5 * (
            ops.add(
                logvar,
                ops.multiply(ops.square(ops.subtract(z, mu)), ops.reciprocal(var)),
            )
        )
        return log_norm

    def log_normal_standard(self, z):
        log_norm = -0.5 * ops.square(z)
        return log_norm

    def call(self, qm, q_std, pm, p_std, z0, zk):
        log_q_z0 = self.log_normal_diag(z0, qm, q_std)
        log_p_zk = self.log_normal_diag(zk, pm, p_std)

        kl = log_q_z0 - log_p_zk
        return kl

    def cr_loss(self, zk):
        # CR-VAE loss for flow-based models
        # These two are not necessarily compatible since the flow uses transformed samples,
        # not distributions. An L2 distance might work, but this diverges from literature.
        # This returns 0 for stages with flow, and the CR loss is only computed on the non-flow stages.
        return ops.zeros(ops.shape(zk)[0] // 2)
