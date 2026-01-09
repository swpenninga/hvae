from utils.distributions import SoftPlus

from keras import layers, losses, ops


class BinaryCE(losses.Loss):
    def __init__(self, dataset="mnist"):
        super(BinaryCE, self).__init__()
        self.logits = True
        self.loss = losses.BinaryCrossentropy(
            from_logits=self.logits,
            reduction="none",
        )
        self.reduction = "none"

    def call(self, targets, logits):
        #   targets: B, H, W, C   [e.g. N,32,32,1]
        #   logits:  B, H, W, C   [e.g. N,32,32,1]

        # Crop mnist loss back to [28, 28]
        targets = targets[:, 2:30, 2:30, :]
        logits = logits[:, 2:30, 2:30, :]

        if not self.logits:
            logits = ops.sigmoid(logits)

        loss = self.loss(targets, logits)
        loss = ops.sum(loss, axis=[1, 2])
        return loss


class CE(losses.Loss):
    def __init__(self, num_bits, num_channels, shape):
        super(CE, self).__init__()
        self.reduction = "none"
        self.num_bits = num_bits
        self.num_channels = num_channels
        self.shape = shape
        self.reshape = layers.Reshape(
            [self.shape, self.shape, self.num_channels, 2**self.num_bits]
        )

    def call(self, targets, logits):
        logits = self.reshape(logits)
        loss = losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
        loss = ops.sum(loss, axis=[1, 2, 3])
        return loss


class DiscMixLogistic(losses.Loss):
    def __init__(self, num_bits, num_mixtures, num_channels):
        super(DiscMixLogistic, self).__init__()
        self.reduction = "none"
        self.num_bits = num_bits
        self.num_mixtures = num_mixtures
        self.num_channels = num_channels
        self.num_classes = 2.0**self.num_bits - 1.0
        self.min_pix_value = -1
        self.max_pix_value = 1
        self.min_mol_logscale = -250
        self.softplus = SoftPlus(0.69314718056)

        self.dt = "float32"

    def call(self, targets, logits):
        #   targets: B, H, W, C                 [e.g. N,32,32,3]
        #   logits:  B, H, W, M * (3 * C + 1)   [e.g. N,32,32,100] (10 * (3 * 3 + 1))

        B, H, W, C = (
            ops.shape(targets)[0],
            ops.shape(targets)[1],
            ops.shape(targets)[2],
            ops.shape(targets)[3],
        )

        assert C == 3 or C == 1  # Only RGB or grayscale images are supported

        targets = ops.cast(ops.expand_dims(targets, -1), dtype=self.dt)  # B, H, W, C, 1

        logit_probs = logits[:, :, :, : self.num_mixtures]  # B, H, W, M * 1
        l = logits[:, :, :, self.num_mixtures :]  # B, H, W, M*C*3
        l = layers.Reshape([H, W, self.num_channels, 3 * self.num_mixtures])(
            l
        )  # B, H, W, C, 3*M

        model_means = l[:, :, :, :, : self.num_mixtures]  # B, H, W, C, M

        log_scales = self.min_mol_logscale + self.softplus(
            l[:, :, :, :, self.num_mixtures : 2 * self.num_mixtures]
            - self.min_mol_logscale
        )

        model_coeffs = ops.tanh(
            l[:, :, :, :, 2 * self.num_mixtures : 3 * self.num_mixtures]
        )  # B, H, W, C, M

        # RGB AR
        if C == 3:
            mean1 = model_means[:, :, :, 0:1, :]  # B, H, W, 1, M
            mean2 = ops.add(
                model_means[:, :, :, 1:2, :],
                ops.multiply(model_coeffs[:, :, :, 0:1, :], targets[:, :, :, 0:1, :]),
            )  # B, H, W, 1, M
            mean3 = ops.add(
                ops.add(
                    model_means[:, :, :, 2:3, :],
                    ops.multiply(
                        model_coeffs[:, :, :, 1:2, :], targets[:, :, :, 0:1, :]
                    ),
                ),
                ops.multiply(model_coeffs[:, :, :, 2:3, :], targets[:, :, :, 1:2, :]),
            )  # B, H, W, 1, M
            means = ops.concatenate([mean1, mean2, mean3], axis=3)  # B, H, W, C, M
        else:
            means = model_means
        centered = targets - means  # B, H, W, C, M

        inv_stdv = ops.exp(-log_scales)  # B, H, W, C, M
        plus_in = ops.multiply(inv_stdv, (centered + 1.0 / self.num_classes))
        cdf_plus = ops.sigmoid(plus_in)
        min_in = ops.multiply(inv_stdv, (centered - 1.0 / self.num_classes))
        cdf_min = ops.sigmoid(min_in)

        log_cdf_plus = plus_in - ops.softplus(
            plus_in
        )  # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -ops.softplus(
            min_in
        )  # log probability for edge case of 255 (before scaling)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min  # B, H, W, C, M

        mid_in = ops.multiply(inv_stdv, centered)
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in this code)
        log_pdf_mid = mid_in - log_scales - 2.0 * ops.softplus(mid_in)

        # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        broadcast_targets = ops.broadcast_to(
            targets, shape=(B, H, W, C, self.num_mixtures)
        )

        # Explanation of the nested where statements:
        # First where statement, choose log probability of 0 if target == 0, else continue to:
        # Second where statement, choose log probability of 255 if target == 255 else continue to:
        # Third where statement, not a clue: https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py line 77
        log_probs = ops.where(
            broadcast_targets < (self.min_pix_value + 0.001),
            log_cdf_plus,
            ops.where(
                broadcast_targets > (self.max_pix_value - 0.001),
                log_one_minus_cdf_min,
                ops.where(
                    cdf_delta > 1e-5,
                    ops.log(ops.maximum(cdf_delta, 1e-12)),
                    ops.subtract(log_pdf_mid, ops.log(self.num_classes / 2)),
                ),
            ),
        )  # B, H, W, C, M

        log_probs = (
            ops.sum(log_probs, axis=3) if C == 3 else ops.squeeze(log_probs, axis=3)
        )  # B, H, W, M
        log_probs += ops.log_softmax(logit_probs, axis=3)  # B, H, W, M

        negative_log_probs = -ops.logsumexp(log_probs, axis=-1)  # B, H, W
        loss = ops.sum(negative_log_probs, axis=[1, 2])
        return loss
