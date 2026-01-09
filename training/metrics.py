from keras import metrics, ops


class GradientNorms(metrics.Metric):
    # Tracker for gradient norms
    def __init__(self, name="grad_norm", **kwargs):
        super(GradientNorms, self).__init__(name=name, **kwargs)
        self.grads = self.add_variable(
            shape=(),
            initializer="zeros",
            name="grads",
        )

    def update_state(self, grads, sample_weight=None):
        grads = ops.cast(grads, dtype="float32")
        self.grads.assign(grads)

    def result(self):
        return self.grads
