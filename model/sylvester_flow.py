import numpy as np
import keras
from keras import layers, ops, random

"""
All of this code is adapted from the original Sylvester normalizing flow implementation in PyTorch.
https://github.com/riannevdberg/sylvester-flows/tree/master

It works with Jax, tensorflow and torch and it jit_compiles

The triangular and householder versions of the sylvester flow scale with (2D^2 + D) where D is zdim.
This does not scale with a HVAE (VDVAE) as the latents >> image_dim.
We opt for the orthogonal sylvester flow as it scales with (MD + 2M^2 + M) where we can choose M ourselves as num_ortho_vecs and keep the model relatively small.
"""


class OrthogonalSylvesterStack(layers.Layer):
    def __init__(
        self,
        num_flows,
        num_ortho_vecs,
        z_ch,
        build_shape,
        model_depth=1,
    ):
        super(OrthogonalSylvesterStack, self).__init__()

        self.num_flows = num_flows
        self.num_ortho_vecs = num_ortho_vecs
        self.build_shape = build_shape
        self.model_depth = model_depth

        # Note that in the rest of the model, zdim is the number of channels of z.
        # Here, zdim is the number of elements in z (we flatten height and width and make a new variable z_ch)
        self.z_ch = z_ch
        self.zdim = self.z_ch * build_shape[1] * build_shape[2]

        assert (self.num_ortho_vecs <= self.zdim) and (self.num_ortho_vecs > 0)

        if keras.dtype_policies.DTypePolicy().name == "mixed_bfloat16":
            self.dt = "bfloat16"
        else:
            self.dt = "float32"

        # Untrainable helper tensors
        self.identity = ops.expand_dims(
            ops.eye(self.num_ortho_vecs, self.num_ortho_vecs), axis=0
        )

        self.identity = ops.cast(self.identity, self.dt)

        triu_mask = ops.triu(ops.ones([self.num_ortho_vecs, self.num_ortho_vecs]), k=1)
        self.triu_mask = ops.expand_dims(ops.expand_dims(triu_mask, axis=0), axis=3)
        self.flatten = layers.Flatten()
        self.diag_activation = ops.tanh
        self.i = ops.expand_dims(ops.eye(self.num_ortho_vecs), axis=[0, -1])

        # Trainable layers

        self.amor_d = layers.Dense(
            self.num_flows * self.num_ortho_vecs * self.num_ortho_vecs
        )
        self.amor_diag1 = layers.Dense(
            self.num_flows * self.num_ortho_vecs, activation=self.diag_activation
        )
        self.amor_diag2 = layers.Dense(
            self.num_flows * self.num_ortho_vecs, activation=self.diag_activation
        )
        self.amor_q = layers.Dense(self.num_flows * self.num_ortho_vecs * self.zdim)
        self.amor_b = layers.Dense(self.num_flows * self.num_ortho_vecs)

        self.flows = [Sylvester(num_ortho_vecs) for _ in range(num_flows)]

        # Iteration parameters
        if self.num_ortho_vecs == self.zdim:
            self.cond = 1.0e-5
        else:
            self.cond = 1.0e-6

        self.steps = 100

    def orthogonal_loop_body(self, amat):
        tmp = ops.matmul(ops.transpose(amat, [0, 2, 1]), amat)
        tmp = ops.subtract(self.identity, tmp)
        tmp = ops.add(self.identity, 0.5 * tmp)
        amat = ops.matmul(amat, tmp)
        return amat

    def orthogonal_loop_cond(self, amat):
        test = ops.subtract(
            ops.matmul(ops.transpose(amat, [0, 2, 1]), amat), self.identity
        )
        norms2 = ops.sum(ops.norm(test, ord=2, axis=2) ** 2, axis=1)
        norms = ops.sqrt(norms2)
        max_norm = ops.max(norms)
        return ops.greater(max_norm, self.cond)

    def batch_construct_orthogonal(self, q):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size, num_flows * zdim * num_ortho_vecs)
        :return: batches of orthogonalized matrices, shape: (num_flows, batch_size, zdim, num_ortho_vecs)
        """

        q = ops.reshape(q, [-1, self.zdim * self.num_ortho_vecs])
        norm = ops.norm(q, ord=2, axis=1, keepdims=True)
        amat = ops.divide(q, norm)
        dim0 = ops.shape(amat)[0]
        amat = ops.reshape(amat, [dim0, self.zdim, self.num_ortho_vecs])

        amat = ops.while_loop(
            self.orthogonal_loop_cond,
            self.orthogonal_loop_body,
            amat,
            maximum_iterations=self.steps,
        )

        amat = ops.reshape(amat, [-1, self.num_flows, self.zdim, self.num_ortho_vecs])
        amat = ops.transpose(amat, [1, 0, 2, 3])
        return amat

    def encode(self, h):
        # Incoming batch of [B, input_size, input_size, flow_in_ch] shape
        B = ops.shape(h)[0]

        h = self.flatten(h)  # [B, input_size * input_size * flow_in_ch]

        # Amortized r1, r2, b for all flows
        full_d = self.amor_d(h)  # [B, num_flows * num_ortho_vecs * num_ortho_vecs]
        diag1 = self.amor_diag1(h)  # [B, num_flows * num_ortho_vecs]
        diag2 = self.amor_diag2(h)  # [B, num_flows * num_ortho_vecs]
        q = self.amor_q(h)  # [B, num_flows * zdim * num_ortho_vecs]
        b = self.amor_b(h)  # [B, num_flows * num_ortho_vecs]

        full_d = ops.reshape(
            full_d, [B, self.num_ortho_vecs, self.num_ortho_vecs, self.num_flows]
        )  # <- shape
        diag_1 = ops.reshape(
            diag1, [B, self.num_ortho_vecs, self.num_flows]
        )  # <- shape
        diag_2 = ops.reshape(
            diag2, [B, self.num_ortho_vecs, self.num_flows]
        )  # <- shape

        r1 = ops.multiply(
            full_d, self.triu_mask
        )  # [B, num_ortho_vecs, num_ortho_vecs, num_flows] * [1, num_ortho_vecs, num_ortho_vecs, 1]
        r2 = ops.multiply(
            ops.transpose(full_d, axes=(0, 2, 1, 3)), self.triu_mask
        )  # (...) * [1, num_ortho_vecs, num_ortho_vecs, 1]

        # Tensorflow does not like item-assignment :(
        diag_1 = ops.expand_dims(diag_1, axis=2)  # [B, num_ortho_vecs, 1, num_flows]
        diag_2 = ops.expand_dims(diag_2, axis=2)  # [B, num_ortho_vecs, 1, num_flows]
        r1 += ops.multiply(
            self.i, diag_1
        )  # [1, num_ortho_vecs, num_ortho_vecs, 1] * [B, num_ortho_vecs, 1, num_flows] = [B, num_ortho_vecs, num_ortho_vecs, num_flows]
        r2 += ops.multiply(
            self.i, diag_2
        )  # [1, num_ortho_vecs, num_ortho_vecs, 1] * [B, num_ortho_vecs, 1, num_flows] = [B, num_ortho_vecs, num_ortho_vecs, num_flows]

        b = ops.reshape(
            b, [B, 1, self.num_ortho_vecs, self.num_flows]
        )  # [B, 1, num_ortho_vecs, num_flows]

        return r1, r2, q, b

    def call(self, z, h):
        # z is of shape [batch, num_particles, h, w, zdim]
        z_shape = ops.shape(z)
        z = self.flatten(z)

        r1, r2, q, b = self.encode(h)
        q_ortho = self.batch_construct_orthogonal(q)

        log_det_jacobian = 0.0
        for k in range(self.num_flows):
            z, ldj = self.flows[k](
                z, r1[:, :, :, k], r2[:, :, :, k], q_ortho[k, :, :, :], b[:, :, :, k]
            )
            log_det_jacobian += ldj

        z = ops.reshape(z, z_shape)
        return z, log_det_jacobian

    def build(self):
        _ = self.call(
            z=random.normal(
                (
                    self.build_shape[0],
                    self.build_shape[1],
                    self.build_shape[2],
                    self.z_ch,
                )
            ),
            h=random.normal(self.build_shape),
        )
        # HERE WE RESCALE THE OUTPUT OF THE FLOW LAYERS WITH 1/SQRT(DEPTH)
        layers = [
            self.amor_d,
            self.amor_diag1,
            self.amor_diag2,
            self.amor_q,
            self.amor_b,
        ]
        for layer in layers:
            weights = layer.get_weights()
            weights[0] *= np.sqrt(1 / self.model_depth)
            layer.set_weights(weights)


class Sylvester(layers.Layer):
    def __init__(self, num_ortho_vecs):
        super(Sylvester, self).__init__()
        self.num_ortho_vecs = num_ortho_vecs
        self.h = ops.tanh
        self.triu_mask = ops.expand_dims(
            ops.triu(ops.ones([self.num_ortho_vecs, self.num_ortho_vecs]), k=1), axis=0
        )
        self.diag_idx = ops.arange(0, self.num_ortho_vecs, dtype="int64")

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def call(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        """
        All flow parameters are amortized. Conditions on diagonals of R1 and R2 for invertibility need to be satisfied
        outside of this function. Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        :param zk: shape: (batch_size, zdim)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param q_ortho: shape (batch_size, zdim , num_ortho_vecs)
        :param b: shape: (batch_size, 1, self.num_ortho_vecs) (????)
        :return: z, log_det_j
        """

        zk = ops.expand_dims(zk, axis=1)

        diag_r1 = ops.diagonal(r1, axis1=1, axis2=2)
        diag_r2 = ops.diagonal(r2, axis1=1, axis2=2)

        r1_hat = r1
        r2_hat = r2

        qr2 = ops.matmul(q_ortho, ops.transpose(r2_hat, [0, 2, 1]))
        qr1 = ops.matmul(q_ortho, r1_hat)

        r2qzb = ops.matmul(zk, qr2) + b

        z = ops.matmul(self.h(r2qzb), ops.transpose(qr1, [0, 2, 1])) + zk
        z = ops.squeeze(z, axis=1)

        diag_j = diag_r1 * diag_r2
        diag_j = ops.squeeze(self.der_tanh(r2qzb), axis=1) * diag_j
        diag_j += 1.0
        log_diag_j = ops.log(ops.absolute(diag_j))

        if sum_ldj:
            log_det_j = ops.sum(log_diag_j, axis=-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j


if __name__ == "__main__":
    zdim = 3
    flow_in_ch = 3
    input_size = 32
    num_flows = 5
    num_ortho_vecs = 4
    flow = OrthogonalSylvesterStack(
        num_flows=num_flows,
        num_ortho_vecs=num_ortho_vecs,
        z_ch=zdim,
        build_shape=(2, input_size, input_size, flow_in_ch),
    )
    flow.build()
    print(flow)
