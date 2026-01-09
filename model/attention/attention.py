from keras import layers, ops


def attention_block(q, k, v):
    # Performs a single attention operation layerwise
    # inputs
    # q: (batch, height, width, c_attention)
    # k: (batch, height, width, c_attention, layers)
    # v: (batch, height, width, channels, layers)

    # q: (batch, height, width, 1, c_attention)
    q = ops.expand_dims(q, axis=-2)
    # qk: (batch, height, width, 1, layers)
    qk = ops.matmul(q, k) / ops.sqrt(ops.shape(q)[-1])

    # attn_scores: (batch, height, width, 1, layers)
    attn_scores = ops.softmax(qk, axis=-1)
    # attn_scores: (batch, height, width, layers, 1)
    attn_scores = ops.transpose(attn_scores, (0, 1, 2, 4, 3))

    # out: (batch, height, width, channels, 1)
    out = ops.matmul(v, attn_scores)

    return ops.squeeze(out, axis=-1)


def prepare_activations(activations):
    # input activations = list of act, containing [v_q, k_q]
    list_stages = []
    # start at highest resolution
    for i, act in enumerate(activations):
        v, k = act
        if v is not None:
            # Add layer dimension to each
            v, k = ops.expand_dims(v, axis=-1), ops.expand_dims(k, axis=-1)

            if i == 0:
                # 32x32 is first entry
                activation_set = [v, k]
            else:
                # 32x32 is pooled and 16x16 is concatenated behind
                v_stack = ops.average_pool(activation_set[0], pool_size=(2, 2, 1))
                k_stack = ops.average_pool(activation_set[1], pool_size=(2, 2, 1))

                v_stack = ops.concatenate([v_stack, v], axis=-1)
                k_stack = ops.concatenate([k_stack, k], axis=-1)

                activation_set = [v_stack, k_stack]
            # first entry is [32x32] second entry is [32x32, 16x16] third entry is [32x32, 16x16, 8x8], etc.
            list_stages.append(activation_set)
        else:
            # If the stage is empty, we averagepool only
            v_stack = ops.average_pool(activation_set[0], pool_size=(2, 2, 1))
            k_stack = ops.average_pool(activation_set[1], pool_size=(2, 2, 1))
            activation_set = [v_stack, k_stack]
            list_stages.append([ops.convert_to_tensor(0)])

    return list_stages


class AttentionUnpool(layers.Layer):
    def __init__(self, input_size, data_size):
        super().__init__()
        self.upsampling = layers.UpSampling2D(2)
        self.upsample = input_size < data_size

    def call(self, x):
        # Input is v (batch, height, width, channels, layers)
        # Input is k (batch, height, width, c_attn, layers)
        # upsample across height and width
        if self.upsample:
            B, H, W, C, L = ops.shape(x)
            x = ops.reshape(x, (B, H, W, C * L))
            x = self.upsampling(x)
            x = ops.reshape(x, (B, H * 2, W * 2, C, L))
        return x
