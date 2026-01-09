from keras import ops


def matrix_log(B, terms=10):
    assert ops.shape(B)[0] == ops.shape(B)[1]
    I = ops.eye(ops.shape(B)[0])
    B_min_I = B - I

    # for k = 1.
    product = B_min_I
    result = B_min_I

    is_minus = -1
    for k in range(2, terms):
        # Reweighing with k term.
        product = ops.matmul(product, B_min_I) * (k - 1) / k
        result = result + is_minus * product

        is_minus *= -1

    return result


def matrix_exp(M, terms=10):
    assert ops.shape(M)[0] == ops.shape(M)[1]
    I = ops.eye(ops.shape(M)[0])
    # for i = 0.
    result = I
    product = I

    for i in range(1, terms + 1):
        product = ops.matmul(product, M) / i
        result = result + product

    return result


def conv_exp(input, kernel, terms=10, dynamic_truncation=0, verbose=False):
    B, H, W, C = ops.shape(input)
    assert ops.shape(kernel)[0] == ops.shape(kernel)[1]
    result = input
    product = input

    for i in range(1, terms + 1):
        product = ops.conv(product, kernel, strides=1, padding="same") / i
        result += product

        if dynamic_truncation != 0 and i > 5:
            if ops.max(ops.abs(product)) < dynamic_truncation:
                break

    if verbose:
        print("Maximum element size in term: {}".format(ops.max(ops.abs(product))))

    return result


def inv_conv_exp(input, kernel, terms=10, dynamic_truncation=0, verbose=False):
    return conv_exp(input, -kernel, terms, dynamic_truncation, verbose)


def log_det(kernel):
    K1, K2, _, _ = ops.shape(kernel)
    M1 = (K1 - 1) // 2
    M2 = (K2 - 1) // 2
    trace = ops.trace(kernel[M1, M2, :, :])
    return trace


def convergence_scale(c, kernel_size):
    K1, K2, C_in, C_out = kernel_size

    d = C_in * K1 * K2

    return c / d
