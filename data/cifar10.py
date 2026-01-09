import tensorflow as tf
from keras import datasets, ops


def get_cifar10():
    (x_train, _), (x_test, _) = datasets.cifar10.load_data()

    x_train = tf.random.shuffle(x_train)
    x_test = tf.random.shuffle(x_test, seed=1337)

    x_train = ops.floor(x_train)  # [0, 255]
    x_test = ops.floor(x_test)  # [0, 255]

    x_train = (x_train - 127.5) / 127.5  # [-1, 1]
    x_test = (x_test - 127.5) / 127.5  # [-1, 1]

    x_train = ops.cast(x_train, dtype="float32")
    x_test = ops.cast(x_test, dtype="float32")

    x_val = x_test[: len(x_test) // 2]  # 5000, 32, 32, 3
    x_test = x_test[len(x_test) // 2 :]  # 5000, 32, 32, 3

    return x_train, x_val, x_test
