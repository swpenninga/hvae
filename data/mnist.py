import tensorflow as tf
from keras import datasets, layers, ops


def get_mnist():
    (x_train, _), (x_test, _) = datasets.mnist.load_data()
    x_train = x_train / 255  # from [0, 255] -> [0, 1]
    x_test = x_test / 255  # from [0, 255] -> [0, 1]

    x_train = ops.expand_dims(x_train, -1)  # 60000, 28, 28, 1
    x_test = ops.expand_dims(x_test, -1)  # 10000, 28, 28, 1

    x_train = layers.ZeroPadding2D(padding=(2, 2))(x_train)  # 60000, 32, 32, 1
    x_test = layers.ZeroPadding2D(padding=(2, 2))(x_test)  # 10000, 32, 32, 1

    x_test = tf.random.shuffle(x_test, seed=1337)  # Fixed shuffle of val/test split

    x_val = x_test[: len(x_test) // 2]  # 5000, 32, 32, 1
    x_test = x_test[len(x_test) // 2 :]  # 5000, 32, 32, 1
    return x_train, x_val, x_test
