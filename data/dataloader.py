import tensorflow as tf
from data.mnist import get_mnist
from cifar10 import get_cifar10
from keras import layers, ops

##### CR-VAE uses these exact parameters in torchvision.transforms.RandomAffine #####
##### https://github.com/sinhasam/CRVAE/blob/main/CRVAE/augmentations.py        #####
interpolation_method = "nearest"
translation = layers.RandomTranslation(
    height_factor=0.05,
    width_factor=0.05,
    interpolation=interpolation_method,
)
rotation = layers.RandomRotation(
    factor=15 / 360,
    interpolation=interpolation_method,
)
zoom = layers.RandomZoom(
    height_factor=0.1,
    interpolation=interpolation_method,
)


def get_data(dataset="mnist", batch_size=100, cr=False):
    if dataset == "mnist":
        x_train, x_val, x_test = get_mnist()
        shuffle_buffer = 10000
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
        test_dataset = tf.data.Dataset.from_tensor_slices(x_test)

    elif dataset == "cifar10":
        x_train, x_val, x_test = get_cifar10()
        shuffle_buffer = 10000
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
        test_dataset = tf.data.Dataset.from_tensor_slices(x_test)

    elif dataset == "imagenet32":
        from imagenet32 import get_imagenet32

        train_dataset, val_dataset, test_dataset = get_imagenet32()
        shuffle_buffer = 1000

    else:
        raise ValueError("Not a valid dataset")

    train_batch_size = batch_size // 2 if cr else batch_size
    train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer)
    train_dataset = train_dataset.batch(train_batch_size, drop_remainder=True)

    # After batching, we concatenate a transformed batch to the original batch for CR-VAE
    if cr:
        train_dataset = train_dataset.map(
            lambda x: ops.concatenate([x, translation(x)], axis=0),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        train_dataset = train_dataset.map(
            lambda x: ops.concatenate(
                [x[: len(x) // 2], rotation(x[len(x) // 2 :])], axis=0
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        train_dataset = train_dataset.map(
            lambda x: ops.concatenate(
                [x[: len(x) // 2], zoom(x[len(x) // 2 :])], axis=0
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_dataset = val_dataset.shuffle(buffer_size=1024)
    val_dataset = val_dataset.batch(batch_size)

    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset
