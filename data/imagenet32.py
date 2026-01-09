import tensorflow as tf

import tensorflow_datasets as tfds


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 127.5 - 1.0  # e.g. scale to [-1,1]
    return image


def get_imagenet32():
    train_dataset = tfds.load(
        "imagenet_resized/32x32",
        split="train",
        shuffle_files=True,
        as_supervised=True,  # yields (image, label) pairs
    ).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    val_dataset = tfds.load(
        "imagenet_resized/32x32",
        split="validation",
        shuffle_files=True,
        as_supervised=True,  # yields (image, label) pairs
    ).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    test_dataset = tfds.load(
        "imagenet_resized/32x32",
        split="validation",
        shuffle_files=True,
        as_supervised=True,  # yields (image, label) pairs
    ).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
