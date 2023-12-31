import functools
import random

import tensorflow as tf


@tf.function
def center_crop(image, size):
    # for image of shape [batch, height, width, channels] or [height, width, channels]
    if not isinstance(size, (tuple, list)):
        size = [size, size]
    offset_height = (tf.shape(image)[-3] - size[0]) // 2
    offset_width = (tf.shape(image)[-2] - size[1]) // 2
    return tf.image.crop_to_bounding_box(image, offset_height, offset_width, size[0], size[1])


@tf.function
def color_jitter(image, brightness=0, contrast=0, saturation=0, hue=0):
    """Color jitter.

    Examples
    --------
    >>> color_jitter(img, 25, 0.2, 0.2, 0.1)

    """
    tforms = []
    if brightness > 0:
        tforms.append(functools.partial(tf.image.random_brightness, max_delta=brightness))
    if contrast > 0:
        tforms.append(functools.partial(tf.image.random_contrast, lower=max(0, 1 - contrast), upper=1 + contrast))
    if saturation > 0:
        tforms.append(functools.partial(tf.image.random_saturation, lower=max(0, 1 - saturation), upper=1 + saturation))
    if hue > 0:
        tforms.append(functools.partial(tf.image.random_hue, max_delta=hue))

    random.shuffle(tforms)
    for tform in tforms:
        image = tform(image)

    return image


@tf.function
def random_grayscale(image, p=0.1):
    return tf.cond(pred=tf.random.uniform(()) < p,
                   true_fn=lambda: tf.image.adjust_saturation(image, 0),
                   false_fn=lambda: image)


@tf.function
def random_rotate(images, max_degrees, interpolation='BILINEAR'):
    # Randomly rotate image(s) counterclockwise by the angle(s) uniformly chosen from [-max_degree(s), max_degree(s)].
    max_degrees = tf.convert_to_tensor(max_degrees, dtype=tf.float32)
    angles_deg = tf.random.uniform(tf.shape(max_degrees), minval=-max_degrees, maxval=max_degrees)

    # Convert images tensor from [0, 1] range to [0, 255], as `apply_affine_transform` expects that range.
    images_255 = tf.cast(images * 255, dtype=tf.uint8)

    # Use a python loop to rotate each image individually
    rotated_images = []
    for i in range(images_255.shape[0]):
        rotated_image = tf.keras.preprocessing.image.apply_affine_transform(
            images_255[i].numpy(),
            theta=angles_deg[i].numpy(),
            channel_axis=2,
            row_axis=0,
            col_axis=1,
            fill_mode=interpolation.lower()
        )
        rotated_images.append(rotated_image)

    # Convert list of arrays back to tensor and normalize to [0, 1] range.
    return tf.convert_to_tensor(rotated_images, dtype=tf.float32) / 255.0
