import tensorflow as _tf
from utils import logutil as _logutil
import math

_logging = _logutil.get_logger()
_FLAGS = _tf.app.flags.FLAGS


def resize(x, y):
    x = _tf.image.resize_images(x, (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w))
    y = _tf.image.resize_images(y, (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w))
    return x, y


def _scale(x, y):
    assert x.shape == (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w, 3)
    assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]
    size_h, size_w = (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w)
    fraction = _tf.random_uniform([], 1., 1.5)
    scaled_size = _tf.cast(_tf.stack((size_h * fraction, size_w * fraction), axis=0), dtype=_tf.int32)

    try:
        _x = _tf.image.resize_images(x, scaled_size)
        _y = _tf.image.resize_images(y, scaled_size)

        _x = _tf.image.resize_image_with_crop_or_pad(_x, size_h, size_w)
        _y = _tf.image.resize_image_with_crop_or_pad(_y, size_h, size_w) / fraction

        x = _x
        y = _y
    except Exception as e:
        print(e)

    return x, y


def _rot(x, y):
    angle = _tf.random_uniform([], -5. * math.pi / 180.0, 5. * math.pi / 180.0)
    x = _tf.contrib.image.rotate(x, angle)
    y = _tf.contrib.image.rotate(y, angle)
    return x, y


def _trans(x, y):
    assert x.shape == (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w, 3)
    assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]
    size_h, size_w = (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w)
    off_w = _tf.cast(_tf.random_uniform([], 0, size_w - _FLAGS.dim_input_w), _tf.int32)
    off_h = _tf.cast(_tf.random_uniform([], 0, size_h - _FLAGS.dim_input_h), _tf.int32)
    try:
        _x = _tf.image.crop_to_bounding_box(x, offset_height=off_h, offset_width=off_w,
                                            target_width=_FLAGS.dim_input_w,
                                            target_height=_FLAGS.dim_input_h)
        _y = _tf.image.crop_to_bounding_box(y, offset_height=off_h, offset_width=off_w,
                                            target_width=_FLAGS.dim_input_w,
                                            target_height=_FLAGS.dim_input_h)
        _x = _tf.image.resize_images(_x, (size_h, size_w))
        _y = _tf.image.resize_images(_y / (2 - _FLAGS.dim_input_w / size_w), (size_h, size_w))
        x = _x
        y = _y
    except Exception as e:
        print(e)
    return x, y


def _color(x, y):
    factor = _tf.random_uniform([], 0.8, 1.2)
    x = x * factor

    return x, y


def _flips(x, y):
    x = _tf.image.flip_left_right(x)
    y = _tf.image.flip_left_right(y)

    return x, y


aug_fn = {'scale': _scale, 'trans': _trans, 'color': _color, 'flips': _flips, 'rot': _rot}
rand_aug_fn = {'color': _color, 'flips': _flips}


def _random_aug(stream):
    x, y = stream
    for fn in rand_aug_fn.items():
        x, y = _tf.cond(_tf.less(_tf.random_uniform([], 0, 1.0), .7),
                        lambda: fn[1](x, y),
                        lambda: (x, y))

    x, y = _tf.cond(_tf.less(_tf.random_uniform([], 0, 1.0), .7),
                    lambda: _rot(x, y),
                    lambda: (x, y))

    return resize(x, y)


def _without_aug(stream):
    x, y = stream
    return resize(x, y)


def _sequence_aug(stream, tasks):
    x, y = stream
    for task in tasks:
        if task in aug_fn:
            x, y = aug_fn[task](x, y)
        else:
            _logging.warning("Invalid option(s) [\'%s\']", task)
    return resize(x, y)


def augment(stream, tasks):
    if 'without_aug' in tasks or len(tasks) < 1:
        return _without_aug(stream)
    elif 'random' in tasks:
        return _random_aug(stream)
    else:
        return _sequence_aug(stream, tasks)
