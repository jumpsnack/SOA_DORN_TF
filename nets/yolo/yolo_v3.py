import numpy as _np
import tensorflow as _tf

_slim = _tf.contrib.slim
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(10, 14), (23, 27), (37, 58),
            (81, 82), (135, 169), (344, 319)]

_FLAGS = _tf.app.flags.FLAGS

def _get_size(shape, data_format):
    if len(shape) == 4:
        shape = shape[1:]
    return shape[1:3] if data_format == 'NCHW' else shape[0:2]


@_tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('NHWC' or 'NCHW').
      mode: The mode for tf.pad.
    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if kwargs['data_format'] == 'NCHW':
        padded_inputs = _tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]],
                               mode=mode)
    else:
        padded_inputs = _tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs


def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = _slim.conv2d(inputs, filters, kernel_size, stride=strides,
                          padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


def _detection_layer(inputs, num_classes, anchors, img_size, data_format):
    num_anchors = len(anchors)
    predictions = _slim.conv2d(inputs, num_anchors * (5 + num_classes), 1,
                               stride=1, normalizer_fn=None,
                               activation_fn=None,
                               biases_initializer=_tf.zeros_initializer())

    shape = predictions.get_shape().as_list()
    grid_size = _get_size(shape, data_format)
    dim = grid_size[0] * grid_size[1]
    bbox_attrs = 5 + num_classes

    if data_format == 'NCHW':
        predictions = _tf.reshape(
            predictions, [-1, num_anchors * bbox_attrs, dim])
        predictions = _tf.transpose(predictions, [0, 2, 1])

    predictions = _tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])

    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])

    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

    box_centers, box_sizes, confidence, classes = _tf.split(
        predictions, [2, 2, 1, num_classes], axis=-1)

    box_centers = _tf.nn.sigmoid(box_centers)
    confidence = _tf.nn.sigmoid(confidence)

    grid_x = _tf.range(grid_size[0], dtype=_tf.float32)
    grid_y = _tf.range(grid_size[1], dtype=_tf.float32)
    a, b = _tf.meshgrid(grid_x, grid_y)

    x_offset = _tf.reshape(a, (-1, 1))
    y_offset = _tf.reshape(b, (-1, 1))

    x_y_offset = _tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = _tf.reshape(_tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    anchors = _tf.tile(anchors, [dim, 1])
    box_sizes = _tf.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride

    detections = _tf.concat([box_centers, box_sizes, confidence], axis=-1)

    classes = _tf.nn.sigmoid(classes)
    predictions = _tf.concat([detections, classes], axis=-1)
    return predictions


def _upsample(inputs, out_shape, data_format='NCHW'):
    # tf.image.resize_nearest_neighbor accepts input in format NHWC
    if data_format == 'NCHW':
        inputs = _tf.transpose(inputs, [0, 2, 3, 1])

    if data_format == 'NCHW':
        new_height = out_shape[3]
        new_width = out_shape[2]
    else:
        new_height = out_shape[2]
        new_width = out_shape[1]

    inputs = _tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    # back to NCHW if needed
    if data_format == 'NCHW':
        inputs = _tf.transpose(inputs, [0, 3, 1, 2])

    inputs = _tf.identity(inputs, name='upsampled')
    return inputs


def yolo_v3_tiny(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    img_size = inputs.get_shape().as_list()[1:3]

    if data_format == 'NCHW':
        inputs = _tf.transpose(inputs, [0, 3, 1, 2])

    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    end_points = []
    with _slim.arg_scope([_slim.conv2d, _slim.batch_norm, _fixed_padding, _slim.max_pool2d],
                         data_format=data_format):
        with _slim.arg_scope([_slim.conv2d, _slim.batch_norm, _fixed_padding], reuse=reuse):
            with _slim.arg_scope([_slim.conv2d],
                                 normalizer_fn=_slim.batch_norm,
                                 normalizer_params=batch_norm_params,
                                 biases_initializer=None,
                                 activation_fn=lambda x: _tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):

                with _tf.variable_scope('yolo_v3_tiny'):
                    for i in range(6):
                        inputs = _conv2d_fixed_padding(
                            inputs, 16 * pow(2, i), 3)
                        end_points.append(inputs)
                        if i == 4:
                            route_1 = inputs
                            end_points.append(_tf.identity(route_1, name='route_1'))
                        if i == 5:
                            inputs = _slim.max_pool2d(
                                inputs, [2, 2], stride=1, padding="SAME", scope='pool2')
                        else:
                            inputs = _slim.max_pool2d(
                                inputs, [2, 2], scope='pool2')

                    inputs = _conv2d_fixed_padding(inputs, 1024, 3)
                    end_points.append(inputs)
                    inputs = _conv2d_fixed_padding(inputs, 256, 1)
                    end_points.append(inputs)
                    route_2 = inputs
                    end_points.append(_tf.identity(route_2, name='route_2'))

                    inputs = _conv2d_fixed_padding(inputs, 512, 3)
                    # inputs = _conv2d_fixed_padding(inputs, 255, 1)

                    # detect_1 = _detection_layer(
                    #     inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
                    # detect_1 = _tf.identity(detect_1, name='detect_1')

                    inputs = _conv2d_fixed_padding(route_2, 128, 1)
                    end_points.append(inputs)
                    upsample_size = route_1.get_shape().as_list()
                    inputs = _upsample(inputs, upsample_size, data_format)
                    end_points.append(inputs)
                    inputs = _tf.transpose(inputs, [0, 2, 1, 3])

                    inputs = _tf.concat([inputs, route_1],
                                        axis=1 if data_format == 'NCHW' else 3)
                    end_points.append(inputs)

                    # inputs = _conv2d_fixed_padding(inputs, 256, 3)
                    inputs = _conv2d_fixed_padding(inputs, num_classes, 3)
                    end_points.append(inputs)

                    # detect_2 = _detection_layer(
                    #     inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
                    # detect_2 = _tf.identity(detect_2, name='detect_2')
                    #
                    # detections = _tf.concat([detect_1, detect_2], axis=1)
                    # detections = _tf.identity(detections, name='detections')

    pretrained_variables = []
    trainable_variables = _slim.get_trainable_variables(scope='yolo_v3_tiny')
    if _FLAGS.use_pretrained_data:
        for var in trainable_variables:
            for exclusion in ['yolo_v3_tiny' + '/fullconv']:
                if var.op.name.startswith(exclusion):
                    break
                else:
                    pretrained_variables.append(var)

    trainable_variables = _slim.get_trainable_variables(scope='yolo_v3_tiny')
    for var in _tf.global_variables(scope='yolo_v3_tiny'):
        for include in ['moving_mean', 'moving_variance']:
            if var.op.name.endswith(include):
                trainable_variables.append(var)

    return inputs, end_points, pretrained_variables, {'yolo_v3_tiny': trainable_variables}
