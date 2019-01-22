import tensorflow as _tf

_slim = _tf.contrib.slim
_FLAGS = _tf.app.flags.FLAGS

def prelu(x, name=None):
    '''
    Performs the parametric relu operation. This implementation is based on:
    https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow

    For the decoder portion, prelu becomes just a normal prelu

    INPUTS:
    - x(Tensor): a 4D Tensor that undergoes prelu
    - scope(str): the string to name your prelu operation's alpha variable.
    - decoder(bool): if True, prelu becomes a normal relu.

    OUTPUTS:
    - pos + neg / x (Tensor): gives prelu output only during training; otherwise, just return x.

    '''
    scope = name+'/' if name is not None else ''
    alpha = _tf.get_variable(scope + 'alpha', x.get_shape()[-1],
                            initializer=_tf.constant_initializer(0.0),
                            dtype=_tf.float32)
    pos = _tf.nn.relu(x)
    neg = alpha * (x - abs(x)) * 0.5
    return pos + neg

def scene_understanding_modular(feats,num_classes=142, depth=512, atrous_rate=None, is_training=True):
    (N, H, W, C) = feats.get_shape().as_list()
    if atrous_rate is None:
        atrous_rate = [6, 12, 18]

    _branches = []
    _activation = _tf.nn.relu
    with _tf.variable_scope('scene_understanding_modular'):
        with _tf.variable_scope('full_image_encoder'):
            global_pooling = _slim.avg_pool2d(feats, int(H / 4), int(H / 4))
            # TOP: (?,4,5,2048)
            global_drop = _slim.dropout(global_pooling, 0.5, is_training=is_training)
            global_drop = _slim.flatten(global_drop)
            global_fc = _slim.fully_connected(global_drop, depth, activation_fn=_activation)
            # TOP: (?,512)
            global_reshape = _tf.reshape(global_fc, [-1, 1, 1, depth])
            # TOP: (?,1,1,512)
            conv6_1 = _slim.conv2d(global_reshape, depth, 1, 1, activation_fn=_activation)
            # TOP: (?,1,1,512)
            conv6_1 = _tf.tile(conv6_1, [1, H, W, 1])
            # conv6_1 = _tf.image.resize_bilinear(conv6_1, [H, W])
            _branches.append(conv6_1)

        with _tf.variable_scope('aspp_0'):
            # aspp_1 = _slim.conv2d(_tf.pad(feats, [[0, 0], [0, 0], [0, 0], [0, 0]]), depth, kernel_size=1, rate=1,
            #                       activation_fn=_activation, padding='VALID')
            aspp_1 = _slim.conv2d(feats, depth, 1, rate=1,activation_fn=_activation)
            conv6_2 = _slim.conv2d(aspp_1, depth, kernel_size=1, activation_fn=_activation)
            _branches.append(conv6_2)

        for i, rate in enumerate(atrous_rate, 1):
            with _tf.variable_scope('aspp_%d' % i):
                _aspp = _slim.conv2d(feats, depth, 3, rate=rate,activation_fn=_activation)
                _conv = _slim.conv2d(_aspp, depth, kernel_size=1, stride=1, activation_fn=_activation)
                _branches.append(_conv)

        with _tf.variable_scope('post_proc'):
            conv6_concat = _tf.concat(_branches, axis=3)
            # TOP: (?,49,65,2560)
            conv6 = _slim.dropout(conv6_concat, 0.5, is_training=is_training)

            conv7 = _slim.conv2d(conv6, 2048, 1, 1, activation_fn=_activation)
            # TOP: (?,49,65,2048)
            conv7 = _slim.dropout(conv7, 0.5, is_training=is_training)

            # TOP: (?,49,65,2048)
            conv8 = _slim.conv2d(conv7, num_classes, 1, 1, activation_fn=None)
            # TOP: (?,49,65,142)
            conv8 = _tf.image.resize_bilinear(conv8, [_FLAGS.dim_input_h, _FLAGS.dim_input_w])

    trainable_variables = _slim.get_trainable_variables(scope='scene_understanding_modular')
    for var in _tf.global_variables(scope='scene_understanding_modular'):
        for include in ['moving_mean', 'moving_variance']:
            if var.op.name.endswith(include):
                trainable_variables.append(var)
    """[None, 385, 513, 142]"""
    return conv8, {'scene_understanding_modular': trainable_variables}
