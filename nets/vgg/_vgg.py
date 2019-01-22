import tensorflow as _tf
import tensorflow.contrib.slim.nets as _slim_nets

_slim = _tf.contrib.slim
_FLAGS = _tf.app.flags.FLAGS

def vgg_16(
        inputs,
        num_classes=142,
        is_training=True,
        dropout_keep_prob = 0.5,
        spatial_squeeze=False):
    scope = 'vgg_16'
    with _slim.arg_scope(_slim_nets.vgg.vgg_arg_scope()):
        net, end_points = _slim_nets.vgg.vgg_16(
            inputs,
            num_classes=num_classes,
            is_training=is_training,
            dropout_keep_prob=dropout_keep_prob,
            spatial_squeeze=spatial_squeeze,
            scope=scope
        )
        net = end_points["vgg_16/conv5/conv5_3"]

    pretrained_variables = []

    if _FLAGS.use_pretrained_data:
        for var in _slim.get_trainable_variables(scope=scope):
            for exclusion in [scope + '/fc8']:
                if var.op.name.startswith(exclusion):
                    break
                else:
                    pretrained_variables.append(var)

    trainable_variables = _slim.get_trainable_variables(scope=scope)
    for var in _tf.global_variables(scope=scope):
        for include in ['moving_mean', 'moving_variance']:
            if var.op.name.endswith(include):
                trainable_variables.append(var)


    return net, end_points, pretrained_variables, {scope: trainable_variables}

def vgg_19(
        inputs,
        num_classes=142,
        is_training=True,
        dropout_keep_prob = 0.5,
        spatial_squeeze=False):
    scope = 'vgg_19'
    with _slim.arg_scope(_slim_nets.vgg.vgg_arg_scope()):
        net, end_points = _slim_nets.vgg.vgg_19(
            inputs,
            num_classes=num_classes,
            is_training=is_training,
            dropout_keep_prob=dropout_keep_prob,
            spatial_squeeze=spatial_squeeze,
            scope=scope
        )

    pretrained_variables = []

    if _FLAGS.use_pretrained_data:
        for var in _slim.get_trainable_variables(scope=scope):
            for exclusion in [scope + '/logits']:
                if var.op.name.startswith(exclusion):
                    break
                else:
                    pretrained_variables.append(var)

    trainable_variables = _slim.get_trainable_variables(scope=scope)
    for var in _tf.global_variables(scope=scope):
        for include in ['moving_mean', 'moving_variance']:
            if var.op.name.endswith(include):
                trainable_variables.append(var)


    return net, end_points, pretrained_variables, {scope: trainable_variables}
