import tensorflow as _tf
import tensorflow.contrib.slim.nets as _slim_nets

_slim = _tf.contrib.slim
_FLAGS = _tf.app.flags.FLAGS


def resnet_v1_50(
        inputs,
        num_classes=142,
        is_training=True,
        global_pool=False,
        output_stride=8,
        reuse=None):
    scope = 'resnet_v1_50'
    with _slim.arg_scope(_slim_nets.resnet_v1.resnet_arg_scope()):
        net, end_points = _slim_nets.resnet_v1.resnet_v1_50(
            inputs=inputs,
            num_classes=num_classes,
            is_training=is_training,
            global_pool=global_pool,
            output_stride=output_stride,
            reuse=reuse
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


def resnet_v1_101(
        inputs,
        num_classes=142,
        is_training=True,
        global_pool=False,
        output_stride=8,
        reuse=None):
    with _slim.arg_scope(_slim_nets.resnet_v1.resnet_arg_scope()):
        net, end_points = _slim_nets.resnet_v1.resnet_v1_101(
            inputs=inputs,
            num_classes=num_classes,
            is_training=is_training,
            global_pool=global_pool,
            output_stride=output_stride,
            reuse=reuse
        )

    pretrained_variables = []

    if _FLAGS.use_pretrained_data:
        for var in _slim.get_trainable_variables(scope='resnet_v1_101'):
            for exclusion in ['resnet_v1_101/logits']:
                if var.op.name.startswith(exclusion):
                    break
                else:
                    pretrained_variables.append(var)

    trainable_variables = _slim.get_trainable_variables(scope='resnet_v1_101')
    for var in _tf.global_variables(scope='resnet_v1_101'):
        for include in ['moving_mean', 'moving_variance']:
            if var.op.name.endswith(include):
                trainable_variables.append(var)

    return net, end_points, pretrained_variables, {'resnet_v1_101': trainable_variables}


def resnet_v1_152(
        inputs,
        num_classes=142,
        is_training=True,
        global_pool=False,
        output_stride=8,
        reuse=None):
    with _slim.arg_scope(_slim_nets.resnet_v1.resnet_arg_scope()):
        net, end_points = _slim_nets.resnet_v1.resnet_v1_152(
            inputs=inputs,
            num_classes=num_classes,
            is_training=is_training,
            global_pool=global_pool,
            output_stride=output_stride,
            reuse=reuse
        )

    pretrained_variables = []

    if _FLAGS.use_pretrained_data:
        for var in _slim.get_trainable_variables(scope='resnet_v1_152'):
            for exclusion in ['resnet_v1_152/logits']:
                if var.op.name.startswith(exclusion):
                    break
                else:
                    pretrained_variables.append(var)

    trainable_variables = _slim.get_trainable_variables(scope='resnet_v1_152')
    for var in _tf.global_variables(scope='resnet_v1_152'):
        for include in ['moving_mean', 'moving_variance']:
            if var.op.name.endswith(include):
                trainable_variables.append(var)

    return net, end_points, pretrained_variables, {'resnet_v1_152': trainable_variables}
