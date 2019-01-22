import tensorflow as _tf
from nets._tf_models import xception as _xception

_slim = _tf.contrib.slim
_FLAGS = _tf.app.flags.FLAGS


def xception_41(inputs,
                num_classes=None,
                is_training=True,
                global_pool=False,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None):
    scope = 'xception_41'
    with _slim.arg_scope(_xception.xception_arg_scope()):
        net, end_points = _xception.xception_41(
            inputs,
            num_classes=num_classes,
            is_training=is_training,
            global_pool=global_pool,
            keep_prob=keep_prob,
            output_stride=output_stride,
            regularize_depthwise=regularize_depthwise,
            multi_grid=multi_grid,
            reuse=reuse,
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


def xception_65(inputs,
                num_classes=None,
                is_training=True,
                global_pool=False,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None):
    scope = 'xception_65'
    with _slim.arg_scope(_xception.xception_arg_scope()):
        net, end_points = _xception.xception_65(
            inputs,
            num_classes=num_classes,
            is_training=is_training,
            global_pool=global_pool,
            keep_prob=keep_prob,
            output_stride=output_stride,
            regularize_depthwise=regularize_depthwise,
            multi_grid=multi_grid,
            reuse=reuse,
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


def xception_71(inputs,
                num_classes=None,
                is_training=True,
                global_pool=False,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None):
    scope = 'xception_71'
    with _slim.arg_scope(_xception.xception_arg_scope()):
        net, end_points = _xception.xception_71(
            inputs,
            num_classes=num_classes,
            is_training=is_training,
            global_pool=global_pool,
            keep_prob=keep_prob,
            output_stride=output_stride,
            regularize_depthwise=regularize_depthwise,
            multi_grid=multi_grid,
            reuse=reuse,
            scope=scope
        )

    pretrained_variables = []

    if _FLAGS.use_pretrained_data:
        for var in _slim.get_trainable_variables(scope=scope):
            for exclusion in [scope+'/logits']:
                if var.op.name.startswith(exclusion) > 0:
                    break
                else:
                    pretrained_variables.append(var)

    trainable_variables = _slim.get_trainable_variables(scope=scope)
    for var in _tf.global_variables(scope=scope):
        for include in ['moving_mean', 'moving_variance']:
            if var.op.name.endswith(include):
                trainable_variables.append(var)

    return net, end_points, pretrained_variables, {scope: trainable_variables}