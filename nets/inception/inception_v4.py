import tensorflow as _tf
from nets._tf_models.inception_resnet_v2 import inception_resnet_v2 as _inception_resnet_v2
from nets._tf_models.inception_resnet_v2 import inception_resnet_v2_arg_scope as _inception_resnet_v2_arg_scope
from nets._tf_models.inception_v4 import inception_v4 as _inception_v4
from nets._tf_models.inception_v4 import inception_v4_arg_scope as _inception_v4_arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib

_slim = _tf.contrib.slim
_FLAGS = _tf.app.flags.FLAGS


def inception_resnet_v2(inputs, num_classes=1001, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        create_aux_logits=True,
                        activation_fn=_tf.nn.relu):
    scope = 'InceptionResnetV2'
    with _slim.arg_scope(_inception_resnet_v2_arg_scope()):
        net, end_points = _inception_resnet_v2(
            inputs, num_classes=num_classes, is_training=is_training,
            dropout_keep_prob=dropout_keep_prob,
            reuse=reuse,
            scope=scope,
            create_aux_logits=create_aux_logits,
            activation_fn=activation_fn
        )

    pretrained_variables = []

    if _FLAGS.use_pretrained_data:
        for var in _slim.get_trainable_variables(scope=scope):
            for exclusion in [scope + '/Logits']:
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


def inception_v4(inputs, num_classes=1001, is_training=True,
                 dropout_keep_prob=0.8,
                 reuse=None,
                 create_aux_logits=True):
    scope = 'InceptionV4'
    with _slim.arg_scope(_inception_v4_arg_scope()):
        net, end_points = _inception_v4(inputs, num_classes=num_classes, is_training=is_training,
                                        dropout_keep_prob=dropout_keep_prob,
                                        reuse=reuse,
                                        scope=scope,
                                        create_aux_logits=create_aux_logits)

    pretrained_variables = []

    if _FLAGS.use_pretrained_data:
        for var in _slim.get_trainable_variables(scope=scope):
            for exclusion in [scope + '/Logits']:
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
