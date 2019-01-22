import tensorflow as _tf
import tensorflow.contrib.slim.nets as _slim_nets
from tensorflow.contrib.layers.python.layers import layers as layers_lib

_slim = _tf.contrib.slim
_FLAGS = _tf.app.flags.FLAGS


def inception_v3(inputs,
                 num_classes,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=layers_lib.softmax,
                 spatial_squeeze=False):
    scope = 'InceptionV3'
    with _slim.arg_scope(_slim_nets.inception.inception_v3_arg_scope()):
        net, end_points = _slim_nets.inception.inception_v3(
            inputs,
            num_classes=num_classes,
            is_training=is_training,
            dropout_keep_prob=dropout_keep_prob,
            min_depth=min_depth,
            depth_multiplier=depth_multiplier,
            prediction_fn=prediction_fn,
            spatial_squeeze=spatial_squeeze,
            reuse=None,
            scope=scope
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
