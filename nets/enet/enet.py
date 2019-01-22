from nets._tf_models.enet import ENet, ENet_arg_scope
import tensorflow as _tf
import tensorflow.contrib.slim.nets as _slim_nets

_slim = _tf.contrib.slim
_FLAGS = _tf.app.flags.FLAGS


def enet(inputs, batch_size, num_classes, is_training, weight_decay=2e-4, num_initial_blocks=1, stage_two_repeat=2,
         skip_connections=True):
    scope = 'ENet'
    with _slim.arg_scope(ENet_arg_scope(weight_decay=weight_decay)):
        logits, trainable_variables = ENet(inputs,
                                     num_classes,
                                     batch_size=batch_size,
                                     is_training=is_training,
                                     reuse=None,
                                     num_initial_blocks=num_initial_blocks,
                                     stage_two_repeat=stage_two_repeat,
                                     skip_connections=skip_connections,
                                           do_upsampling=True)

    pretrained_variables = []

    if _FLAGS.use_pretrained_data:
        for var in trainable_variables:
            for exclusion in [scope + '/fullconv']:
                if var.op.name.startswith(exclusion):
                    break
                else:
                    pretrained_variables.append(var)

    for var in _tf.global_variables(scope=scope):
        for include in ['moving_mean', 'moving_variance']:
            if var.op.name.endswith(include):
                trainable_variables.append(var)

    return logits, None, pretrained_variables, {scope: trainable_variables}
