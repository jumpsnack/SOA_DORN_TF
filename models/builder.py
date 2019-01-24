import tensorflow as _tf
import nets
from utils import logutil as _logutil
from math import exp, log

_aspp = nets.custom
_resnet = nets.resnet
_enet = nets.enet
_yolo = nets.yolo
_vgg = nets.vgg
_inception = nets.inception
_FLAGS = _tf.app.flags.FLAGS
_logging = _logutil.get_logger()
_slim = _tf.contrib.slim


def _gen_disc_labels(nrof_labels, min, max):
    min += _FLAGS.dsc_shift
    max += _FLAGS.dsc_shift

    labels = []
    keys = []
    _logging.debug("=======SID bins======")
    for _k in range(nrof_labels + 1):
        disc_label = exp(log(min) + (log(max / min) * _k) / nrof_labels)
        labels.append(disc_label)
        keys.append(_k)
        _logging.debug("[%d]\t%f" % (_k, disc_label))
    _logging.debug("=====================")
    return labels, _tf.contrib.lookup.HashTable(
        _tf.contrib.lookup.KeyValueTensorInitializer(keys, labels, value_dtype=_tf.float64), 0)


labels_gt, LUT_gt = _gen_disc_labels(_FLAGS.num_bins, _FLAGS.GT_minima, _FLAGS.GT_maxima)


def build(machine):
    INPUTS = _tf.placeholder(_tf.float32, [None, _FLAGS.dim_input_h, _FLAGS.dim_input_w, 3],
                             name="input_placeholder_x")
    OUTPUTS = _tf.placeholder(_tf.float32, [None, _FLAGS.dim_output_h, _FLAGS.dim_output_w, 1])
    IS_TRAINING = _tf.placeholder(bool)
    GLOBAL_STEP = _tf.Variable(0, name='global_step', trainable=False)

    outputs, ord_p, nn_out_labels, pretrained_variables, trainable_variables = _define_nn(INPUTS, IS_TRAINING)
    loss, encoded_gt_labels = _define_loss_fn(ord_p, OUTPUTS)

    decoded_nn_out = _get_decoded_labels(nn_out_labels)
    decoded_gt = _get_decoded_labels(encoded_gt_labels)

    with _tf.control_dependencies(_tf.get_collection(_tf.GraphKeys.UPDATE_OPS)):
        learning_rate = _tf.train.polynomial_decay(_FLAGS.learning_rate, GLOBAL_STEP, power=0.9, decay_steps=1000,
                                                   end_learning_rate=_FLAGS.learning_rate * 0.1)
        optimizer = _tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        train = optimizer.minimize(loss, GLOBAL_STEP)

    a1, a2, a3, abs_rel, sq_rel, rmse, rmse_log, irmse, silog = _get_errors(decoded_nn_out, decoded_gt)

    trainable_variables.append(GLOBAL_STEP)

    machine.LUT_gt = LUT_gt
    machine.INPUTS = INPUTS
    machine.OUTPUTS = OUTPUTS
    machine.IS_TRAINING = IS_TRAINING
    machine.GLOBAL_STEP = GLOBAL_STEP
    machine.loss = loss
    machine.train = train
    machine.decoded_nn_out = decoded_nn_out
    machine.decoded_gt = decoded_gt
    machine.pretrained_variables = pretrained_variables
    machine.trainable_variables = trainable_variables
    machine.abs_rel = abs_rel
    machine.sq_rel = sq_rel
    machine.rmse = rmse
    machine.rmse_log = rmse_log
    machine.irmse = irmse
    machine.a1 = a1
    machine.a2 = a2
    machine.a3 = a3
    machine.silog = silog


def summary(model_vars_dict):
    for label, model_vars in model_vars_dict.items():
        total_size, total_bytes = _slim.model_analyzer.analyze_vars(model_vars, print_info=False)
        _logging.info('====================================')
        _logging.info('\t%s' % label)
        _logging.info('--------------')
        _logging.info('Total size of variables: %s' % format(total_size, ','))
        _logging.info('Total bytes of variables: %s' % format(total_bytes, ','))
        _logging.info('====================================')


def _define_nn(INPUTS, IS_TRAINING):
    def _set_feature_extractor(inputs, is_training):
        if _FLAGS.use_pretrained_data:
            _logging.warning("Use pre-trained data")
        feature_extractor = str(_FLAGS.feature_extractor)
        if feature_extractor.startswith('resnet'):
            version, layers = feature_extractor.split('_')[1:3]
            _logging.info("===================")
            _logging.info(" *Backbone: resnet")
            _logging.info(" *Layers: %s" % layers)
            _logging.info(" *Version: %s" % version)
            _logging.info("===================")

            if version.startswith('v1'):
                if layers.startswith('50'):
                    return _resnet.resnet_v1.resnet_v1_50(
                        inputs=inputs,
                        num_classes=None,
                        is_training=is_training,
                        global_pool=False,
                        output_stride=_FLAGS.output_feature_stride,
                    )
                elif layers.startswith('101'):
                    return _resnet.resnet_v1.resnet_v1_101(
                        inputs=inputs,
                        num_classes=None,
                        is_training=is_training,
                        global_pool=False,
                        output_stride=_FLAGS.output_feature_stride,
                    )
                elif layers.startswith('152'):
                    return _resnet.resnet_v1.resnet_v1_152(
                        inputs=inputs,
                        num_classes=None,
                        is_training=is_training,
                        global_pool=False,
                        output_stride=_FLAGS.output_feature_stride,
                    )
                else:
                    raise AttributeError(
                        "Other the number of resnet layers is incompatible (Declared layers: %s)" % layers)
                pass
            elif version.startswith('v2'):
                if layers.startswith('101'):
                    return _resnet.resnet_v2.resnet_v2_101(
                        inputs=inputs,
                        num_classes=None,
                        is_training=is_training,
                        global_pool=False,
                        output_stride=_FLAGS.output_feature_stride,
                    )
                else:
                    raise AttributeError(
                        "Other the number of resnet layers is incompatible (Declared layers: %s)" % layers)
            else:
                raise AttributeError("Other resnet version is incompatible (Declared verions: %s)" % version)

        elif feature_extractor.startswith('vgg'):
            layers = feature_extractor.split('_')[1]
            _logging.info("===================")
            _logging.info(" *Backbone: vgg")
            _logging.info(" *Layers: %s" % layers)
            _logging.info("===================")

            if layers.startswith("16"):
                return _vgg.vgg_16(inputs,
                                   num_classes=1,
                                   is_training=is_training,
                                   )
            elif layers.startswith("19"):
                return _vgg.vgg_19(inputs,
                                   num_classes=1,
                                   is_training=is_training,
                                   )
            else:
                raise AttributeError(
                    "Other layer size is incompatible (Declared layers: %s)" % layers)

            pass
        elif feature_extractor.startswith('xception'):
            layers = feature_extractor.split('_')[1]
            _logging.info("===================")
            _logging.info(" *Backbone: xception")
            _logging.info(" *Layers: %s" % layers)
            _logging.info("===================")

            if layers.startswith("41"):
                return _inception.xception.xception_41(inputs,
                                                       num_classes=None,
                                                       is_training=is_training,
                                                       output_stride=_FLAGS.output_feature_stride,
                                                       regularize_depthwise=False,
                                                       global_pool=False)
            elif layers.startswith("65"):
                return _inception.xception.xception_65(inputs,
                                                       num_classes=None,
                                                       is_training=is_training,
                                                       output_stride=_FLAGS.output_feature_stride,
                                                       regularize_depthwise=False,
                                                       global_pool=False)
            elif layers.startswith("71"):
                return _inception.xception.xception_71(inputs,
                                                       num_classes=None,
                                                       is_training=is_training,
                                                       output_stride=_FLAGS.output_feature_stride,
                                                       regularize_depthwise=False,
                                                       global_pool=False)
            else:
                raise AttributeError(
                    "Other layer size is incompatible (Declared layers: %s)" % layers)
                pass
        elif feature_extractor.startswith('inception'):
            version = feature_extractor.split('_')[1]
            _logging.info("===================")
            _logging.info(" *Backbone: inception")
            _logging.info(" *version: %s" % version)
            _logging.info("===================")

            if version.startswith("v1"):
                return
            elif version.startswith("v2"):
                return
            elif version.startswith("v3"):
                return _inception.inception_v3.inception_v3(inputs, num_classes=None,
                                                            is_training=is_training)
            elif version.startswith("v4"):
                return _inception.inception_v4.inception_v4(inputs, num_classes=None,
                                                            is_training=is_training)
            elif version.startswith("resnet_v2"):
                return _inception.inception_v4.inception_resnet_v2(inputs, num_classes=None,
                                                                   is_training=is_training)
            else:
                raise AttributeError(
                    "Other version is incompatible (Declared version: %s)" % version)
                pass
        elif feature_extractor.startswith('enet'):
            _logging.info("===================")
            _logging.info(" *Backbone: enet")
            _logging.info("===================")

            return _enet.enet.enet(inputs, 3, 142, is_training, weight_decay=2e-4, num_initial_blocks=1,
                                   stage_two_repeat=2, skip_connections=False)
        elif feature_extractor.startswith('yolo'):
            version = feature_extractor.split('_')[1]
            spec = feature_extractor.split('_')[2]
            _logging.info("===================")
            _logging.info(" *Backbone: yolo")
            _logging.info(" *version: %s" % version)
            if spec:
                _logging.info(" *spec: %s" % spec)
            _logging.info("===================")

            if version.startswith("v1"):
                return
            elif version.startswith("v2"):
                return
            elif version.startswith("v3"):
                return _yolo.yolo_v3.yolo_v3_tiny(inputs, 142, is_training=is_training, data_format='NHWC')
            else:
                raise AttributeError(
                    "Other version is incompatible (Declared version: %s)" % version)
                pass
        else:
            raise AttributeError("Compatible with resnet, vgg, xception only")

    def _set_aspp(feats, is_training):
        return _aspp.DORN_H_Fu.scene_understanding_modular(
            feats=feats,
            atrous_rate=[6, 12, 18],
            is_training=is_training
        )

    def _get_ordinal_values(outputs):
        (H, W, C) = outputs.get_shape().as_list()[1:]
        N = _FLAGS.batch_size
        ord_num = int(C / 2)
        ord_p = _tf.nn.softmax(_tf.reshape(outputs, shape=[N, H, W, ord_num, 2]), axis=4)
        nn_out_labels = _tf.reduce_sum(_tf.argmax(ord_p, axis=4, output_type=_tf.int32), axis=3, keepdims=True)
        return ord_p, nn_out_labels

    pretrained_vars = []
    trainable_vars = []

    # =================================
    #     feature extractor
    # =================================
    feats, end_points, feats_extractor_pretrained_vars, feats_extractor_trainable_vars = _set_feature_extractor(INPUTS,
                                                                                                                IS_TRAINING)
    summary(feats_extractor_trainable_vars)
    trainable_vars.extend(feats_extractor_trainable_vars.popitem()[1])
    pretrained_vars.extend(feats_extractor_pretrained_vars)

    # =================================
    #     ASPP
    # =================================
    outputs, aspp_trainable_vars = _set_aspp(feats, IS_TRAINING)
    summary(aspp_trainable_vars)
    trainable_vars.extend(aspp_trainable_vars.popitem()[1])

    # =================================
    #     Ordinal value
    # =================================
    ord_p, nn_out_labels = _get_ordinal_values(outputs)

    return outputs, ord_p, nn_out_labels, pretrained_vars, trainable_vars


def _define_loss_fn(ord_p, groundtruth):
    def _get_ordinal_encoded_gt(groundtruth):
        (N, H, W, C) = groundtruth.get_shape().as_list()
        if N is None:
            N = _FLAGS.batch_size
        groundtruth = _tf.reshape(groundtruth, shape=[N, H, W, 1])
        ones = _tf.ones_like(groundtruth)
        zeros = _tf.zeros_like(groundtruth)
        decoded_label = _tf.zeros_like(groundtruth)
        k = _tf.constant(0)

        shifted_groundtruth = _tf.cast(groundtruth + _FLAGS.dsc_shift, dtype=_tf.float64)

        _condition = lambda decoded_label, k, shifted_grountruth: _tf.less(k, _FLAGS.num_bins)
        _body = lambda decoded_label, k, shifted_grountruth: (
            _tf.add(
                decoded_label,
                _tf.where(
                    _tf.logical_and(_tf.greater_equal(shifted_grountruth, LUT_gt.lookup(k)),
                                    _tf.less(shifted_grountruth, LUT_gt.lookup(_tf.add(k, 1)))),
                    _tf.multiply(ones, _tf.cast(k, dtype=_tf.float32)),
                    zeros
                )
            ),
            _tf.add(k, 1),
            shifted_grountruth
        )

        result = _tf.while_loop(_condition, _body, (decoded_label, k, shifted_groundtruth))

        encoded_label = _tf.where(
            _tf.greater_equal(shifted_groundtruth, _FLAGS.GT_maxima + _FLAGS.dsc_shift),
            _tf.multiply(
                ones, _tf.constant(_FLAGS.num_bins - 1, dtype=_tf.float32)
            ),
            result[0]
        )

        encoded_label = _tf.where(
            _tf.less_equal(groundtruth, _FLAGS.GT_minima),
            -ones,
            encoded_label
        )

        return _tf.reshape(encoded_label, shape=[N, H, W, 1])

    (ord_log_n_pk, ord_log_pk) = _tf.unstack(_tf.log(ord_p), axis=4)
    encoded_gt_labels = _get_ordinal_encoded_gt(groundtruth)

    (N, H, W, C) = ord_log_pk.get_shape().as_list()
    if N is None:
        N = _FLAGS.batch_size

    global_mask = _tf.where(_tf.less(encoded_gt_labels, 0), _tf.zeros_like(encoded_gt_labels),
                            _tf.ones_like(encoded_gt_labels))
    foreground_mask = _tf.reshape(_tf.sequence_mask(encoded_gt_labels - 1, C), shape=[N, H, W, C])

    sum_of_p = _tf.reduce_sum(_tf.where(foreground_mask, ord_log_pk, ord_log_n_pk), axis=3, keepdims=True)
    loss = -(_tf.div(_tf.reduce_sum(sum_of_p * global_mask), _tf.reduce_sum(global_mask)))

    return loss, encoded_gt_labels


def _get_decoded_labels(encoded_labels):
    encoded_labels = _tf.cast(encoded_labels, dtype=_tf.int32)
    inv_mask = _tf.cast(_tf.greater_equal(encoded_labels, 0), dtype=_tf.float64)

    decoded_labels = (LUT_gt.lookup(encoded_labels) + LUT_gt.lookup(_tf.add(1, encoded_labels))) / 2.0
    decoded_labels *= inv_mask

    return decoded_labels


def _get_errors(pred, gt):
    pred = _tf.clip_by_value(_tf.cast(pred, _tf.float32), _FLAGS.GT_minima, _FLAGS.GT_maxima)
    gt = _tf.cast(gt, _tf.float32)
    mask = _tf.logical_and(gt > _FLAGS.GT_minima, gt <= _FLAGS.GT_maxima)
    pred = _tf.boolean_mask(pred, mask)
    gt = _tf.boolean_mask(gt, mask)
    npix = _tf.reduce_sum(_tf.cast(mask, dtype=_tf.float32))

    thresh = _tf.maximum((gt / pred), (pred / gt))
    a1 = (npix - _tf.reduce_sum(_tf.cast(_tf.greater_equal(thresh, 1.25), dtype=_tf.float32))) / npix
    a2 = (npix - _tf.reduce_sum(_tf.cast(_tf.greater_equal(thresh, 1.25 ** 2), dtype=_tf.float32))) / npix
    a3 = (npix - _tf.reduce_sum(_tf.cast(_tf.greater_equal(thresh, 1.25 ** 3), dtype=_tf.float32))) / npix

    rmse = _tf.reduce_sum(((gt - pred) ** 2)) / npix
    rmse = _tf.sqrt(rmse)

    rmse_log = _tf.reduce_sum(((_tf.log(gt) - _tf.log(pred)) ** 2)) / npix
    rmse_log = _tf.sqrt(rmse_log)

    irmse = _tf.sqrt(_tf.reduce_sum(((1 / pred - 1 / gt)) ** 2) / npix)

    abs_rel = _tf.reduce_sum((_tf.abs(pred - gt) / gt)) / npix

    sq_rel = _tf.reduce_sum((((pred - gt) ** 2) / gt)) / npix

    d = _tf.log(pred) - _tf.log(gt)
    silog = (_tf.reduce_sum(d ** 2) / npix) - (((_tf.reduce_sum(d)) ** 2) / (npix ** 2))

    return a1, a2, a3, abs_rel, sq_rel, rmse, rmse_log, irmse, silog
