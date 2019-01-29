from utils import logutil
import tensorflow as _tf
import numpy as _np
import os as _os
from tensorflow.python.platform import gfile as _gfile
import matplotlib.pyplot as _plt
from prettytable import PrettyTable as _pt
import time
import shutil
import cv2

_logging = logutil.get_logger()
_FLAGS = _tf.app.flags.FLAGS
_slim = _tf.contrib.slim


class _Machine:
    def __init__(self, tf_session, model_builder, dataset_loader):
        model_builder.build(self)
        dataset_loader.load(self)

        self.sess = tf_session
        self.sess.run([_tf.global_variables_initializer(), _tf.local_variables_initializer()])
        self.LUT_gt.init.run()

        self.pretrain_restorer = None
        self.trainable_restorer = None

        self.restore()
        self.set_summaries()

    def set_summaries(self):
        if _FLAGS.use_tensorboard:
            _tf.summary.scalar("total_loss", self.loss)
            _tf.summary.scalar("errors/absErrorRel", self.abs_rel)
            _tf.summary.scalar("errors/sqErrorRel", self.sq_rel)
            _tf.summary.scalar("errors/RMSE", self.rmse)
            _tf.summary.scalar("errors/RMSElog", self.rmse_log)
            _tf.summary.scalar("errors/iRMSE", self.irmse)
            _tf.summary.scalar("errors/SILog", self.silog)
            _tf.summary.scalar("accuracy/a1", self.a1)
            _tf.summary.scalar("accuracy/a2", self.a2)
            _tf.summary.scalar("accuracy/a3", self.a3)

            [_tf.summary.histogram(var.op.name, var) for var in self.trainable_variables]

            path_to_summaries = _os.path.join(
                _FLAGS.log_dir,
                _FLAGS.type,
                _FLAGS.feature_extractor if _FLAGS.use_pretrained_data else _FLAGS.feature_extractor + "_scratch",
                "tensorboard"
            )
            if not _os.path.exists(path_to_summaries):
                _os.makedirs(path_to_summaries)
            if self.GLOBAL_STEP.eval() == 0:
                filesindir = _os.listdir(path_to_summaries)
                for file in filesindir:
                    try:
                        delfile = _os.path.join(path_to_summaries, file)
                        shutil.rmtree(delfile, ignore_errors=True)
                    except:
                        _logging.warning('Warning, Tensorboard file not removed')
            self.train_summary_writer = _tf.summary.FileWriter(_os.path.join(path_to_summaries, "train"),
                                                               self.sess.graph)
            self.valid_summary_writer = _tf.summary.FileWriter(_os.path.join(path_to_summaries, "valid"))
            self.test_summary_writer = _tf.summary.FileWriter(_os.path.join(path_to_summaries, "test"))
            self.summary_op = _tf.summary.merge_all()
            _logging.info("Run Tensorboard!")
            _logging.info("tensorboard --logdir=%s" % _os.path.abspath(path_to_summaries))

    def learn(self, max_number_of_steps):
        while self.GLOBAL_STEP.eval() < max_number_of_steps:
            train_rgb, train_gt = self.dataset_loader.get_stream_batch(self.sess, is_training=True)
            (N, H, W, C) = train_gt.shape

            assert H == _FLAGS.dim_input_h and W == _FLAGS.dim_input_w

            _, loss_val, global_step = self.sess.run([self.train, self.loss, self.GLOBAL_STEP],
                                                     feed_dict={self.INPUTS: train_rgb, self.OUTPUTS: train_gt,
                                                                self.IS_TRAINING: True})

            _logging.info(
                "[{:5d}]\tloss_ {:.3f}".format(global_step, loss_val))

            if _FLAGS.use_sampling and global_step % _FLAGS.sampling_interval == 0:
                for sampling_dir, flag_stream in [(_FLAGS.learn_sampling_dir, _FLAGS.trainset_stream),
                                                  (_FLAGS.valid_sampling_dir, _FLAGS.validset_stream),
                                                  (_FLAGS.test_sampling_dir, _FLAGS.testset_stream)]:
                    self.save_samples(sampling_dir, flag_stream, global_step)

            if _FLAGS.use_tensorboard and global_step % _FLAGS.tensorboard_interval == 0:
                for summary_writer, flag_stream, is_training in [
                    (self.train_summary_writer, _FLAGS.trainset_stream, True),
                    (self.valid_summary_writer, _FLAGS.validset_stream, False),
                    (self.test_summary_writer, _FLAGS.testset_stream, False)
                ]:
                    self.save_board_logs(summary_writer, flag_stream, is_training)

            if global_step % _FLAGS.backup_every_n_steps == 0:
                self.backup()

    def evaluate(self):
        for suffix, flag_stream, dataset_size in [
            # ("valid", _FLAGS.validset_stream, _FLAGS.validset_size),
            ("test", _FLAGS.testset_stream, _FLAGS.testset_size)
        ]:
            table = _pt(
                ["a1", "a2", "a3", "rmse", "rmse_log", "irmse", "abs_rel", "sq_rel", "silog"])
            table.float_format = '.3'
            values = _np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype(_np.float64)
            max_iterations = int(dataset_size / _FLAGS.batch_size)
            max_iterations = 1 if max_iterations == 0 else max_iterations
            for iter in range(0, max_iterations):
                rgb, gt = self.dataset_loader.get_stream_batch(self.sess, stream_name=flag_stream)
                (N, H, W, C) = gt.shape

                assert H == _FLAGS.dim_dataset_h and W == _FLAGS.dim_dataset_w

                ord_score = _np.zeros((N, H, W, C), dtype=_np.float32)
                decoded_gt = _np.zeros((N, H, W, C), dtype=_np.float32)
                counts = _np.zeros((N, H, W, C), dtype=_np.float32)

                mask = _np.logical_and(gt > _FLAGS.GT_minima, gt <= _FLAGS.GT_maxima)

                start_time = time.time()
                for i in range(0, _FLAGS.num_frag):
                    h0 = 0
                    h1 = _FLAGS.dim_input_h
                    w0 = i * int(_FLAGS.dim_input_w / 2)
                    w1 = w0 + _FLAGS.dim_input_w
                    if w1 > W:
                        w0 = W - _FLAGS.dim_input_w
                        w1 = W
                    _frag_rgb = rgb[:, h0:h1, w0:w1, :]
                    _frag_gt = gt[:, h0:h1, w0:w1, :]
                    _decoded_nn_val, _decoded_gt_val = self.sess.run([self.decoded_nn_out, self.decoded_gt],
                                                                     feed_dict={self.INPUTS: _frag_rgb,
                                                                                self.OUTPUTS: _frag_gt,
                                                                                self.IS_TRAINING: False})

                    ord_score[:, h0:h1, w0:w1, :] += _decoded_nn_val
                    decoded_gt[:, h0:h1, w0:w1, :] += _decoded_gt_val
                    counts[:, h0:h1, w0:w1, :] += 1.0
                print("\n--- %s seconds ---\n" % (time.time() - start_time))
                ord_score = ord_score / counts - _FLAGS.dsc_shift
                decoded_gt = decoded_gt / counts - _FLAGS.dsc_shift
                new_w_start = 0.0359477 * W
                new_w_end = 0.96405229 * W
                new_h_start = 0.3324324 * H
                new_h_end = 0.91351351 * H
                crop = _np.array([new_h_start, new_h_end,
                                  new_w_start, new_w_end]).astype(_np.int32)
                crop_mask = _np.zeros(mask.shape)
                crop_mask[:, crop[0]:crop[1], crop[2]:crop[3], :] = 1
                mask = _np.logical_and(mask, crop_mask)

                cropped_pred = ord_score[:, crop[0]:crop[1], crop[2]:crop[3], :]
                cropped_gt = decoded_gt[:, crop[0]:crop[1], crop[2]:crop[3], :]
                cropped_pred[cropped_pred < _FLAGS.GT_minima] = _FLAGS.GT_minima
                cropped_pred[cropped_pred > _FLAGS.GT_maxima] = _FLAGS.GT_maxima
                cropped_gt[cropped_gt < _FLAGS.GT_minima] = _FLAGS.GT_minima
                cropped_gt[cropped_gt > _FLAGS.GT_maxima] = _FLAGS.GT_maxima

                for i in range(0, 3):
                    _ord_score = ord_score[i]
                    _decoded_gt = decoded_gt[i]
                    _mask = mask[i]

                    a1, a2, a3, rmse, rmse_log, irmse, abs_rel, sq_rel, silog = self._compute_erros(_ord_score,
                                                                                                    _decoded_gt,
                                                                                                    _mask)
                    values[0] += a1
                    values[1] += a2
                    values[2] += a3
                    values[3] += rmse
                    values[4] += rmse_log
                    values[5] += irmse
                    values[6] += abs_rel
                    values[7] += sq_rel
                    values[8] += silog
                    table.add_row([a1, a2, a3, rmse, rmse_log, irmse, abs_rel, sq_rel, silog])
                print("\r%s evaluating... %.1f%s" % (suffix, iter / max_iterations * 100, "%"), end="")
                print("\r")
            table.add_row((values / (max_iterations * 3)).tolist())
            _logging.info("\n::::EVALUATE:::: %s\n%s" % (suffix, str(table)))

    def restore(self):
        def _restore_pretrained_variables():
            ckpt_path = _os.path.join(_FLAGS.pretrained_meta_dir, _FLAGS.feature_extractor)
            ckpt = _tf.train.latest_checkpoint(ckpt_path)
            if ckpt is None:
                raise ValueError("Pre-trained model is not exist")
            self.pretrain_restorer.restore(self.sess, ckpt)
            _logging.info("Pre-trained variables are restored")

        def _restore_trainable_variables():
            ckpt_path = _os.path.join(_FLAGS.meta_dir, _FLAGS.type, _FLAGS.feature_extractor)
            if not _FLAGS.use_pretrained_data:
                ckpt_path += "_scratch"
            ckpt = _tf.train.latest_checkpoint(ckpt_path)
            if ckpt:
                self.trainable_restorer.restore(self.sess, ckpt)
                _logging.info("Trainable variables are restored...")
            else:
                _logging.warning("Meta data of trainable variables doesn't exist")

        if _FLAGS.use_pretrained_data:
            _logging.debug("\t====Pre-trained variables====")
            [_logging.debug("\t-%s" % var.name.split(':')[0]) for var in self.pretrained_variables]
            self.pretrain_restorer = _tf.train.Saver(self.pretrained_variables)

        _logging.debug("\t====Trainable variables====")
        [_logging.debug("\t+%s" % var.name.split(':')[0]) for var in self.trainable_variables]
        self.trainable_restorer = _tf.train.Saver(self.trainable_variables)

        if _FLAGS.use_pretrained_data:
            _restore_pretrained_variables()

        if _FLAGS.restore:
            _restore_trainable_variables()

        pass

    def backup(self):
        def _backup_trainable_variables():
            path = _os.path.join(_FLAGS.meta_dir, _FLAGS.type, _FLAGS.feature_extractor)
            if not _FLAGS.use_pretrained_data:
                path += "_scratch"
            if not _gfile.Exists(path):
                _gfile.MakeDirs(path)

            _tf.train.write_graph(self.sess.graph.as_graph_def(), path, "graph.pb", as_text=False)
            self.trainable_restorer.save(self.sess, _os.path.join(path, "model.ckpt"),
                                         global_step=self.GLOBAL_STEP)
            _logging.info("Meta data is saved")

        if _FLAGS.backup:
            _backup_trainable_variables()
        pass

    def save_samples(self, sampling_dir, flag_stream, step):
        path = _os.path.join(
            _FLAGS.sampling_dir,
            _FLAGS.type,
            _FLAGS.feature_extractor if _FLAGS.use_pretrained_data else _FLAGS.feature_extractor + "_scratch",
            sampling_dir,
            "step_%05d" % (step)
        )
        rgb, gt = self.dataset_loader.get_stream_batch(self.sess, stream_name=flag_stream)
        (data_N, data_H, data_W, data_C) = gt.shape

        if str(_FLAGS.type).startswith('NYU') or _FLAGS.num_frag == 1:
            assert data_H == _FLAGS.dim_output_h and data_W == _FLAGS.dim_output_wadd >
            (out_N, out_H, out_W, out_C) = (data_N, _FLAGS.dim_output_h, _FLAGS.dim_output_w, data_C)
        elif str(_FLAGS.type).startswith('KITTI'):
            assert data_H == _FLAGS.dim_dataset_h and data_W == _FLAGS.dim_dataset_w
            (out_N, out_H, out_W, out_C) = (data_N, data_H, data_W, data_C)
        else:
            (out_N, out_H, out_W, out_C) = (data_N, data_H, data_W, data_C)

        ord_score = _np.zeros((out_N, out_H, out_W, out_C), dtype=_np.float32)
        decoded_gt = _np.zeros((out_N, out_H, out_W, out_C), dtype=_np.float32)
        counts = _np.zeros((out_N, out_H, out_W, out_C), dtype=_np.float32)
        for i in range(0, _FLAGS.num_frag):
            (in_h0, in_h1) = (0, _FLAGS.dim_input_h)
            (out_h0, out_h1) = (0, _FLAGS.dim_output_h)
            in_w0 = i * int(_FLAGS.dim_input_w / 2)
            in_w1 = in_w0 + _FLAGS.dim_input_w
            out_w0 = i * int(_FLAGS.dim_output_w / 2)
            out_w1 = out_w0 + _FLAGS.dim_output_w
            if in_w1 > data_W:
                (in_w0, in_w1) = (data_W - _FLAGS.dim_input_w, data_W)
                (out_w0, out_w1) = (out_W - _FLAGS.dim_output_w, out_W)
            _frag_rgb = rgb[:, in_h0:in_h1, in_w0:in_w1, :]
            _frag_gt = gt[:, in_h0:in_h1, in_w0:in_w1, :]
            _decoded_nn_val, _decoded_gt_val = self.sess.run([self.decoded_nn_out, self.decoded_gt],
                                                             feed_dict={self.INPUTS: _frag_rgb,
                                                                        self.OUTPUTS: _frag_gt,
                                                                        self.IS_TRAINING: False})
            ord_score[:, out_h0:out_h1, out_w0:out_w1, :] += _decoded_nn_val
            decoded_gt[:, out_h0:out_h1, out_w0:out_w1, :] += _decoded_gt_val
            counts[:, out_h0:out_h1, out_w0:out_w1, :] += 1.0

        ord_score = ord_score / counts - _FLAGS.dsc_shift
        decoded_gt = decoded_gt / counts - _FLAGS.dsc_shift
        self._save_batch(path, rgb, gt, decoded_gt, ord_score)
        _logging.info(".......Samples are saved")

    def save_board_logs(self, summary_writer, flag_stream, is_training=False):
        rgb, gt = self.dataset_loader.get_stream_batch(self.sess, stream_name=flag_stream, is_tensorboard=True)

        summary_str = self.sess.run(self.summary_op,
                                    feed_dict={self.INPUTS: rgb,
                                               self.OUTPUTS: gt,
                                               self.IS_TRAINING: is_training})
        summary_writer.add_summary(summary_str, global_step=self.GLOBAL_STEP.eval())

    def _save_batch(self, path, b_rgb, b_gt, b_dcded_gt, b_predicted):
        if not _gfile.Exists(path):
            _gfile.MakeDirs(path)

        [b_gt, b_dcded_gt, b_predicted] = _np.squeeze([b_gt, b_dcded_gt, b_predicted])

        for i, (rgb, gt, dcded_gt, predicted) in enumerate(zip(b_rgb, b_gt, b_dcded_gt, b_predicted)):
            image_name = _os.path.relpath(_os.path.join(path, "%05d_input.jpg" % i),
                                          _os.getcwd())  # Get relative path for opencv windows
            cv2.imwrite(image_name, cv2.cvtColor(_np.uint8(rgb), cv2.COLOR_RGB2BGR))

            predicted[predicted < 0.0] = 0.0
            depth_name = _os.path.relpath(_os.path.join(path, "%05d_depth.png" % i), _os.getcwd())
            if str(_FLAGS.type).startswith("NYU"):
                cv2.imwrite(depth_name, _np.uint8(predicted))
            else:
                cv2.imwrite(depth_name, _np.uint16(predicted * 256.0))

            color_depth_name = "%s/%05d_depth_color.png" % (path, i)
            if str(_FLAGS.type).startswith("NYU"):
                _plt.imsave(color_depth_name, _np.uint8(predicted / _FLAGS.GT_maxima * 256.0), cmap="jet")
            else:
                _plt.imsave(color_depth_name, _np.uint8(predicted), cmap="jet")

            gt[gt < 0.0] = 0.0
            gt_name = _os.path.relpath(_os.path.join(path, "%05d_gt.png" % i), _os.getcwd())
            if str(_FLAGS.type).startswith("NYU"):
                cv2.imwrite(gt_name, _np.uint8(gt))
            else:
                cv2.imwrite(gt_name, _np.uint16(gt * 256.0))

            dcded_gt[dcded_gt < 0.0] = 0.0
            dcded_gt_name = _os.path.relpath(_os.path.join(path, "%05d_gt_dcd.png" % i), _os.getcwd())
            if str(_FLAGS.type).startswith("NYU"):
                cv2.imwrite(dcded_gt_name, _np.uint8(dcded_gt))
            else:
                cv2.imwrite(dcded_gt_name, _np.uint16(dcded_gt * 256.0))

                color_dcded_gt_name = _os.path.join(path, "%05d_gt_dcd_color.png" % i)
            if str(_FLAGS.type).startswith("NYU"):
                _plt.imsave(color_dcded_gt_name, _np.uint8(dcded_gt / _FLAGS.GT_maxima * 256.0), cmap="jet")
            else:
                _plt.imsave(color_dcded_gt_name, _np.uint8(dcded_gt), cmap="jet")

    def _compute_erros(self, pred, gt, mask):
        pred[pred < _FLAGS.GT_minima] = _FLAGS.GT_minima
        pred[pred > _FLAGS.GT_maxima] = _FLAGS.GT_maxima
        pred = pred[mask]
        gt = gt[mask]
        npix = _np.sum(mask)

        threshold = _np.maximum((pred / gt), (gt / pred))
        a1 = (npix - _np.sum(_np.greater_equal(threshold, 1.25).astype(_np.float32))) / npix
        a2 = (npix - _np.sum(_np.greater_equal(threshold, 1.25 ** 2).astype(_np.float32))) / npix
        a3 = (npix - _np.sum(_np.greater_equal(threshold, 1.25 ** 3).astype(_np.float32))) / npix

        rmse = _np.sum((gt - pred) ** 2) / npix
        rmse = _np.sqrt(rmse)

        rmse_log = _np.sum((_np.log(gt) - _np.log(pred)) ** 2) / npix
        rmse_log = _np.sqrt(rmse_log)

        irmse = _np.sqrt(_np.sum((1 / pred - 1 / gt) ** 2) / npix)

        abs_rel = _np.sum(_np.abs(gt - pred) / gt) / npix

        sq_rel = _np.sum(((gt - pred) ** 2) / gt) / npix

        d = _np.log(pred) - _np.log(gt)
        silog = (_np.sum(d ** 2) / npix) - (((_np.sum(d)) ** 2) / (npix ** 2))

        return a1, a2, a3, rmse, rmse_log, irmse, abs_rel, sq_rel, silog
