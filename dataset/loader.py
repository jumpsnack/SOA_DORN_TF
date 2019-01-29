import tensorflow as _tf
import dataset.csv as csv
import os as _os
from utils import logutil as _logutil
from . import augmentation as _augmentation

_tde = _tf.data.experimental
_logging = _logutil.get_logger()
_FLAGS = _tf.app.flags.FLAGS


def load(machine=None):
    csv.inspect()
    if machine is not None:
        machine.dataset_loader = Loader()
    else:
        return Loader()


class Loader():
    def __init__(self):
        self._eval_streams = {}
        self._tensorboard_streams = {}
        self._training_stream = None
        self._generate_stream()
        pass

    def _resize(self, stream):
        x, y = stream
        x = _tf.image.resize_images(x, (_FLAGS.dim_input_h, _FLAGS.dim_input_w))
        y = _tf.image.resize_images(y, (_FLAGS.dim_output_h, _FLAGS.dim_output_w))
        return x, y

    def _random_crop(self, stream):
        x, y = stream
        (W, H, _) = x.shape
        assert (W, H) == (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w)
        size_h, size_w = (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w)
        if size_h > _FLAGS.dim_input_h or size_w > _FLAGS.dim_input_w:
            off_w = _tf.cast(_tf.random_uniform([], 0, size_w - _FLAGS.dim_input_w), _tf.int32)
            off_h = _tf.cast(_tf.random_uniform([], 0, size_h - _FLAGS.dim_input_h), _tf.int32)

            try:
                _x = _tf.image.crop_to_bounding_box(x, offset_height=off_h, offset_width=off_w,
                                                    target_width=_FLAGS.dim_input_w,
                                                    target_height=_FLAGS.dim_input_h)
                _y = _tf.image.crop_to_bounding_box(y, offset_height=off_h, offset_width=off_w,
                                                    target_width=_FLAGS.dim_input_w,
                                                    target_height=_FLAGS.dim_input_h)
                x = _x
                y = _y
            except Exception as e:
                print(e)

        return self._resize((x, y))

    def _center_crop(self, stream):
        x, y = stream
        (W, H, _) = x.shape
        assert (W, H) == (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w)
        size_h, size_w = (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w)
        off_w = _tf.cast((size_w - _FLAGS.dim_input_w) / 2.0, _tf.int32)
        off_h = _tf.cast(size_h - _FLAGS.dim_input_h, _tf.int32)

        try:
            _x = _tf.image.crop_to_bounding_box(x, offset_height=off_h, offset_width=off_w,
                                                target_width=_FLAGS.dim_input_w,
                                                target_height=_FLAGS.dim_input_h)
            _y = _tf.image.crop_to_bounding_box(y, offset_height=off_h, offset_width=off_w,
                                                target_width=_FLAGS.dim_input_w,
                                                target_height=_FLAGS.dim_input_h)
            x = _x
            y = _y
        except Exception as e:
            print(e)

        return self._resize((x, y))

    def _generate_stream(self):
        def _config_batch_stream(csv_path, stream_bufsiz, augmentation_fn=None, pre_process_fn=None):
            data_pool = _tf.data.TextLineDataset([csv_path])
            data_pool = data_pool.apply(_tde.shuffle_and_repeat(stream_bufsiz))
            data_pool = data_pool.apply(_tde.map_and_batch(
                map_func=lambda row: self._read_row(row, augmentation_fn, pre_process_fn),
                batch_size=batch_size,
                num_parallel_batches=_FLAGS.num_parallel_batches
            ))
            batch_stream = data_pool.make_one_shot_iterator().get_next()
            return batch_stream

        for key in [_key for _key in _FLAGS if str(_key).endswith('csv')]:
            csv_file = _FLAGS[key]._value
            csv_path = _os.path.join(_FLAGS.dataset_dir, _FLAGS.type,
                                     csv_file)
            key = str(key).split('_')[0]

            batch_size = _FLAGS.batch_size

            augs = []
            if key.find('train') >= 0:
                for _key in [_key for _key in _FLAGS]:
                    if str(_key).startswith('augment'):
                        aug_flag = _FLAGS[_key]._value
                        if aug_flag:
                            augs.append(str(_key).split('_')[1])
                self._training_stream = _config_batch_stream(csv_path, _FLAGS.buffer_size, augs, self._random_crop)

            print("Stream... " + key)
            self._eval_streams[key] = _config_batch_stream(csv_path, 1)

            if _FLAGS.use_tensorboard:
                print("Stream for tensorboard... " + key)
                self._tensorboard_streams[key] = _config_batch_stream(csv_path, 1, pre_process_fn=self._center_crop)

    def _read_row(self, csv_row, augs, pre_processing=None):
        rgb_name, gt_name = _tf.decode_csv(csv_row, record_defaults=[[""], [""]])
        rgb_img = self._load_img(rgb_name, channels=3)
        gt_img = self._load_img(gt_name)

        (rgb_img, gt_img) = _augmentation.augment((rgb_img, gt_img), augs)
        if pre_processing is not None:
            (rgb_img, gt_img) = pre_processing((rgb_img, gt_img))
        return rgb_img, gt_img

    def _load_img(self, filename, channels=1):
        file = _tf.read_file(filename)
        if channels == 3:
            image = _tf.image.decode_jpeg(file, channels)
            image = _tf.cast(image, _tf.float32)
        elif channels == 1:

            if str(_FLAGS.type).startswith("KITTI"):
                # ===================================
                #           process KITTI
                # ===================================
                image = _tf.image.decode_png(file, channels, dtype=_tf.uint16)
                image = _tf.cast(image, _tf.float32)
                image = image / 256.0
                image = _tf.where(_tf.less_equal(image, _FLAGS.GT_minima), -_tf.ones_like(image), image)
                image = _tf.where(_tf.greater_equal(image, _FLAGS.GT_maxima), _tf.ones_like(image) * _FLAGS.GT_maxima,
                                  image)
            elif str(_FLAGS.type).startswith("NYU"):
                # ===================================
                #           process NYU
                # ===================================
                image = _tf.image.decode_png(file, channels, dtype=_tf.uint16)
                image = _tf.cast(image, _tf.float32)
                image = image / 256.0
                image = _tf.where(_tf.less_equal(image, _FLAGS.GT_minima), -_tf.ones_like(image), image)
                image = _tf.where(_tf.greater_equal(image, _FLAGS.GT_maxima), _tf.ones_like(image) * _FLAGS.GT_maxima,
                                  image)
            else:
                image = _tf.image.decode_png(file, channels)
                image = _tf.cast(image, _tf.float32)
        else:
            return
        image = _tf.image.resize_images(image,
                                        (_FLAGS.dim_dataset_h, _FLAGS.dim_dataset_w))
        return image

    def get_stream_names(self):
        return list(self._eval_streams.keys())

    def get_stream_batch(self, sess, **kwargs):
        is_training = kwargs.get('is_training', False)
        is_tensorboard = kwargs.get('is_tensorboard', False)
        stream_name = kwargs.get('stream_name', '')
        try:
            if is_training:
                assert self._training_stream is not None
                stream = self._training_stream
            elif is_tensorboard:
                assert stream_name is not '' and self._tensorboard_streams[stream_name] is not None
                stream = self._tensorboard_streams[stream_name]
            else:
                assert stream_name is not '' and self._eval_streams[stream_name] is not None
                stream = self._eval_streams[stream_name]
        except KeyError:
            if sess is not None:
                sess.close()
            raise _tf.errors.InternalError(None, None, message="Stream named '%s' doesn't exists" % stream_name)

        if sess is not None:
            for retries in range(1, 6):
                try:
                    input, label = sess.run(stream)
                except KeyboardInterrupt:
                    sess.close()
                    raise _tf.errors.CancelledError(None, None, message="KeyboardInterrupt")
                except:
                    _logging.warning('\nGet trainset on batch', 'retry...({})'.format(retries))
                    continue
                else:
                    return input, label
            raise _tf.errors.DataLossError(None, None, message='Retries expired')
        else:
            return stream

        pass
