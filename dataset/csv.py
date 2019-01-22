import tensorflow as _tf
import os as _os
import glob as _glob
from tensorflow.python.platform import gfile as _gfile
from utils import logutil as _logutil

_FLAGS = _tf.app.flags.FLAGS
_logging = _logutil.get_logger()


def inspect():
    for key, value in _FLAGS.__flags.items():
        if str(key).endswith('csv'):
            csv_file = value._value
            csv_path = _os.path.join(_FLAGS.dataset_dir, _FLAGS.type, csv_file)
            if not _gfile.Exists(csv_path):
                _generate(key, csv_path)


def _generate(key, csv_path):
    _logging.info("Generating... %s", key)
    ext_rgb = _FLAGS.ext_rgb
    ext_gt = _FLAGS.ext_gt

    base_path = _os.path.join(
        _FLAGS.dataset_dir,
        _FLAGS.type,
        _FLAGS[str(key).replace('csv', 'img_dir')]._value
    )

    scenes = [_os.path.join(base_path, name) for name in _os.listdir(base_path) if
              _os.path.isdir(_os.path.join(base_path, name))]

    if  str(_FLAGS.type).startswith("KITTI") and str(_FLAGS.type).find("eigen") > 0:
        _recontructed = []
        for scene in scenes:
            _recontructed.extend([_os.path.join(scene, drive) for drive in _os.listdir(scene)])
        scenes = _recontructed

    if len(scenes) < 1:
        scenes = [base_path]

    for idx, scene in enumerate(sorted(scenes)):
        rgb_path = scene
        gt_path = scene
        if str(_FLAGS.type).startswith("KITTI"):
            rgb_path = _os.path.join(scene, "proj_depth", "image", _FLAGS.KITTI_side)
            gt_path = _os.path.join(scene, "proj_depth", "groundtruth", _FLAGS.KITTI_side)
            if str(_FLAGS.type).find("eigen") > 0:
                rgb_path = _os.path.join(scene,  _FLAGS.KITTI_side, "data")
                gt_path = _os.path.join(scene,  _FLAGS.KITTI_side, "groundtruth")
        rgb_files = _glob.glob(_os.path.join(rgb_path, '*.' + _FLAGS.ext_rgb))
        gt_files = _glob.glob(_os.path.join(gt_path, '*.' + _FLAGS.ext_gt))

        rgb_set = set(map(lambda item: _os.path.basename(item).split('.')[0], rgb_files))
        gt_set = set(map(lambda item: _os.path.basename(item).split('.')[0], gt_files))

        target_set = rgb_set & gt_set
        invalid_rgbs = rgb_set - gt_set
        invalid_gts = gt_set - rgb_set

        with open(csv_path, 'a') as output:
            for i, item in enumerate(target_set):
                input_path, depth_path = _get_item_pair(rgb_path, gt_path, item, ext_rgb, ext_gt)
                output.write("%s,%s" % (input_path, depth_path))
                output.write("\n")

        if invalid_rgbs.__len__() > 0:
            _logging.warning('===Remove mismatched RGB data /*.%s', ext_rgb)
            _logging.warning(
                '\n'.join('\t\t' + str(inv) for inv in map(lambda item: "%s.%s" % (item, ext_rgb), invalid_rgbs)))
        if invalid_gts.__len__() > 0:
            _logging.warning('===Remove mismatched GT data /*.%s', ext_gt)
            _logging.warning(
                '\n'.join('\t\t' + str(inv) for inv in map(lambda item: "%s.%s" % (item, ext_gt), invalid_gts)))
    pass


def _get_item_pair(rgb_base, gt_base, item, ext_rgb, ext_gt):
    return _os.path.join(rgb_base, "%s.%s" % (item, ext_rgb)), _os.path.join(gt_base, "%s.%s" % (item, ext_gt))
