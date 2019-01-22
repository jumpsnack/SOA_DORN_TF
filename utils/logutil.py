import os as _os
import tensorflow as _tf
from time import gmtime, strftime
import logging
import logging.handlers

_logger = None
_FLAGS = _tf.app.flags.FLAGS


def _generate_logger():
    # log_format = "%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s"
    log_format = "%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s"

    logger = logging.getLogger('default')
    logger.setLevel(logging.DEBUG)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(streamHandler)

    path_to_feature_extractor = _FLAGS.feature_extractor
    if not _FLAGS.use_pretrained_data:
        path_to_feature_extractor += "_scratch"
    log_path = _os.path.join(
        _FLAGS.log_dir,
        _FLAGS.type,
        path_to_feature_extractor
    )
    if not _os.path.exists(log_path):
        _os.makedirs(log_path)

    log_path = _os.path.join(log_path, strftime("%Y-%m-%d_%H;%M;%S", gmtime()) + '.log')
    fileHandler = logging.FileHandler(log_path)
    logger.addHandler(fileHandler)
    return logger


def get_logger():
    global _logger
    if not _logger:
        _logger = _generate_logger()
    return _logger
