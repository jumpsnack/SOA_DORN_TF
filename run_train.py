from utils import confutil as _confuti
from utils import logutil as _logutil

import tensorflow as tf
import os
import machine
import dataset
import models

_logging = _logutil.get_logger()
_FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as tf_session:
        depth_predict_machine = machine.create(tf_session, models.builder, dataset.loader)
        depth_predict_machine.learn(_FLAGS.max_number_of_steps)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.app.run()
