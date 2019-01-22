from utils import logutil
import tensorflow as _tf
from ._machine import _Machine

_logging = logutil.get_logger()
_FLAGS = _tf.app.flags.FLAGS


def create(tf_session, model_builder, dataset_loader):
    assert tf_session is not None
    assert model_builder is not None
    assert dataset_loader is not None

    _logging.debug("Machine created!...")

    return _Machine(tf_session, model_builder, dataset_loader)
