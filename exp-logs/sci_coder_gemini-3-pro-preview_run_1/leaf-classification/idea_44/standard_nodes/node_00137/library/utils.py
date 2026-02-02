import json
import hashlib
import numpy as np
from library.config import set_seed


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy data types for configuration serialization.
    Ensures that numpy scalars and arrays can be hashed deterministically.
    """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def compute_config_hash(config):
    """
    Computes a deterministic MD5 hash of a configuration object.

    This function is used to generate unique cache keys based on feature engineering
    configurations, hyperparameters, or any other dictionary/list structure.
    It ensures that the same configuration always produces the same hash,
    regardless of memory address or key insertion order.

    Args:
        config (dict | list | Any): The configuration object to hash.

    Returns:
        str: A hexadecimal MD5 hash string.
    """
    # Serialize the configuration to a JSON string.
    # sort_keys=True ensures that dictionary key order doesn't affect the hash.
    # cls=NumpyEncoder handles any numpy types present in the config.
    config_str = json.dumps(config, sort_keys=True, cls=NumpyEncoder)

    # Compute and return the MD5 digest
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()
