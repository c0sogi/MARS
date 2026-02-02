import library.data_loader as dl
from library.data_loader import NeutrinoDataLoader


class FeatureEngineeringPipeline:
    """
    Pipeline for loading and preparing the Neutrino dataset features.

    This class acts as a wrapper around the NeutrinoDataLoader to provide
    a high-level interface for retrieving training, validation, and test sets.
    It supports dynamic configuration of debug sampling sizes and caching.
    """

    def __init__(self, debug_size=None):
        """
        Initialize the pipeline.

        Args:
            debug_size (int, optional): If provided, limits the number of events
                                        processed for debugging purposes.
        """
        self.loader = NeutrinoDataLoader()
        self.debug_size = debug_size

    def _apply_debug_config(self):
        """
        Apply the debug size configuration to the data loader module.
        This patches the global variable in the library module to control sampling.
        """
        if self.debug_size is not None:
            dl.DEBUG_SAMPLE_SIZE = self.debug_size
            print(
                f"[FeatureEngineering] Debug mode enabled. Limiting to {self.debug_size} events."
            )

    def get_train_data(self, load_cached_data=True):
        """
        Load the training dataset.

        Args:
            load_cached_data (bool): If True, attempts to load from cache.

        Returns:
            tuple: (X, y, ids) where X is the feature matrix, y is the target matrix,
                   and ids are the event IDs.
        """
        self._apply_debug_config()
        return self.loader.load_split("train", load_cached_data=load_cached_data)

    def get_val_data(self, load_cached_data=True):
        """
        Load the validation dataset.

        Args:
            load_cached_data (bool): If True, attempts to load from cache.

        Returns:
            tuple: (X, y, ids) where X is the feature matrix, y is the target matrix,
                   and ids are the event IDs.
        """
        self._apply_debug_config()
        return self.loader.load_split("val", load_cached_data=load_cached_data)

    def get_test_data(self, load_cached_data=True):
        """
        Load the test dataset.

        Args:
            load_cached_data (bool): If True, attempts to load from cache.

        Returns:
            tuple: (X, y, ids) where X is the feature matrix, y is None (for test),
                   and ids are the event IDs.
        """
        # Ensure test set is always full size for valid submission
        dl.DEBUG_SAMPLE_SIZE = None
        return self.loader.load_split("test", load_cached_data=load_cached_data)


def load_datasets(load_cached_data=True, debug_size=None):
    """
    Load all dataset splits (Train, Validation, Test) using the pipeline.

    Args:
        load_cached_data (bool): Whether to use cached data if available.
        debug_size (int, optional): Number of events to load for debugging.

    Returns:
        tuple: A tuple containing three tuples:
               ((X_train, y_train), (X_val, y_val), (X_test, ids_test))
    """
    pipeline = FeatureEngineeringPipeline(debug_size=debug_size)

    print("\n[FeatureEngineering] Loading Training Data...")
    X_train, y_train, _ = pipeline.get_train_data(load_cached_data=load_cached_data)

    print("\n[FeatureEngineering] Loading Validation Data...")
    X_val, y_val, _ = pipeline.get_val_data(load_cached_data=load_cached_data)

    print("\n[FeatureEngineering] Loading Test Data...")
    # Cite debug_lesson_3: Decouple Debug Sampling from Submission Generation.
    # Force reload of test data to prevent loading a stale, debug-sized cache.
    X_test, _, ids_test = pipeline.get_test_data(load_cached_data=False)

    return (X_train, y_train), (X_val, y_val), (X_test, ids_test)
