import copy
from xgboost import XGBClassifier
from library.config import Config


class ModelFactory:
    """
    Factory class for creating XGBoost classifiers for the Dual-Stream architecture.

    This factory encapsulates the creation logic for the two distinct modeling streams:
    - Stream A (The Collider): Optimized for complex Player-Player interactions using
      absolute kinematics and relative geometry.
    - Stream B (The Accelerometer): Optimized for Player-Ground impacts using
      strict biomechanical invariants and ego-dynamics.
    """

    @staticmethod
    def get_classifier(stream, **kwargs):
        """
        Creates and returns an XGBClassifier configured for the specified stream.

        Args:
            stream (str): The stream identifier.
                          'A' for Stream A (Collider/Player-Player).
                          'B' for Stream B (Accelerometer/Player-Ground).
            **kwargs: Additional arguments to override the default configuration
                      (e.g., n_estimators, learning_rate).

        Returns:
            XGBClassifier: The configured XGBoost classifier instance ready for training.

        Raises:
            ValueError: If an invalid stream identifier is provided.
        """
        # Determine which configuration to use based on the stream identifier
        if stream == "A":
            # Stream A: The Collider
            # Uses parameters optimized for complex feature interactions (deeper depth)
            # defined in Config.XGB_PARAMS_STREAM_A
            params = copy.deepcopy(Config.XGB_PARAMS_STREAM_A)

        elif stream == "B":
            # Stream B: The Accelerometer
            # Uses parameters optimized for simple physical invariants (shallower depth, higher regularization)
            # defined in Config.XGB_PARAMS_STREAM_B
            params = copy.deepcopy(Config.XGB_PARAMS_STREAM_B)

        else:
            raise ValueError(
                f"Invalid stream identifier '{stream}'. Expected 'A' or 'B'."
            )

        # Apply any runtime overrides provided via kwargs
        if kwargs:
            params.update(kwargs)

        # Instantiate the XGBClassifier
        # The configuration in Config already includes 'device': 'cuda' and 'tree_method': 'hist'
        # to ensure GPU acceleration is utilized.
        clf = XGBClassifier(**params)

        return clf
