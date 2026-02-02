import pandas as pd
import numpy as np
from library.utils import setup_logger

logger = setup_logger("encoders")


class MultiClassTargetEncoder:
    """
    Multi-class Target Encoder with Bayesian Smoothing.

    This encoder replaces categorical variables with the posterior probability
    of each target class, given the categorical level. It uses Bayesian smoothing
    (m-estimate) to regularize estimates for rare categories towards the global prior.
    """

    def __init__(self, columns=None, smoothing=10.0):
        """
        Initializes the encoder.

        Args:
            columns (list of str): The names of the categorical columns to encode.
            smoothing (float): The smoothing parameter 'm'. Higher values trust the
                               global prior more, useful for categories with few samples.
        """
        self.columns = columns if columns is not None else []
        self.smoothing = float(smoothing)
        self.maps = {}
        self.global_priors = None
        self.classes = None
        self.fitted = False

    def fit(self, X, y):
        """
        Fits the encoder on the training data.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series or np.ndarray): The target class labels.

        Returns:
            self
        """
        logger.info("Fitting MultiClassTargetEncoder...")

        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index)

        # Identify target classes
        self.classes = np.sort(y.unique())

        # Compute Global Priors: P(Class)
        # These are used when a category level is unknown or has very few samples
        global_counts = y.value_counts().reindex(self.classes, fill_value=0)
        self.global_priors = (global_counts / len(y)).astype(np.float32)

        for col in self.columns:
            if col not in X.columns:
                logger.warning(f"Column '{col}' not found in training data. Skipping.")
                continue

            # Create a temporary DataFrame for aggregation
            # We use the index of X to align with y
            temp_df = pd.DataFrame({"feature": X[col], "target": y})

            # 1. Calculate N (total count per category level)
            cat_counts = temp_df.groupby("feature")["target"].count()

            # 2. Calculate n_c (count of each class per category level)
            # Result is a DataFrame with Index=FeatureValue, Columns=ClassLabel
            class_counts = (
                temp_df.groupby(["feature", "target"]).size().unstack(fill_value=0)
            )

            # Ensure all classes exist in the columns
            class_counts = class_counts.reindex(columns=self.classes, fill_value=0)

            # 3. Calculate Smoothed Probabilities
            # Formula: P_est = (n_c + m * prior) / (N + m)

            mapping_df = pd.DataFrame(index=class_counts.index, columns=self.classes)

            for cls in self.classes:
                prior = self.global_priors[cls]
                n_c = class_counts[cls]
                # N is cat_counts. We rely on index alignment between class_counts and cat_counts

                # Vectorized calculation
                numerator = n_c + (self.smoothing * prior)
                denominator = cat_counts + self.smoothing

                mapping_df[cls] = numerator / denominator

            # Store the mapping (cast to float32 to save memory)
            self.maps[col] = mapping_df.astype(np.float32)

        self.fitted = True
        logger.info(f"Encoder fitted on {len(self.maps)} columns.")
        return self

    def transform(self, X):
        """
        Transforms the input data using the fitted encodings.

        Args:
            X (pd.DataFrame): The input features to transform.

        Returns:
            pd.DataFrame: A new DataFrame with the original features plus the
                          new target-encoded features.
        """
        if not self.fitted:
            raise RuntimeError("Encoder must be fitted before calling transform.")

        # Avoid modifying original dataframe
        X_out = X.copy()

        for col in self.columns:
            if col not in self.maps:
                continue

            mapping = self.maps[col]

            for cls in self.classes:
                # Construct new feature name
                new_col_name = f"{col}_target_{cls}"

                # Get the probability map for this class
                # It is a Series where index=category_value, value=probability
                prob_map = mapping[cls]

                # Map the column values to probabilities
                # .map() handles the lookup
                # .fillna() handles unknown categories (values not seen in fit) using the global prior
                prior = self.global_priors[cls]
                encoded_col = X_out[col].map(prob_map).fillna(prior)

                X_out[new_col_name] = encoded_col.astype(np.float32)

        return X_out
