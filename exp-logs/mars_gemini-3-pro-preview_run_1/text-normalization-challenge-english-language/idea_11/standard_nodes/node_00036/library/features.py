import re
import os
import json
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import save_parquet, load_parquet, ensure_dir, setup_logger

# Initialize logger for this module
logger = setup_logger("features")


class RegexFeatureExtractor:
    """
    Extracts explicit binary morphological features using Regex patterns.
    These features help the model handle OOV tokens by identifying structure
    (e.g., digits, currency, capitalization).
    """

    def __init__(self):
        # Define regex patterns
        # We compile them for efficiency
        self.patterns = {
            "is_digit": re.compile(r"^\d+$"),
            "has_digit": re.compile(r"\d"),
            "is_alpha": re.compile(r"^[a-zA-Z]+$"),
            "is_upper": re.compile(r"^[A-Z]+$"),
            "is_title": re.compile(r"^[A-Z][a-z]+$"),
            "is_punct": re.compile(r"^[^\w\s]+$"),
            "is_currency": re.compile(r"[$£€¥¢₹]"),
            "is_decimal": re.compile(r"^\d+\.\d+$"),
            "has_comma": re.compile(r","),
            "is_time": re.compile(
                r"^\d{1,2}:\d{2}(:\d{2})?( ?[ap]\.?m\.?)?$", re.IGNORECASE
            ),
            "is_date_slash": re.compile(r"^\d{1,4}/\d{1,2}/\d{1,4}$"),
            "is_date_dash": re.compile(r"^\d{1,4}-\d{1,2}-\d{1,4}$"),
            "is_measure": re.compile(
                r"^(km|kg|mm|cm|mg|ml|ltr|lb|oz|ft|in|m|g|s)$", re.IGNORECASE
            ),
            "is_url": re.compile(r"(http|www|\.com|\.org|\.net)", re.IGNORECASE),
            "is_email": re.compile(r"\S+@\S+"),
            "is_roman": re.compile(
                r"^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$",
                re.IGNORECASE,
            ),
            "is_ordinal": re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE),
        }
        # Sort keys to ensure consistent vector ordering
        self.feature_names = sorted(list(self.patterns.keys()))
        self.dim = len(self.feature_names)

    def extract(self, text):
        """
        Extracts features for a single string.
        Returns a numpy array of shape (dim,).
        """
        if not isinstance(text, str):
            text = str(text)

        features = np.zeros(self.dim, dtype=np.float32)
        for i, name in enumerate(self.feature_names):
            if self.patterns[name].search(text):
                features[i] = 1.0
        return features

    def extract_batch(self, texts):
        """
        Extracts features for a list of strings.
        Returns a numpy array of shape (len(texts), dim).
        """
        results = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            results[i] = self.extract(text)
        return results

    def get_feature_dim(self):
        return self.dim


class GlobalPriorManager:
    """
    Manages global prior features: empirical class probabilities for each token
    derived from the training set.
    """

    def __init__(self):
        self.priors_map = {}  # token -> np.array
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.num_classes = 0
        self.vocab_classes_path = os.path.join(Config.VOCAB_DIR, "vocab_classes.json")

    def build_or_load(self, train_df, load_cached_data=True):
        """
        Loads priors from cache if available, otherwise builds them from train_df.

        Args:
            train_df (pd.DataFrame): Must contain 'before' and 'class' columns.
            load_cached_data (bool): Whether to attempt loading from disk.
        """
        # 1. Try to load class vocabulary first
        self._load_class_vocab()

        # 2. Try to load priors
        if load_cached_data and os.path.exists(Config.PRIORS_PATH):
            logger.info(f"Loading global priors from {Config.PRIORS_PATH}...")
            try:
                df_priors = load_parquet(Config.PRIORS_PATH)
                if df_priors is not None:
                    self._dataframe_to_map(df_priors)
                    logger.info(
                        f"Loaded priors for {len(self.priors_map)} unique tokens."
                    )
                    return
            except Exception as e:
                logger.warning(f"Failed to load priors: {e}. Rebuilding...")

        # 3. Build from scratch
        logger.info("Building global priors from training data...")
        self._build_priors(train_df)

        # 4. Save to cache
        logger.info(f"Saving global priors to {Config.PRIORS_PATH}...")
        self._save_priors()

    def get_prior(self, token):
        """
        Returns the prior probability vector for a given token.
        If token is OOV, returns a zero vector.
        """
        if token in self.priors_map:
            return self.priors_map[token]
        else:
            return np.zeros(self.num_classes, dtype=np.float32)

    def get_class_vocab(self):
        return self.class_to_idx

    def _build_priors(self, df):
        """
        Internal method to calculate priors from dataframe.
        """
        # Ensure strings
        df = df.copy()
        df["before"] = df["before"].astype(str)
        df["class"] = df["class"].astype(str)

        # 1. Define Class Vocabulary if not already loaded
        if not self.class_to_idx:
            unique_classes = sorted(df["class"].unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
            self.num_classes = len(unique_classes)
            self._save_class_vocab()

        # 2. Aggregation
        # Count (before, class) pairs
        counts = df.groupby(["before", "class"]).size().reset_index(name="count")

        # Pivot to wide format: index=before, columns=class, values=count
        pivot = counts.pivot(index="before", columns="class", values="count").fillna(0)

        # Ensure all classes are present in columns
        for cls in self.class_to_idx:
            if cls not in pivot.columns:
                pivot[cls] = 0.0

        # Sort columns by class index to ensure vector alignment
        sorted_cols = [self.idx_to_class[i] for i in range(self.num_classes)]
        pivot = pivot[sorted_cols]

        # Normalize rows to sum to 1 (probabilities)
        # Add epsilon to avoid division by zero (though groupby ensures count > 0)
        row_sums = pivot.sum(axis=1)
        probs = pivot.div(row_sums, axis=0)

        # Store in map
        # Convert to numpy arrays for fast lookup
        # We iterate over the dataframe. This might be slow for huge vocabs but
        # is done once.
        self.priors_map = {}
        # Using itertuples is faster than iterrows
        for row in probs.itertuples(index=True):
            token = row.Index
            # row[1:] contains the probabilities corresponding to sorted classes
            vector = np.array(row[1:], dtype=np.float32)
            self.priors_map[token] = vector

    def _dataframe_to_map(self, df):
        """
        Converts the loaded parquet dataframe back to the internal map.
        Assumes dataframe has 'token' column and columns for each class index
        (or a specific format).

        To keep parquet simple, we will save it as:
        token, prob_0, prob_1, ... prob_N
        """
        # Check if class vocab is loaded
        if not self.class_to_idx:
            # Try to infer from columns if they are named by class
            cols = [c for c in df.columns if c != "token"]
            # We assume columns are class names. We sort them to create vocab.
            # Ideally vocab should be loaded from json.
            pass

        # We assume columns are "class_NAME".
        # Let's rely on the vocab file being the source of truth for ordering.
        if not self.class_to_idx:
            raise ValueError(
                "Class vocabulary not found. Cannot reconstruct priors map."
            )

        sorted_classes = [self.idx_to_class[i] for i in range(self.num_classes)]

        # Check if columns match
        # The parquet is saved with class names as columns
        for row in df.itertuples(index=False):
            token = row.token
            # Extract values in correct order
            # getattr is safe because column names in parquet are valid identifiers usually,
            # but classes might have weird chars.
            # Safer: use values from the row corresponding to class names
            # However, itertuples uses named attributes.

            # Alternative: Use numpy array directly from df
            pass

        # Faster approach:
        tokens = df["token"].values
        # Extract feature columns in order
        feature_cols = sorted_classes
        # Check if all feature cols exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            # If missing, it means those classes had 0 prob for all tokens in the saved file?
            # Or the file format is different.
            # We'll fill 0.
            for c in missing:
                df[c] = 0.0

        vectors = df[feature_cols].values.astype(np.float32)

        self.priors_map = {t: v for t, v in zip(tokens, vectors)}

    def _save_priors(self):
        """
        Saves the priors map to a parquet file.
        """
        if not self.priors_map:
            return

        # Convert map to DataFrame
        # This can be memory intensive.
        # We construct a list of dicts or similar.

        # Recover class names
        sorted_classes = [self.idx_to_class[i] for i in range(self.num_classes)]

        data = []
        for token, vector in self.priors_map.items():
            row = {"token": token}
            for i, prob in enumerate(vector):
                row[sorted_classes[i]] = float(prob)
            data.append(row)

        df = pd.DataFrame(data)
        save_parquet(df, Config.PRIORS_PATH)

    def _save_class_vocab(self):
        ensure_dir(self.vocab_classes_path)
        with open(self.vocab_classes_path, "w") as f:
            json.dump(self.class_to_idx, f, indent=4)

    def _load_class_vocab(self):
        if os.path.exists(self.vocab_classes_path):
            with open(self.vocab_classes_path, "r") as f:
                data = json.load(f)

            # Handle structured vocabulary format from Vocabulary.save()
            if "token2idx" in data:
                self.class_to_idx = data["token2idx"]
            else:
                self.class_to_idx = data

            self.idx_to_class = {
                int(v) if isinstance(v, int) else v: k
                for k, v in self.class_to_idx.items()
            }
            self.num_classes = len(self.class_to_idx)
