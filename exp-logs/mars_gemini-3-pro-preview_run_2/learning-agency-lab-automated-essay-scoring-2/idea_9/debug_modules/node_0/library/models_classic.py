import os
import gc
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import get_logger, seed_everything, compute_qwk
from library.data import preprocess_text


class ClassicBranch:
    """
    Implements a Ridge Regression branch based on TF-IDF features.
    Handles both Lexical (Word N-gram) and Morphological (Char N-gram) branches.
    """

    def __init__(self, name, analyzer, ngram_range, min_df):
        """
        Args:
            name (str): Identifier for the branch (e.g., 'lexical', 'morphological').
            analyzer (str): 'word' or 'char'.
            ngram_range (tuple): The lower and upper boundary of the range of n-values.
            min_df (int): Ignore terms that have a document frequency strictly lower than this.
        """
        self.name = name
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.logger = get_logger(name)

    def run(self, load_cached_data=True):
        """
        Executes the 5-fold CV training and test prediction pipeline.

        Args:
            load_cached_data (bool): If True, attempts to load predictions from disk.

        Returns:
            tuple: (oof_preds, test_preds)
        """
        seed_everything(Config.seed)

        # Define cache paths
        oof_path = os.path.join(Config.output_dir, f"{self.name}_oof.npy")
        test_pred_path = os.path.join(Config.output_dir, f"{self.name}_test_preds.npy")

        # 1. Check Cache
        if (
            load_cached_data
            and os.path.exists(oof_path)
            and os.path.exists(test_pred_path)
        ):
            self.logger.info(
                f"Loading cached predictions for {self.name} from {oof_path}"
            )
            oof_preds = np.load(oof_path)
            test_preds = np.load(test_pred_path)
            return oof_preds, test_preds

        self.logger.info(f"Starting {self.name} branch training...")

        # 2. Load Data
        # We load the metadata to reconstruct the full dataset for CV
        if not os.path.exists(Config.train_path):
            raise FileNotFoundError(f"Train metadata not found at {Config.train_path}")

        df_train = pd.read_csv(Config.train_path)
        df_val = pd.read_csv(Config.val_path)
        df_test = pd.read_csv(Config.test_path)

        # Merge train and val for Stratified K-Fold
        df_full = pd.concat([df_train, df_val]).reset_index(drop=True)

        # Handle Debug Mode
        if Config.debug:
            self.logger.info(
                f"Debug mode: Subsetting data to {Config.debug_sample_size} samples."
            )
            df_full = df_full.head(Config.debug_sample_size)
            df_test = df_test.head(Config.debug_sample_size)

        # 3. Preprocess Text
        self.logger.info("Preprocessing text data...")
        # Apply minimal cleaning (whitespace normalization)
        full_texts = [preprocess_text(t) for t in df_full["full_text"]]
        test_texts = [preprocess_text(t) for t in df_test["full_text"]]
        y = df_full["score"].values

        # 4. Cross-Validation Loop
        oof_preds = np.zeros(len(df_full))
        test_preds_accum = np.zeros((len(df_test), Config.n_folds))

        skf = StratifiedKFold(
            n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, y.astype(str))):
            self.logger.info(f"--- Fold {fold} ---")

            # Split Data
            X_train_text = [full_texts[i] for i in train_idx]
            X_val_text = [full_texts[i] for i in val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            # Initialize Vectorizer
            # token_pattern=r"(?u)\b\w+\b" ensures single character words are kept (e.g., "I", "a")
            # This is only used if analyzer='word'
            token_pattern = r"(?u)\b\w+\b" if self.analyzer == "word" else None

            vectorizer = TfidfVectorizer(
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                sublinear_tf=True,
                use_idf=True,
                strip_accents="unicode",
                token_pattern=token_pattern,
            )

            # Fit on Train, Transform Val and Test
            # This prevents data leakage from validation/test sets into the IDF calculation
            X_train_vec = vectorizer.fit_transform(X_train_text)
            X_val_vec = vectorizer.transform(X_val_text)
            X_test_vec = vectorizer.transform(test_texts)

            # Train Ridge Regressor
            model = Ridge(
                alpha=1.0,
                random_state=Config.seed,
                solver="auto",
                fit_intercept=True,
            )
            model.fit(X_train_vec, y_train)

            # Predict Validation (OOF)
            val_pred = model.predict(X_val_vec)
            # Clip predictions to valid range [1, 6]
            val_pred = np.clip(val_pred, 1, 6)
            oof_preds[val_idx] = val_pred

            # Predict Test
            test_pred = model.predict(X_test_vec)
            test_pred = np.clip(test_pred, 1, 6)
            test_preds_accum[:, fold] = test_pred

            # Evaluate Fold
            qwk = compute_qwk(y_val, val_pred)
            self.logger.info(f"Fold {fold} QWK: {qwk}")

            # Cleanup to save memory
            del vectorizer, model, X_train_vec, X_val_vec, X_test_vec
            gc.collect()

        # 5. Finalize Results
        overall_qwk = compute_qwk(y, oof_preds)
        self.logger.info(f"Overall CV QWK: {overall_qwk}")

        # Average predictions across folds for the test set
        avg_test_preds = np.mean(test_preds_accum, axis=1)

        # 6. Save to Cache
        os.makedirs(Config.output_dir, exist_ok=True)
        np.save(oof_path, oof_preds)
        np.save(test_pred_path, avg_test_preds)
        self.logger.info(f"Saved OOF and Test predictions to {Config.output_dir}")

        return oof_preds, avg_test_preds


def run_classic_branches():
    """
    Orchestrates the execution of both Lexical and Morphological branches.
    """
    # 1. Lexical Branch (Word N-grams)
    # Captures vocabulary richness and specific word usage
    lexical = ClassicBranch(
        name="lexical",
        analyzer="word",
        ngram_range=Config.word_ngram_range,
        min_df=Config.word_min_df,
    )
    lexical.run(load_cached_data=True)

    # 2. Morphological Branch (Character N-grams)
    # Captures spelling, morphology, and sub-word structures
    morph = ClassicBranch(
        name="morphological",
        analyzer="char",
        ngram_range=Config.char_ngram_range,
        min_df=Config.char_min_df,
    )
    morph.run(load_cached_data=True)
