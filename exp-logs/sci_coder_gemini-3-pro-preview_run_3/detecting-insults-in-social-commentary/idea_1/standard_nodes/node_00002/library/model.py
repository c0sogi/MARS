import os
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.nb_transformer import NBTransformer
from library.data_loader import load_datasets
from library.utils import set_seed


class NBLRModel:
    """
    Naive Bayes-weighted Logistic Regression Model for Insult Detection.
    Combines Word and Character n-grams with NB feature weighting.
    """

    def __init__(self):
        # Initialize Word-level TF-IDF Vectorizer
        self.word_vec = TfidfVectorizer(
            ngram_range=Config.WORD_NGRAM_RANGE,
            min_df=Config.WORD_MIN_DF,
            max_features=Config.WORD_MAX_FEATURES,
            use_idf=Config.WORD_USE_IDF,
            smooth_idf=Config.WORD_SMOOTH_IDF,
            sublinear_tf=Config.WORD_SUBLINEAR_TF,
            token_pattern=Config.WORD_TOKEN_PATTERN,
            strip_accents="unicode",
        )

        # Initialize Character-level TF-IDF Vectorizer
        self.char_vec = TfidfVectorizer(
            ngram_range=Config.CHAR_NGRAM_RANGE,
            min_df=Config.CHAR_MIN_DF,
            max_features=Config.CHAR_MAX_FEATURES,
            use_idf=Config.CHAR_USE_IDF,
            smooth_idf=Config.CHAR_SMOOTH_IDF,
            sublinear_tf=Config.CHAR_SUBLINEAR_TF,
            analyzer="char",
            strip_accents="unicode",
        )

        # Initialize Naive Bayes Transformer
        self.nb_transformer = NBTransformer(alpha=1.0)

        # Initialize Logistic Regression Classifier
        self.lr_model = LogisticRegression(
            C=Config.LR_C,
            solver=Config.LR_SOLVER,
            max_iter=Config.LR_MAX_ITER,
            penalty=Config.LR_PENALTY,
            dual=Config.LR_DUAL,
            class_weight=Config.LR_CLASS_WEIGHT,
            random_state=Config.SEED,
        )

    def fit(self, train_df, val_df):
        """
        Fits the vectorizers, NB transformer, and Logistic Regression model.
        Evaluates on the validation set.
        """
        print("Fitting NBLRModel...")

        # Prepare Training Data
        X_train_text = train_df[Config.TEXT_COL]
        y_train = train_df[Config.LABEL_COL].values

        # Prepare Validation Data
        X_val_text = val_df[Config.TEXT_COL]
        y_val = val_df[Config.LABEL_COL].values

        # 1. Feature Extraction (TF-IDF)
        print("Extracting Word features...")
        X_train_word = self.word_vec.fit_transform(X_train_text)
        X_val_word = self.word_vec.transform(X_val_text)

        print("Extracting Char features...")
        X_train_char = self.char_vec.fit_transform(X_train_text)
        X_val_char = self.char_vec.transform(X_val_text)

        # 2. Feature Stacking
        print("Stacking features...")
        X_train = sparse.hstack([X_train_word, X_train_char])
        X_val = sparse.hstack([X_val_word, X_val_char])

        # 3. Naive Bayes Weighting
        print("Applying NB weighting...")
        self.nb_transformer.fit(X_train, y_train)
        X_train_nb = self.nb_transformer.transform(X_train)
        X_val_nb = self.nb_transformer.transform(X_val)

        # 4. Logistic Regression Training
        print("Training Logistic Regression...")
        self.lr_model.fit(X_train_nb, y_train)

        # 5. Validation Evaluation
        print("Evaluating on Validation set...")
        val_preds = self.lr_model.predict_proba(X_val_nb)[:, 1]
        auc = roc_auc_score(y_val, val_preds)

        # Print metric with full precision
        print(f"Validation AUC: {auc}")

        return auc

    def predict_proba(self, test_df):
        """
        Generates probability predictions for the test set.
        """
        X_test_text = test_df[Config.TEXT_COL]

        # 1. Feature Extraction
        X_test_word = self.word_vec.transform(X_test_text)
        X_test_char = self.char_vec.transform(X_test_text)

        # 2. Feature Stacking
        X_test = sparse.hstack([X_test_word, X_test_char])

        # 3. Naive Bayes Weighting
        X_test_nb = self.nb_transformer.transform(X_test)

        # 4. Prediction
        preds = self.lr_model.predict_proba(X_test_nb)[:, 1]
        return preds

    def save(self, path):
        """
        Saves the model artifacts using joblib.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to {path}")


def run_pipeline():
    """
    Orchestrates the end-to-end pipeline:
    1. Load Data
    2. Train Model
    3. Generate Submission
    """
    set_seed(Config.SEED)

    # 1. Load Datasets
    print("Loading datasets...")
    train_df, val_df, test_df = load_datasets(load_cached_data=True)

    # 2. Initialize and Train Model
    model = NBLRModel()
    model.fit(train_df, val_df)

    # 3. Save Model Artifacts
    model.save(Config.MODEL_ARTIFACT_PATH)

    # 4. Generate Predictions for Test Set
    print("Generating predictions for Test set...")
    test_preds = model.predict_proba(test_df)

    # 5. Create Submission File
    # We use the structure from the test dataframe and add the predictions
    submission_df = test_df.copy()

    # Ensure the submission has the correct columns: Insult, Date, Comment
    # The sample submission has 'Insult' as the first column.
    submission_df["Insult"] = test_preds

    # Select and reorder columns to match sample submission format
    # Sample format: Insult, Date, Comment
    cols = ["Insult", Config.DATE_COL, Config.TEXT_COL]

    # Verify columns exist
    missing_cols = [c for c in cols if c not in submission_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for submission: {missing_cols}")

    submission_df = submission_df[cols]

    # Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
    print("Head of submission:")
    print(submission_df.head())
