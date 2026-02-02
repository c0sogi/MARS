import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from library.config import Config
from library.data_factory import DataFactory
from library.classifier import TokenClassifier
from library.dictionary import NormalizationDictionary


def generate_predictions(load_cached_data=True):
    """
    Orchestrates the inference pipeline: loads data and models, predicts classes,
    applies dictionary-based normalization, and saves the submission file.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed test features
                                 from the cache directory.
    """
    print("Inference: Starting prediction pipeline...")

    # 1. Prepare Test Data
    # DataFactory handles feature engineering and caching
    factory = DataFactory()
    print("Inference: Loading test data...")
    X_test, meta_df = factory.prepare_test_data(load_cached_data=load_cached_data)

    print(f"Inference: Test data loaded. Shape: {X_test.shape}")

    # 2. Load Trained Classifier
    classifier = TokenClassifier()
    try:
        classifier.load()
    except FileNotFoundError:
        print(
            f"Error: Model file not found at {Config.MODEL_FILE}. Please train the model first."
        )
        return

    # 3. Load Label Encoder Classes
    # We need to map predicted integer indices back to class strings (e.g., 0 -> "PLAIN")
    if not os.path.exists(Config.LABEL_ENCODER_PATH):
        print(
            f"Error: Label encoder classes not found at {Config.LABEL_ENCODER_PATH}. Please train the model first."
        )
        return

    print("Inference: Loading label encoder...")
    classes = np.load(Config.LABEL_ENCODER_PATH, allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = classes

    # 4. Predict Semiotic Classes
    print("Inference: Predicting token classes with XGBoost...")
    # Returns integer indices
    pred_indices = classifier.predict(X_test)
    # Convert to string labels
    pred_class_names = le.inverse_transform(pred_indices)

    # 5. Apply Normalization Dictionary
    print("Inference: Applying normalization dictionary...")
    # Ensure dictionary is loaded
    norm_dict = factory.get_normalization_dictionary()

    # Extract raw tokens as strings to ensure safe lookup
    raw_tokens = meta_df["before"].astype(str).values

    # Vectorized lookup is hard with a dictionary of dictionaries, so we use a fast list comprehension
    # The dictionary handles the fallback logic (returning raw token if lookup fails)
    normalized_texts = [
        norm_dict.get_normalization(token, cls_name)
        for token, cls_name in zip(raw_tokens, pred_class_names)
    ]

    # 6. Generate Submission File
    print("Inference: Formatting submission DataFrame...")
    submission_df = pd.DataFrame({"id": meta_df["id"], "after": normalized_texts})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    print(f"Inference: Saving submission to {Config.SUBMISSION_FILE}...")
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)

    print("Inference: Process complete.")
    print("Head of submission:")
    print(submission_df.head())
