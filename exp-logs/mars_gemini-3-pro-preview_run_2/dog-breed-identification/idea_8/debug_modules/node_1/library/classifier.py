import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
import library.config as config


def get_breed_list():
    """
    Retrieves the sorted list of breed names from the training metadata.
    This corresponds to the class indices 0..N used by the model.
    """
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Training metadata not found at {config.TRAIN_METADATA_PATH}"
        )

    df = pd.read_csv(config.TRAIN_METADATA_PATH)
    # The dataset class sorts breeds alphabetically to assign indices
    return sorted(df["breed"].unique())


def train_model(train_embeddings, train_labels):
    """
    Trains a LogisticRegressionCV model on the provided embeddings.
    """
    print(f"Training LogisticRegressionCV on data shape {train_embeddings.shape}...")

    # Initialize model with config parameters
    # verbose=0 ensures silent execution as required
    clf = LogisticRegressionCV(
        cv=config.CLF_CV_FOLDS,
        max_iter=config.CLF_MAX_ITER,
        solver=config.CLF_SOLVER,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=config.SEED,
        verbose=0,
    )

    clf.fit(train_embeddings, train_labels)
    return clf


def train_and_predict(
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    test_embeddings,
    test_ids,
    load_cached_model=True,
):
    """
    Main pipeline function:
    1. Loads or trains the model.
    2. Evaluates on validation set.
    3. Generates predictions for test set.
    4. Saves submission file.
    """

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    model_path = config.MODEL_SAVE_PATH
    model = None

    # --- 1. Model Loading / Training ---
    if load_cached_model and os.path.exists(model_path):
        print(f"Loading cached model from {model_path}...")
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Failed to load cached model: {e}. Retraining...")
            model = None

    if model is None:
        model = train_model(train_embeddings, train_labels)
        print(f"Saving trained model to {model_path}...")
        joblib.dump(model, model_path)

    # --- 2. Validation Evaluation ---
    print("Evaluating model on validation set...")
    # predict_proba returns (N_samples, N_classes)
    val_probs = model.predict_proba(val_embeddings)

    # Calculate Log Loss
    # labels are integers, probs are float arrays
    loss = log_loss(val_labels, val_probs)
    print(f"Validation Multi Class Log Loss: {loss}")

    # --- 3. Submission Generation ---
    print("Generating predictions for test set...")
    test_probs = model.predict_proba(test_embeddings)

    # Get column headers (breed names)
    breeds = get_breed_list()

    # Verify shape consistency
    if test_probs.shape[1] != len(breeds):
        raise ValueError(
            f"Model predicts {test_probs.shape[1]} classes, but found {len(breeds)} unique breeds in metadata."
        )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(test_probs, columns=breeds)

    # Insert ID column at the beginning
    submission_df.insert(0, "id", test_ids)

    # Save to CSV
    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    return loss
