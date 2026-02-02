import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library import config, dataset, transforms, classifier


def train_transductive(
    load_cached_data=True,
    debug_mode=config.DEBUG_MODE,
    debug_size=config.DEBUG_SUBSET_SIZE,
    pseudo_label_threshold=config.PSEUDO_LABEL_THRESHOLD,
):
    """
    Executes the Transductive Self-Training pipeline.

    1. Loads augmented data (cached).
    2. Applies Power Transformation and Scaling (cached).
    3. Trains LDA with Transductive Pseudo-Labeling.
    4. Evaluates on Validation set.
    5. Generates Submission file.
    """

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load Data (Augmented)
    # -------------------------------------------------------------------------
    print("Loading dataset...")
    # library.dataset handles the caching of geometric augmentation
    train_data, val_data, test_data = dataset.load_dataset(
        load_cached_data=load_cached_data, debug_mode=debug_mode, debug_size=debug_size
    )

    X_train_raw, y_train, train_ids = train_data
    X_val_raw, y_val, val_ids = val_data
    X_test_raw, test_ids = test_data

    # -------------------------------------------------------------------------
    # 2. Preprocessing (Power Transform + Scaling)
    # -------------------------------------------------------------------------
    # Define cache paths for transformed features
    # We append a suffix for debug mode to avoid cache collisions
    suffix = "_debug" if debug_mode else ""
    cache_X_train = os.path.join(config.WORKING_DIR, f"X_train_transformed{suffix}.npy")
    cache_X_val = os.path.join(config.WORKING_DIR, f"X_val_transformed{suffix}.npy")
    cache_X_test = os.path.join(config.WORKING_DIR, f"X_test_transformed{suffix}.npy")

    # Check if cached transformed data exists
    if (
        load_cached_data
        and os.path.exists(cache_X_train)
        and os.path.exists(cache_X_val)
        and os.path.exists(cache_X_test)
    ):
        print("Loading cached transformed features...")
        X_train = np.load(cache_X_train)
        X_val = np.load(cache_X_val)
        X_test = np.load(cache_X_test)
    else:
        print("Fitting preprocessing pipeline and transforming features...")
        # Get pipeline (PowerTransformer + StandardScaler)
        pipeline = transforms.get_pipeline()

        # Fit on training data ONLY
        pipeline.fit(X_train_raw)

        # Transform all splits
        X_train = pipeline.transform(X_train_raw)
        X_val = pipeline.transform(X_val_raw)
        X_test = pipeline.transform(X_test_raw)

        # Save to cache
        print("Saving transformed features to cache...")
        np.save(cache_X_train, X_train)
        np.save(cache_X_val, X_val)
        np.save(cache_X_test, X_test)

    # -------------------------------------------------------------------------
    # 3. Model Training (Transductive LDA)
    # -------------------------------------------------------------------------
    print(
        f"Initializing LeafLDA with solver={config.LDA_SOLVER}, shrinkage={config.LDA_SHRINKAGE}"
    )
    model = classifier.LeafLDA()

    # Fit Transductive
    # This handles Supervisor -> Pseudo-Label -> Student
    model.fit_transductive(
        X_train, y_train, X_test, pseudo_label_threshold=pseudo_label_threshold
    )

    # -------------------------------------------------------------------------
    # 4. Validation
    # -------------------------------------------------------------------------
    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    # We use model.classes_ to ensure correct mapping of probabilities to labels
    loss = log_loss(y_val, val_probs, labels=model.classes_)
    print(f"Validation Log Loss: {loss}")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    print("Generating predictions for Test Set...")
    test_probs = model.predict_proba(X_test)

    # Load sample submission to get the required column order
    sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

    # Create DataFrame with predictions using model classes
    submission_df = pd.DataFrame(test_probs, columns=model.classes_)

    # Insert ID column
    submission_df.insert(0, "id", test_ids.values)

    # Identify target columns from sample submission (excluding 'id')
    target_cols = [c for c in sample_sub.columns if c != "id"]

    # Ensure all required columns exist (fill missing with 0 if any class was absent in train)
    missing_cols = set(target_cols) - set(submission_df.columns)
    for c in missing_cols:
        submission_df[c] = 0.0

    # Reorder columns to strictly match sample_submission
    final_cols = ["id"] + target_cols
    submission_df = submission_df[final_cols]

    # Clip probabilities as per metric description: max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    submission_df[target_cols] = submission_df[target_cols].clip(
        lower=epsilon, upper=1 - epsilon
    )

    # Save Submission
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    print(f"Saving submission to {config.SUBMISSION_FILE_PATH}...")
    submission_df.to_csv(config.SUBMISSION_FILE_PATH, index=False)

    print("Training and submission generation completed.")
