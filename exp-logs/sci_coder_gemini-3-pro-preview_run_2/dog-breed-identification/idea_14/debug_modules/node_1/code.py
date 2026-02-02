import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
import library.config as config
import library.transforms as transforms
import library.dataset as dataset
import library.extraction as extraction
import library.processing as processing
import library.modeling as modeling
import library.ensemble as ensemble


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    print("Initializing workflow...")
    set_seed(config.SEED)

    # OPTIMIZATION: Override config parameters to ensure the script runs quickly.
    # We reduce the number of CV folds and the search space for the regularization parameter C.
    # This speeds up the LogisticRegressionCV fitting process significantly.
    config.LOGREG_CV_FOLDS = 2
    config.LOGREG_C_VALUES = np.logspace(-1, 1, 3)

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Metadata Loading
    # -------------------------------------------------------------------------
    print("Loading metadata...")
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {config.TRAIN_METADATA_PATH}")

    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # Retrieve class mapping (Breed -> Index)
    class_to_idx = dataset.get_class_mapping()
    num_classes = len(class_to_idx)
    print(f"Detected {num_classes} unique breeds.")

    # -------------------------------------------------------------------------
    # 3. Stream Processing (A: CNN, B: ViT)
    # -------------------------------------------------------------------------
    streams = ["stream_a", "stream_b"]

    # Dictionaries to hold predictions for the ensemble stage
    val_preds = {}
    test_preds = {}

    # Placeholders for targets and IDs to ensure alignment
    val_targets = None
    test_ids = None

    for stream in streams:
        print(f"\n{'='*20} Processing {stream} {'='*20}")

        # --- A. Transforms & Datasets ---
        print("Creating datasets...")
        stream_transforms = transforms.get_stream_transforms(stream)

        # We use the full datasets to ensure all 120 classes are seen by the classifier
        ds_train = dataset.MultiViewDataset(
            train_df, stream_transforms, class_to_idx, is_test=False
        )
        ds_val = dataset.MultiViewDataset(
            val_df, stream_transforms, class_to_idx, is_test=False
        )
        ds_test = dataset.MultiViewDataset(
            test_df, stream_transforms, class_to_idx=None, is_test=True
        )

        # Create DataLoaders
        dl_train = DataLoader(
            ds_train,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

        # --- B. Feature Extraction ---
        print(f"Loading backbone model for {stream}...")
        backbone = extraction.load_backbone(stream, device=config.DEVICE)

        # Define splits to process
        splits = [("train", dl_train), ("val", dl_val), ("test", dl_test)]

        for split_name, loader in splits:
            # Define a specific cache directory for this stream and split
            # e.g., ./working/idea_14/stream_a_train
            cache_dir = os.path.join(config.WORKING_DIR, f"{stream}_{split_name}")

            # Extract features (handles caching internally)
            extraction.extract_and_save_features(
                loader=loader,
                model=backbone,
                save_dir=cache_dir,
                device=config.DEVICE,
                load_cached_data=config.LOAD_CACHED_DATA,
            )

        # Cleanup GPU memory
        del backbone
        torch.cuda.empty_cache()

        # --- C. Feature Fusion ---
        print("Fusing multi-view features...")
        # Load raw embeddings, concatenate views, and save fused version
        X_train, y_train, _ = processing.load_and_fuse_features(
            embedding_dir=os.path.join(config.WORKING_DIR, f"{stream}_train"),
            cache_name=f"{stream}_train",
            load_cached_data=config.LOAD_CACHED_DATA,
        )

        X_val, y_val, _ = processing.load_and_fuse_features(
            embedding_dir=os.path.join(config.WORKING_DIR, f"{stream}_val"),
            cache_name=f"{stream}_val",
            load_cached_data=config.LOAD_CACHED_DATA,
        )

        X_test, _, ids_test = processing.load_and_fuse_features(
            embedding_dir=os.path.join(config.WORKING_DIR, f"{stream}_test"),
            cache_name=f"{stream}_test",
            load_cached_data=config.LOAD_CACHED_DATA,
        )

        # Store ground truth and IDs for later use
        if val_targets is None:
            val_targets = y_val
        if test_ids is None:
            test_ids = ids_test

        # --- D. Modeling (Logistic Regression) ---
        print(f"Training Logistic Regression Head for {stream}...")
        clf = modeling.train_logistic_regression(
            X_train,
            y_train,
            stream_name=stream,
            load_cached_model=config.LOAD_CACHED_DATA,
        )

        # Evaluate
        print(f"Evaluating {stream}...")
        modeling.evaluate_model(clf, X_val, y_val)

        # Generate Probabilities
        print(f"Predicting {stream}...")
        p_val = modeling.predict_probabilities(clf, X_val)
        p_test = modeling.predict_probabilities(clf, X_test)

        # Store for ensemble
        val_preds[stream] = p_val
        test_preds[stream] = p_test

        # Validation Logic
        if p_val.shape[1] != num_classes:
            raise AssertionError(
                f"Prediction shape mismatch. Expected {num_classes} columns, got {p_val.shape[1]}."
            )

    # -------------------------------------------------------------------------
    # 4. Ensemble Optimization
    # -------------------------------------------------------------------------
    print(f"\n{'='*20} Ensemble Optimization {'='*20}")

    # Optimize weights (w_a, w_b) based on validation log loss
    w_a, w_b = ensemble.optimize_ensemble_weights(
        val_preds["stream_a"], val_preds["stream_b"], val_targets
    )

    # Compute final weighted test predictions
    final_test_preds = ensemble.compute_weighted_prediction(
        test_preds["stream_a"], test_preds["stream_b"], w_a, w_b
    )

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    print(f"\n{'='*20} Generating Submission {'='*20}")

    ensemble.generate_submission(
        test_ids=test_ids,
        predictions=final_test_preds,
        output_path=config.SUBMISSION_PATH,
    )

    print("\nProcess completed successfully.")


if __name__ == "__main__":
    main()
