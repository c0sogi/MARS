import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import log_loss

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library import config
from library import utils
from library import feature_engine
from library import logistic_model
from library import dataset


def main():
    # 1. Setup and Initialization
    print("Initializing workflow...")
    utils.seed_everything(config.SEED)

    # Define paths for intermediate files (IDs for train/val are not in config but needed for extraction)
    train_ids_path = os.path.join(config.WORKING_DIR, "train_ids.npy")
    val_ids_path = os.path.join(config.WORKING_DIR, "val_ids.npy")
    # Test labels dummy path
    test_labels_path = os.path.join(config.WORKING_DIR, "test_labels_dummy.npy")

    # 2. Feature Extraction
    # The extract_embeddings function handles loading data, running the dual backbone,
    # applying TTA, fusion, and caching.

    print("\n--- Processing Training Set ---")
    feature_engine.extract_embeddings(
        metadata_path=config.TRAIN_METADATA_PATH,
        embedding_path=config.CNN_TRAIN_EMBEDDINGS,
        label_path=config.LABELS_TRAIN_PATH,
        id_path=train_ids_path,
        load_cached_data=True,
    )

    print("\n--- Processing Validation Set ---")
    feature_engine.extract_embeddings(
        metadata_path=config.VAL_METADATA_PATH,
        embedding_path=config.CNN_VAL_EMBEDDINGS,
        label_path=config.LABELS_VAL_PATH,
        id_path=val_ids_path,
        load_cached_data=True,
    )

    print("\n--- Processing Test Set ---")
    feature_engine.extract_embeddings(
        metadata_path=config.TEST_METADATA_PATH,
        embedding_path=config.CNN_TEST_EMBEDDINGS,
        label_path=test_labels_path,
        id_path=config.IDS_TEST_PATH,
        load_cached_data=True,
    )

    # 3. Model Training
    print("\n--- Training Classifier ---")
    # Train Logistic Regression on the fused embeddings
    # We must pass explicit paths because the default paths in logistic_model might differ
    # from where we saved the fused embeddings (we used config.CNN_... variables)
    clf = logistic_model.train_classifier(
        train_embeddings_path=config.CNN_TRAIN_EMBEDDINGS,
        train_labels_path=config.LABELS_TRAIN_PATH,
        val_embeddings_path=config.CNN_VAL_EMBEDDINGS,
        val_labels_path=config.LABELS_VAL_PATH,
        C=config.LOGREG_C,
        max_iter=config.LOGREG_MAX_ITER,
        solver=config.LOGREG_SOLVER,
        seed=config.SEED,
    )

    # 4. Validation Assessment
    print("\n--- Validation Assessment ---")
    # Load validation data
    X_val = utils.load_array(config.CNN_VAL_EMBEDDINGS)
    y_val = utils.load_array(config.LABELS_VAL_PATH)

    # Predict probabilities
    val_probs = clf.predict_proba(X_val)

    # Calculate Metric
    final_metric = log_loss(y_val, val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-sample log loss
    # Get probability assigned to the true class
    # y_val contains indices 0..119
    true_class_probs = val_probs[np.arange(len(y_val)), y_val]
    # Clip to avoid log(0)
    epsilon = 1e-15
    true_class_probs = np.clip(true_class_probs, epsilon, 1 - epsilon)
    sample_losses = -np.log(true_class_probs)

    # Load validation metadata to get image paths
    val_df = pd.read_csv(config.VAL_METADATA_PATH)

    # Collect image metadata (Width, Height)
    # We read images to get accurate dimensions corresponding to the validation set
    widths = []
    heights = []

    # Ensure we process in the same order as the dataframe/embeddings
    for rel_path in val_df["file_path"]:
        full_path = os.path.join(config.INPUT_DIR, rel_path)
        try:
            with Image.open(full_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception as e:
            # Fallback if image read fails (unlikely given checks)
            widths.append(0)
            heights.append(0)

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "loss": sample_losses,
            "width": widths,
            "height": heights,
            "embedding_norm": np.linalg.norm(X_val, axis=1),
        }
    )

    # Calculate correlations
    corr_width = analysis_df["loss"].corr(analysis_df["width"])
    corr_height = analysis_df["loss"].corr(analysis_df["height"])
    corr_norm = analysis_df["loss"].corr(analysis_df["embedding_norm"])

    print("Correlation between model's error magnitude and input features:")
    print(f"  Image Width: {corr_width:.6f}")
    print(f"  Image Height: {corr_height:.6f}")
    print(f"  Embedding Norm: {corr_norm:.6f}")

    # 6. Submission Generation
    threshold = 0.12293165333323357

    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")

        # Predict on Test
        test_probs = logistic_model.predict_probabilities(
            clf, config.CNN_TEST_EMBEDDINGS
        )

        # Generate CSV
        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        logistic_model.generate_submission(
            test_ids_path=config.IDS_TEST_PATH,
            probabilities=test_probs,
            output_path=submission_path,
            train_metadata_path=config.TRAIN_METADATA_PATH,
        )
    else:
        print(f"\nMetric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
