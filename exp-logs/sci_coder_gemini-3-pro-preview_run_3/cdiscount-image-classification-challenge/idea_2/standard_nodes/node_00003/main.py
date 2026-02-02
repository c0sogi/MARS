import os
import sys
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.config import (
    TRAIN_BSON_PATH,
    TEST_BSON_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CATEGORY_NAMES_PATH,
    SUBMISSION_PATH,
    MODEL_SAVE_PATH,
    DEVICE,
    FE_BATCH_SIZE,
    MLP_BATCH_SIZE,
    set_seed,
)
from library.models import FrozenResNet
from library.data_loader import ProductImageDataset, get_label_map
from library.feature_processor import extract_features
from library.trainer import train_mlp, predict_mlp


def main():
    # 1. Setup
    set_seed(42)
    print(f"Using device: {DEVICE}")

    # Define subset sizes for fast baseline execution
    # We limit training data to ensure the pipeline completes quickly (within 2 hours)
    # while maintaining enough data to learn a reasonable baseline.
    TRAIN_SIZE = 200000
    VAL_SIZE = 50000

    # 2. Feature Extraction
    # We use the FrozenResNet to convert images to embeddings.
    # This is the most time-consuming part, so we use subsets for train/val.

    print("Initializing Frozen ResNet-18...")
    resnet = FrozenResNet()

    # --- Train Set ---
    print(f"Preparing Train Dataset (Subset: {TRAIN_SIZE})...")
    train_dataset = ProductImageDataset(
        metadata_path=TRAIN_META_PATH,
        bson_path=TRAIN_BSON_PATH,
        category_names_path=CATEGORY_NAMES_PATH,
        is_test=False,
        debug_size=TRAIN_SIZE,
    )

    train_emb, train_lbl, _ = extract_features(
        dataset=train_dataset,
        model=resnet,
        batch_size=FE_BATCH_SIZE,
        device=DEVICE,
        cache_prefix=f"train_subset_{TRAIN_SIZE}",
        load_cached_data=True,
    )

    # --- Validation Set ---
    print(f"Preparing Validation Dataset (Subset: {VAL_SIZE})...")
    val_dataset = ProductImageDataset(
        metadata_path=VAL_META_PATH,
        bson_path=TRAIN_BSON_PATH,
        category_names_path=CATEGORY_NAMES_PATH,
        is_test=False,
        debug_size=VAL_SIZE,
    )

    val_emb, val_lbl, _ = extract_features(
        dataset=val_dataset,
        model=resnet,
        batch_size=FE_BATCH_SIZE,
        device=DEVICE,
        cache_prefix=f"val_subset_{VAL_SIZE}",
        load_cached_data=True,
    )

    # --- Test Set ---
    print("Preparing Test Dataset (Full)...")
    # We must process the FULL test set to generate a valid submission
    test_dataset = ProductImageDataset(
        metadata_path=TEST_META_PATH,
        bson_path=TEST_BSON_PATH,
        category_names_path=CATEGORY_NAMES_PATH,
        is_test=True,
        debug_size=None,
    )

    test_emb, _, test_ids = extract_features(
        dataset=test_dataset,
        model=resnet,
        batch_size=FE_BATCH_SIZE,
        device=DEVICE,
        cache_prefix="test_full",
        load_cached_data=True,
    )

    # Clean up ResNet to free GPU memory for MLP training
    del resnet
    torch.cuda.empty_cache()

    # 3. Model Training (MLP)
    print("Training MLP Classifier...")
    model = train_mlp(
        train_embeddings=train_emb,
        train_labels=train_lbl,
        val_embeddings=val_emb,
        val_labels=val_lbl,
        batch_size=MLP_BATCH_SIZE,
        device=DEVICE,
        save_path=MODEL_SAVE_PATH,
    )

    # 4. Validation & Failure Analysis
    print("Performing Validation Analysis...")

    # Get predictions on validation set
    val_preds = predict_mlp(model, val_emb, batch_size=MLP_BATCH_SIZE, device=DEVICE)

    # Calculate Metric
    accuracy = np.mean(val_preds == val_lbl)
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis: Correlation between error and embedding magnitude (L2 norm)
    # Hypothesis: Images with low signal (low norm features) might be harder to classify.
    print("Performing Failure Analysis...")
    embedding_norms = np.linalg.norm(val_emb, axis=1)
    errors = (val_preds != val_lbl).astype(int)  # 1 if error, 0 if correct

    # Handle edge case of zero variance
    if np.std(errors) == 0 or np.std(embedding_norms) == 0:
        correlation = 0.0
    else:
        correlation = np.corrcoef(embedding_norms, errors)[0, 1]

    print(f"Correlation between embedding magnitude and error: {correlation}")

    # 5. Submission Generation
    print("Generating Submission...")

    # Get predictions on test set
    test_preds_idx = predict_mlp(
        model, test_emb, batch_size=MLP_BATCH_SIZE, device=DEVICE
    )

    # Map class indices back to category_ids
    print("Mapping predictions to category IDs...")
    _, idx_to_cat = get_label_map(CATEGORY_NAMES_PATH)

    # Efficient mapping using a lookup table
    # idx_to_cat keys are 0..N-1.
    max_idx = max(idx_to_cat.keys())
    lookup_table = np.zeros(max_idx + 1, dtype=np.int64)
    for idx, cat_id in idx_to_cat.items():
        lookup_table[idx] = cat_id

    test_preds_cat_id = lookup_table[test_preds_idx]

    # Create DataFrame
    submission_df = pd.DataFrame({"_id": test_ids, "category_id": test_preds_cat_id})

    # Save to ./submission/submission.csv as requested
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_file_path = os.path.join(submission_dir, "submission.csv")

    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission saved to {submission_file_path}")

    # Also save to the config path (working directory) as a backup
    if os.path.abspath(submission_file_path) != os.path.abspath(SUBMISSION_PATH):
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission also saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
