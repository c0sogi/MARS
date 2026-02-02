import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
from PIL import Image

# Import library modules
import library.config as config
import library.transforms as transforms
import library.dataset as dataset
import library.extraction as extraction
import library.processing as processing
import library.modeling as modeling
import library.ensemble as ensemble


def set_seed(seed=config.SEED):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(df, stream_type, class_to_idx, is_test=False):
    """Creates a DataLoader for a specific stream and dataset split."""
    tfms = transforms.get_stream_transforms(stream_type)
    ds = dataset.MultiViewDataset(
        dataframe=df, transforms_dict=tfms, class_to_idx=class_to_idx, is_test=is_test
    )
    # Shuffle only for training if we were doing SGD, but for feature extraction
    # order doesn't strictly matter as long as IDs match. However, extraction.py
    # returns IDs, so we can align later. We'll keep shuffle=False for extraction stability.
    loader = DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader


def process_stream(stream_name, train_df, val_df, test_df, class_to_idx):
    """
    Executes the full pipeline for a single stream:
    Extraction -> Fusion -> Training -> Inference
    """
    print(f"\n{'='*20} Processing {stream_name} {'='*20}")

    # 1. Feature Extraction
    # We need to load the model first
    print(f"Loading backbone for {stream_name}...")
    model = extraction.load_backbone(stream_name)

    # Define cache directory for this stream
    cache_dir = os.path.join(config.WORKING_DIR, f"{stream_name}_features")

    # Extract for all splits
    loaders = {
        "train": get_dataloaders(train_df, stream_name, class_to_idx, is_test=False),
        "val": get_dataloaders(val_df, stream_name, class_to_idx, is_test=False),
        "test": get_dataloaders(test_df, stream_name, class_to_idx, is_test=True),
    }

    for split_name, loader in loaders.items():
        split_cache_dir = os.path.join(cache_dir, split_name)
        extraction.extract_and_save_features(
            loader=loader,
            model=model,
            save_dir=split_cache_dir,
            load_cached_data=config.LOAD_CACHED_DATA,
        )

    # Cleanup Model to free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Feature Fusion
    print(f"Fusing features for {stream_name}...")
    X_train, y_train, _ = processing.load_and_fuse_features(
        os.path.join(cache_dir, "train"),
        f"{stream_name}_train",
        config.LOAD_CACHED_DATA,
    )
    X_val, y_val, _ = processing.load_and_fuse_features(
        os.path.join(cache_dir, "val"), f"{stream_name}_val", config.LOAD_CACHED_DATA
    )
    X_test, _, test_ids = processing.load_and_fuse_features(
        os.path.join(cache_dir, "test"), f"{stream_name}_test", config.LOAD_CACHED_DATA
    )

    # 3. Model Training
    clf = modeling.train_logistic_regression(
        X_train, y_train, stream_name, load_cached_model=config.LOAD_CACHED_DATA
    )

    # 4. Inference
    print(f"Generating predictions for {stream_name}...")
    val_preds = modeling.predict_probabilities(clf, X_val)
    test_preds = modeling.predict_probabilities(clf, X_test)

    # Evaluate individual stream performance
    val_loss = log_loss(y_val, val_preds)
    print(f"{stream_name} Validation Log Loss: {val_loss:.6f}")

    return val_preds, test_preds, y_val, test_ids


def perform_failure_analysis(val_df, y_val, val_preds, class_to_idx):
    """
    Analyzes the correlation between error magnitude and image properties.
    """
    print(f"\n{'='*20} Failure Analysis {'='*20}")

    # 1. Calculate Per-Sample Log Loss
    # Gather the probability assigned to the true class
    # y_val are indices, val_preds is (N, C)
    true_class_probs = val_preds[np.arange(len(y_val)), y_val]
    # Clip to avoid log(0)
    epsilon = 1e-15
    true_class_probs = np.clip(true_class_probs, epsilon, 1 - epsilon)
    sample_losses = -np.log(true_class_probs)

    # 2. Extract Metadata Features (Width, Height, Aspect Ratio)
    # We need to iterate over the validation images
    widths = []
    heights = []
    ratios = []

    print("Extracting validation image metadata for analysis...")
    for _, row in val_df.iterrows():
        img_path = os.path.join(config.INPUT_DIR, row["file_path"])
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                ratios.append(w / h)
        except Exception:
            # Fallback if image load fails (shouldn't happen based on metadata checks)
            widths.append(0)
            heights.append(0)
            ratios.append(0)

    widths = np.array(widths)
    heights = np.array(heights)
    ratios = np.array(ratios)

    # 3. Compute Correlations
    corr_w, _ = pearsonr(sample_losses, widths)
    corr_h, _ = pearsonr(sample_losses, heights)
    corr_r, _ = pearsonr(sample_losses, ratios)

    print(f"Correlation between Error and Image Width: {corr_w:.4f}")
    print(f"Correlation between Error and Image Height: {corr_h:.4f}")
    print(f"Correlation between Error and Aspect Ratio: {corr_r:.4f}")

    # Identify worst failures
    worst_indices = np.argsort(sample_losses)[-5:]
    print("\nTop 5 Worst Predictions:")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    for idx in worst_indices:
        true_breed = idx_to_class[y_val[idx]]
        pred_breed_idx = np.argmax(val_preds[idx])
        pred_breed = idx_to_class[pred_breed_idx]
        conf = val_preds[idx][pred_breed_idx]
        print(
            f"  ID: {val_df.iloc[idx]['id']} | True: {true_breed} | Pred: {pred_breed} ({conf:.4f}) | Loss: {sample_losses[idx]:.4f}"
        )


def main():
    set_seed()

    # --- Data Loading ---
    print("Loading metadata...")
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            "Metadata not found. Ensure metadata generation script has run."
        )

    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    class_to_idx = dataset.get_class_mapping()
    print(f"Classes: {len(class_to_idx)}")

    # --- Stream A: ConvNeXt ---
    val_preds_a, test_preds_a, y_val, test_ids = process_stream(
        "stream_a", train_df, val_df, test_df, class_to_idx
    )

    # --- Stream B: DINOv2 ---
    val_preds_b, test_preds_b, _, _ = process_stream(
        "stream_b", train_df, val_df, test_df, class_to_idx
    )

    # --- Ensemble Optimization ---
    print(f"\n{'='*20} Ensemble Optimization {'='*20}")
    w_a, w_b = ensemble.optimize_ensemble_weights(val_preds_a, val_preds_b, y_val)

    # Compute Final Validation Predictions
    final_val_preds = ensemble.compute_weighted_prediction(
        val_preds_a, val_preds_b, w_a, w_b
    )
    final_metric = log_loss(y_val, final_val_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    perform_failure_analysis(val_df, y_val, final_val_preds, class_to_idx)

    # --- Submission ---
    THRESHOLD = 0.11640673500383826

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        final_test_preds = ensemble.compute_weighted_prediction(
            test_preds_a, test_preds_b, w_a, w_b
        )
        ensemble.generate_submission(test_ids, final_test_preds)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
