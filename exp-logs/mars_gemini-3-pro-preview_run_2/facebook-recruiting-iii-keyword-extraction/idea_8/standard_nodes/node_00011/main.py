import sys
import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse
from sklearn.metrics import f1_score

# Import provided library modules
from library.config import Config
from library.utils import set_seed, FocalLoss
from library.data_processing import prepare_data, get_dataloaders
from library.model import WideAndDeepModel
from library.trainer import Trainer, optimize_threshold, generate_submission


def run_pipeline():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    set_seed(Config.SEED)

    # Override Config for Fast Baseline Execution
    # We use a subset of data and fewer epochs to meet the time constraints
    Config.DEBUG = True
    Config.DEBUG_SIZE = 100000  # 100k samples
    Config.EPOCHS = 2

    # Ensure Batch Size is appropriate
    Config.BATCH_SIZE = 512

    # --------------------------------------------------------------------------
    # 2. Data Preparation
    # --------------------------------------------------------------------------
    # Prepare data (load cached if available, else process)
    prepare_data(load_cached_data=True, debug=Config.DEBUG)

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    device = torch.device(Config.DEVICE)

    model = WideAndDeepModel(
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        wide_dim=Config.TFIDF_MAX_FEATURES,
        num_classes=Config.NUM_CLASSES,
        filter_sizes=Config.FILTER_SIZES,
        num_filters=Config.NUM_FILTERS,
        attention_dim=Config.ATTENTION_DIM,
        dropout=Config.DROPOUT,
    ).to(device)

    # Optimizer & Criterion
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = FocalLoss(gamma=Config.FOCAL_LOSS_GAMMA, reduction="mean")

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=Config.USE_AMP,
    )

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    trainer.fit(epochs=Config.EPOCHS, patience=2)

    # --------------------------------------------------------------------------
    # 5. Validation & Threshold Optimization
    # --------------------------------------------------------------------------
    # Load best model weights
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    # Predict probabilities on validation set
    val_probs = trainer.predict(val_loader)

    # Load validation targets for scoring
    # We load the sparse matrix and slice it if in DEBUG mode to match val_loader
    val_labels_sparse = scipy.sparse.load_npz(Config.VAL_LABELS_PATH)
    if Config.DEBUG:
        val_labels_sparse = val_labels_sparse[: Config.DEBUG_SIZE]

    # Find optimal threshold
    best_threshold = optimize_threshold(val_probs, val_labels_sparse)

    # Calculate Final Validation Metric
    val_targets = val_labels_sparse.toarray()
    val_preds = (val_probs > best_threshold).astype(int)

    final_metric = f1_score(val_targets, val_preds, average="samples", zero_division=0)

    # Print metric in required format
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate F1 per sample
    # TP = intersection, Pred_Sum = count(preds), Target_Sum = count(targets)
    tp = (val_preds * val_targets).sum(axis=1)
    pred_sum = val_preds.sum(axis=1)
    target_sum = val_targets.sum(axis=1)

    denominator = pred_sum + target_sum
    # Safe divide to calculate F1
    f1_per_sample = np.divide(
        2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0
    )

    error_magnitude = 1.0 - f1_per_sample

    # Get Input Features for Correlation
    # We load raw processed data to avoid iterating loader which is slow/heavy
    val_deep = np.load(Config.VAL_DEEP_PATH)
    val_wide = scipy.sparse.load_npz(Config.VAL_WIDE_PATH)

    if Config.DEBUG:
        val_deep = val_deep[: Config.DEBUG_SIZE]
        val_wide = val_wide[: Config.DEBUG_SIZE]

    # Feature 1: Text Length (Deep) - count non-padding
    text_length = (val_deep != 0).sum(axis=1)

    # Feature 2: TF-IDF Richness (Wide) - count non-zeros
    tfidf_count = val_wide.getnnz(axis=1)

    # Feature 3: Number of Tags (Target Complexity)
    num_tags = target_sum

    # Correlation
    analysis_df = pd.DataFrame(
        {
            "error": error_magnitude,
            "text_length": text_length,
            "tfidf_count": tfidf_count,
            "num_tags": num_tags,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # --------------------------------------------------------------------------
    # 7. Submission Generation
    # --------------------------------------------------------------------------
    submission_threshold = 0.0542101508997596

    if final_metric > submission_threshold:
        generate_submission(model, test_loader, best_threshold, device)
    else:
        print(
            f"Validation metric {final_metric} <= {submission_threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
