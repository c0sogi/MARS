import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import seed_everything, get_device
from library.features import FeaturePipeline
from library.dataset import ArenaDataset
from library.model import ClassifierMLP
from library.trainer import ModelTrainer


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # Define parameters for optimized execution
    # Using full dataset and more epochs as we have sufficient time
    # Cite solution_lesson_node_00002: Optimizing features and training on full data.
    FAST_SAMPLE_SIZE = None  # Use full dataset
    MAX_EPOCHS = 20
    BATCH_SIZE = Config.BATCH_SIZE

    print(f"Starting Optimized Run: Sample Size=Full, Epochs={MAX_EPOCHS}")

    # 2. Data Loading & Processing
    # We use FeaturePipeline to vectorize text and extract scalar features.
    # load_cached_data=True allows using cached v2 features if available
    pipeline = FeaturePipeline()
    X_train, y_train, X_val, y_val, X_test, test_ids = pipeline.process_data(
        load_cached_data=True, debug_sample_size=FAST_SAMPLE_SIZE
    )

    # Create PyTorch Datasets
    train_dataset = ArenaDataset(X_train, y_train)
    val_dataset = ArenaDataset(X_val, y_val)
    test_dataset = ArenaDataset(X_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 3. Model Initialization
    model = ClassifierMLP(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    trainer = ModelTrainer(
        model, train_loader, val_loader, criterion, optimizer, device, Config
    )
    trainer.train(epochs=MAX_EPOCHS)

    # 5. Validation Assessment
    print("\n--- Validation Assessment ---")
    # Generate predictions on validation set
    val_preds = trainer.predict(val_loader)

    # Calculate Log Loss
    # sklearn.metrics.log_loss handles eps=auto (clipping) by default
    metric = log_loss(y_val, val_preds)

    # Print the required metric
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Error Magnitude (Cross Entropy Loss per sample)
    # We clip predictions manually to ensure stability for the log calculation
    epsilon = 1e-15
    val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
    # Cross Entropy: -sum(y_true * log(y_pred))
    per_sample_loss = -np.sum(y_val * np.log(val_preds_clipped), axis=1)

    # Load raw validation metadata to extract interpretable features (text lengths)
    # We must apply the same slicing (head) as the pipeline did to align indices
    # Since FAST_SAMPLE_SIZE is None, we read the whole file
    val_meta_df = pd.read_csv(Config.VAL_DATA_PATH)
    if FAST_SAMPLE_SIZE is not None:
        val_meta_df = val_meta_df.head(FAST_SAMPLE_SIZE)

    # Compute text length features
    val_meta_df["prompt_len"] = val_meta_df["prompt"].fillna("").astype(str).apply(len)
    val_meta_df["response_a_len"] = (
        val_meta_df["response_a"].fillna("").astype(str).apply(len)
    )
    val_meta_df["response_b_len"] = (
        val_meta_df["response_b"].fillna("").astype(str).apply(len)
    )
    val_meta_df["len_diff_abs"] = (
        val_meta_df["response_a_len"] - val_meta_df["response_b_len"]
    ).abs()

    # Add error magnitude to dataframe
    val_meta_df["error_magnitude"] = per_sample_loss

    # Calculate correlation between error magnitude and input features
    analysis_features = [
        "prompt_len",
        "response_a_len",
        "response_b_len",
        "len_diff_abs",
    ]
    correlations = val_meta_df[analysis_features].corrwith(
        val_meta_df["error_magnitude"]
    )

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 7. Submission Generation
    # Only generate submission if metric is below threshold
    THRESHOLD = 1.0680422618700864
    if metric < THRESHOLD:
        print("\n--- Generating Submission ---")
        print(f"Metric {metric} < {THRESHOLD}. Proceeding with submission.")
        # Generate predictions on test set
        test_preds = trainer.predict(test_loader)

        # Format submission DataFrame
        submission_df = pd.DataFrame(
            test_preds, columns=["winner_model_a", "winner_model_b", "winner_tie"]
        )
        submission_df.insert(0, "id", test_ids)

        # Save to disk
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nMetric {metric} >= {THRESHOLD}. Skipping submission generation.")


if __name__ == "__main__":
    main()
