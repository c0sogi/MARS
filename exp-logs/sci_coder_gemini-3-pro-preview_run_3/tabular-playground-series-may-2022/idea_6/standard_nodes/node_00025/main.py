import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, get_device
from library.preprocessing import DataProcessor
from library.dataset import ManufacturingDataset
from library.model import HybridTransformerFunnel
from library.engine import Engine


def main():
    # 1. Setup and Configuration
    # Set random seeds for reproducibility
    set_seed(Config.SEED)

    # Detect device (GPU/CPU)
    device = get_device()

    # Override Config for Fast Baseline Execution
    # Reducing epochs and limiting data ensures the script completes quickly
    Config.EPOCHS = 5
    MAX_TRAIN_SAMPLES = 100000

    # 2. Data Processing
    processor = DataProcessor()
    # Load data, utilizing cache if available
    train_df, val_df, test_df, vocab_sizes = processor.process_data(
        load_cached_data=True
    )

    # Subsample training data for speed
    if len(train_df) > MAX_TRAIN_SAMPLES:
        train_df = train_df.iloc[:MAX_TRAIN_SAMPLES].reset_index(drop=True)

    # 3. Dataset and DataLoader Initialization
    train_dataset = ManufacturingDataset(train_df, is_test=False)
    val_dataset = ManufacturingDataset(val_df, is_test=False)
    test_dataset = ManufacturingDataset(test_df, is_test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Model Initialization
    continuous_dim = len(Config.CONTINUOUS_FEATURE_NAMES)
    model = HybridTransformerFunnel(
        vocab_sizes=vocab_sizes, continuous_dim=continuous_dim
    )

    # 5. Training Loop
    engine = Engine(model, device)
    engine.fit(train_loader, val_loader)

    # 6. Final Validation Assessment
    criterion = nn.BCEWithLogitsLoss()
    avg_loss, final_auc = engine.evaluate(val_loader, criterion)

    # Print the final metric in the required format
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Generate predictions on the validation set
    val_probs = engine.predict(val_loader)
    val_targets = val_df[Config.TARGET_COL].values

    # Calculate absolute prediction errors
    errors = np.abs(val_targets - val_probs)

    # Calculate correlation between error magnitude and continuous features
    correlations = []
    for feature in Config.CONTINUOUS_FEATURE_NAMES:
        if feature in val_df.columns:
            feat_values = val_df[feature].values
            # Avoid correlation calculation if constant
            if np.std(feat_values) > 0 and np.std(errors) > 0:
                corr = np.corrcoef(feat_values, errors)[0, 1]
                correlations.append((feature, corr))
            else:
                correlations.append((feature, 0.0))

    # Sort features by the absolute value of their correlation with error
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with prediction error:")
    for feat, corr in correlations[:5]:
        print(f"{feat}: {corr:.6f}")

    # 8. Submission Generation
    SUBMISSION_THRESHOLD = 0.9971550270448856

    if final_auc > SUBMISSION_THRESHOLD:
        test_ids = test_df[Config.ID_COL].values
        engine.generate_submission(test_loader, test_ids)
    else:
        print(
            f"Validation AUC ({final_auc}) did not exceed threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
