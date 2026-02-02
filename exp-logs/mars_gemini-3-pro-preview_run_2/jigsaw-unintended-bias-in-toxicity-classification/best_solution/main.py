import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.data import prepare_data, ToxicityDataset
from library.engine import Engine
from library.utils import seed_everything


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override Config for Fast Baseline execution
    # We reduce epochs and will subsample data to ensure < 2 hours runtime
    Config.EPOCHS = 2
    # Ensure batch size is appropriate
    Config.TRAIN_BATCH_SIZE = 32
    Config.VALID_BATCH_SIZE = 64
    Config.TEST_BATCH_SIZE = 64

    # Define subsample size for training (e.g., 150,000 samples)
    TRAIN_SUBSAMPLE_SIZE = 150000

    # --------------------------------------------------------------------------
    # 2. Data Preparation
    # --------------------------------------------------------------------------
    print("\n=== Preparing Data ===")

    # --- Train Data ---
    # Load full training data (cached if available)
    t_input, t_mask, t_target, t_identities, t_ids = prepare_data(
        "train", load_cached_data=True
    )

    # Subsample for speed
    if len(t_input) > TRAIN_SUBSAMPLE_SIZE:
        print(f"Subsampling training data: {len(t_input)} -> {TRAIN_SUBSAMPLE_SIZE}")
        # Use fixed seed for subsampling indices to ensure reproducibility
        rng = np.random.RandomState(Config.SEED)
        indices = rng.choice(len(t_input), TRAIN_SUBSAMPLE_SIZE, replace=False)

        t_input = t_input[indices]
        t_mask = t_mask[indices]
        t_target = t_target[indices]
        t_identities = t_identities[indices]
        t_ids = t_ids[indices]

    train_dataset = ToxicityDataset(t_input, t_mask, t_target, t_identities, t_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # --- Validation Data ---
    # Use full validation set for accurate metrics
    v_input, v_mask, v_target, v_identities, v_ids = prepare_data(
        "validation", load_cached_data=True
    )
    val_dataset = ToxicityDataset(v_input, v_mask, v_target, v_identities, v_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Test Data ---
    test_input, test_mask, _, _, test_ids = prepare_data("test", load_cached_data=True)
    test_dataset = ToxicityDataset(test_input, test_mask, sample_ids=test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 3. Training Loop
    # --------------------------------------------------------------------------
    print("\n=== Starting Training ===")
    engine = Engine(device)

    # Run training (includes evaluation at each epoch and saving best model)
    # This also generates the submission.csv automatically at the end.
    engine.run_training(train_loader, val_loader, test_loader)

    # --------------------------------------------------------------------------
    # 4. Final Evaluation & Metric Reporting
    # --------------------------------------------------------------------------
    print("\n=== Final Evaluation ===")

    # Load the best model saved during training
    engine.model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    )
    engine.model.eval()

    # Calculate metrics on validation set
    metrics = engine.evaluate(val_loader)
    final_score = metrics["final_score"]

    # REQUIRED: Print the final validation metric strictly
    print(f"Final Validation Metric: {final_score}")

    # --------------------------------------------------------------------------
    # 5. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")

    # Collect raw predictions and targets for analysis
    all_preds = []
    all_targets = []
    all_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].numpy()

            # Compute sequence length (sum of attention mask)
            lengths = attention_mask.sum(dim=1).cpu().numpy()
            all_lengths.append(lengths)

            # Trim and Predict
            input_ids, attention_mask = engine.trim_batch(input_ids, attention_mask)
            logits, _ = engine.model(input_ids, attention_mask)
            preds = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_lengths = np.concatenate(all_lengths)

    # Calculate Error Magnitude
    errors = np.abs(all_preds - all_targets)

    # Build Analysis DataFrame
    analysis_df = pd.DataFrame(
        {"error": errors, "target": all_targets, "text_length": all_lengths}
    )

    # Add Identity Columns (using v_identities which aligns with val_loader)
    for i, col_name in enumerate(Config.IDENTITY_COLUMNS):
        analysis_df[col_name] = v_identities[:, i]

    # Compute Correlations
    correlations = analysis_df.corrwith(analysis_df["error"])

    # Drop the 'error' self-correlation and sort
    correlations = correlations.drop("error").sort_values(ascending=False)

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # --------------------------------------------------------------------------
    # 6. Submission Logic
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9053225152942936

    if final_score > THRESHOLD:
        print(f"\nValidation Score ({final_score}) exceeds threshold ({THRESHOLD}).")
        print(f"Submission file preserved at {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation Score ({final_score}) does NOT exceed threshold ({THRESHOLD})."
        )
        print("Discarding submission file...")
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)
            print("Submission file removed.")


if __name__ == "__main__":
    main()
