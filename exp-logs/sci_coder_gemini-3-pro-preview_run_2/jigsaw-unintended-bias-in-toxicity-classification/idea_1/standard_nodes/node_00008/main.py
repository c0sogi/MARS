import os
import time
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import (
    SEED,
    DEVICE,
    EPOCHS,
    PATIENCE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SUBMISSION_DIR,
    IDENTITY_COLUMNS,
    TARGET_COL,
    ID_COL,
)
from library.data_loader import get_dataloaders
from library.model import MultiTaskLSTM
from library.trainer import Trainer, set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def perform_failure_analysis(model, val_loader, device):
    """
    Runs inference on the validation set to gather predictions and features,
    then calculates the correlation between error magnitude and features.
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    all_preds = []
    all_targets = []
    all_aux = []
    all_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            target = batch["target"].to(device)
            aux_target = batch["aux_target"].to(device)

            # Calculate sequence length (non-padding tokens)
            # Padding index is 0
            lengths = (input_ids != 0).sum(dim=1).cpu().numpy()

            # Forward pass
            tox_pred, _ = model(input_ids)

            all_preds.extend(tox_pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
            all_aux.extend(aux_target.cpu().numpy())
            all_lengths.extend(lengths)

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(
        {"prediction": all_preds, "target": all_targets, "text_length": all_lengths}
    )

    # Add identity columns
    aux_data = np.array(all_aux)
    for i, col in enumerate(IDENTITY_COLUMNS):
        df_analysis[col] = aux_data[:, i]

    # Calculate Error Magnitude
    df_analysis["error"] = (df_analysis["prediction"] - df_analysis["target"]).abs()

    # Calculate correlations with Error
    # We correlate Error with: Target, Text Length, and Identity Indicators
    cols_to_correlate = ["target", "text_length"] + IDENTITY_COLUMNS
    correlations = df_analysis[cols_to_correlate].corrwith(df_analysis["error"])

    print("Correlation between Error Magnitude and Features:")
    print(correlations.sort_values(ascending=False).to_string())

    return correlations


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    # 2. Data Loading
    # We use load_cached_data=True to use preprocessed files if available
    # We use debug=False to train on the full dataset for a valid baseline
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = MultiTaskLSTM()
    trainer = Trainer(model, device=DEVICE)

    # 4. Training Loop
    print("Starting Training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Train
        train_metrics = trainer.train_epoch(train_loader)

        # Validate
        val_metrics = trainer.evaluate(val_loader)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_metrics['loss']:.5f}")
        print(f"  Val Loss:   {val_metrics['loss']:.5f}")
        print(f"  Val Score:  {val_metrics['score']:.5f}")

        # Early Stopping Logic
        current_val_tox_loss = val_metrics["tox_loss"]
        if current_val_tox_loss < trainer.best_val_loss:
            print(
                f"  [Improvement] Saving model (Loss: {trainer.best_val_loss:.5f} -> {current_val_tox_loss:.5f})"
            )
            trainer.best_val_loss = current_val_tox_loss
            trainer.save_model(MODEL_SAVE_PATH)
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1
            print(f"  [No Improvement] Patience: {trainer.patience_counter}/{PATIENCE}")

        if trainer.patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training finished in {time.time() - start_time:.2f}s")

    # 5. Final Evaluation
    print("Loading best model for final evaluation...")
    trainer.load_model(MODEL_SAVE_PATH)

    final_val_metrics = trainer.evaluate(val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_metrics['score']}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, DEVICE)

    # 7. Submission Generation
    if final_val_metrics["score"] > 0.9053225152942936:
        print("\nGenerating Submission...")
        submission_df = trainer.predict(test_loader)

        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation score {final_val_metrics['score']} did not beat threshold 0.894277. Skipping submission."
        )

    # Verify submission
    print(f"Submission shape: {submission_df.shape}")
    print(submission_df.head())


if __name__ == "__main__":
    main()
