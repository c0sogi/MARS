import os
import sys
import torch
import pandas as pd
import numpy as np
import copy
import time
from sklearn.metrics import f1_score

# Import from provided library files
from library.config import Config, seed_everything
from library.dataset import create_dataloaders
from library.model import EfficientNetB4Native
from library.engine import (
    train_one_epoch,
    evaluate,
    configure_model_for_stage,
    get_optimizer,
    get_scheduler,
)


def run_failure_analysis(val_df, all_targets, all_preds):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    print("\n" + "=" * 30)
    print("FAILURE ANALYSIS")
    print("=" * 30)

    # Create a local copy to avoid modifying the original dataframe
    df_analysis = val_df.copy()

    # Ensure lengths match
    if len(df_analysis) != len(all_targets):
        print(
            f"Warning: Metadata length ({len(df_analysis)}) matches targets ({len(all_targets)})?"
        )
        # If debug mode was used, we need to slice the dataframe
        df_analysis = df_analysis.iloc[: len(all_targets)].copy()

    df_analysis["target"] = all_targets
    df_analysis["prediction"] = all_preds

    # Calculate Error (1 for incorrect, 0 for correct)
    df_analysis["error"] = (df_analysis["target"] != df_analysis["prediction"]).astype(
        int
    )

    print(f"Total Samples: {len(df_analysis)}")
    print(f"Total Errors: {df_analysis['error'].sum()}")
    print(f"Error Rate: {df_analysis['error'].mean():.4f}")

    # Identify numerical columns for correlation
    # Based on EDA: frame_num, seq_num_frames, location, width, height
    numerical_cols = ["frame_num", "seq_num_frames", "location", "width", "height"]
    available_cols = [c for c in numerical_cols if c in df_analysis.columns]

    if available_cols:
        print("\nCorrelation between Error and Metadata Features:")
        correlations = (
            df_analysis[available_cols + ["error"]].corr()["error"].drop("error")
        )
        print(correlations)

        # Find feature with highest absolute correlation
        max_corr_feat = correlations.abs().idxmax()
        print(
            f"\nFeature most associated with error: {max_corr_feat} (Corr: {correlations[max_corr_feat]:.4f})"
        )
    else:
        print("No numerical metadata features available for correlation analysis.")


def generate_submission(model, test_loader, test_df, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("\n" + "=" * 30)
    print("GENERATING SUBMISSION")
    print("=" * 30)

    device = Config.DEVICE
    model.eval()

    all_preds = []

    # Inference loop
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    # Prepare submission dataframe
    # Ensure we use the exact IDs from the test metadata
    # If debug mode was used, test_df might be sliced, but for submission we usually need full.
    # However, create_dataloaders slices test_df in debug mode too.
    # We assume the loader and df are aligned.

    submission_df = pd.DataFrame(
        {"Id": test_df["Id"].iloc[: len(all_preds)], "Predicted": all_preds}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission_df.head())


def main():
    # 1. Setup
    # Override Config for Fast Baseline
    Config.EPOCHS_STAGE1 = 1  # Reduce from 5 to 1 for speed
    Config.EPOCHS_STAGE2 = 3  # Reduce from 10 to 3 for speed

    seed_everything(Config.SEED)

    print(f"Using Device: {Config.DEVICE}")

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader, test_loader = create_dataloaders(load_cached_data=True)

    # Load validation metadata for failure analysis later
    val_meta_path = Config.VAL_METADATA
    val_df = pd.read_csv(val_meta_path)

    # Load test metadata for submission
    test_meta_path = Config.TEST_METADATA
    test_df = pd.read_csv(test_meta_path)

    # 3. Model Initialization
    print("Initializing Model...")
    model = EfficientNetB4Native()
    model.to(Config.DEVICE)

    criterion = torch.nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    # 4. Stage 1: Head Alignment (Frozen Backbone)
    print("\n" + "=" * 30)
    print("STAGE 1: Head Alignment")
    print("=" * 30)

    configure_model_for_stage(model, stage=1)
    optimizer = get_optimizer(model, stage=1)

    for epoch in range(Config.EPOCHS_STAGE1):
        print(f"Epoch {epoch+1}/{Config.EPOCHS_STAGE1}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            Config.DEVICE,
            criterion,
            Config.GRAD_ACCUM_STEPS,
        )
        val_loss, val_f1 = evaluate(model, val_loader, Config.DEVICE, criterion)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())

    # 5. Stage 2: Fine-Tuning (Unfrozen Top Blocks)
    print("\n" + "=" * 30)
    print("STAGE 2: Fine-Tuning")
    print("=" * 30)

    configure_model_for_stage(model, stage=2)
    optimizer = get_optimizer(model, stage=2)
    scheduler = get_scheduler(optimizer, Config.EPOCHS_STAGE2)

    for epoch in range(Config.EPOCHS_STAGE2):
        print(f"Epoch {epoch+1}/{Config.EPOCHS_STAGE2}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            Config.DEVICE,
            criterion,
            Config.GRAD_ACCUM_STEPS,
        )
        val_loss, val_f1 = evaluate(model, val_loader, Config.DEVICE, criterion)

        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"New Best F1: {best_f1:.4f}")

    # 6. Final Evaluation & Failure Analysis
    print("\n" + "=" * 30)
    print("FINAL EVALUATION")
    print("=" * 30)

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Get predictions on validation set for metrics and analysis
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    final_macro_f1 = f1_score(all_targets, all_preds, average="macro")

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_macro_f1}")

    # Failure Analysis
    run_failure_analysis(val_df, all_targets, all_preds)

    # 7. Submission
    THRESHOLD = 0.3978880094708815

    if final_macro_f1 > THRESHOLD:
        print(
            f"\nValidation metric ({final_macro_f1}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        submission_path = "./submission/submission.csv"
        generate_submission(model, test_loader, test_df, submission_path)
    else:
        print(
            f"\nValidation metric ({final_macro_f1}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
