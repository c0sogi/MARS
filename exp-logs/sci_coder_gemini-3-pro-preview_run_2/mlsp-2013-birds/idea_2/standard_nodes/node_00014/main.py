import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import BirdDataset, get_transforms, load_dataframe
from library.model import BirdClassifier
from library.train import run_training
from library.predict import generate_predictions
from library.utils import set_seed, calculate_roc_auc


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # 1. Training
    # ---------------------------------------------------------
    print("Starting Training Pipeline...")
    # run_training handles data loading, model init, training loop, and saving best model
    run_training()

    # ---------------------------------------------------------
    # 2. Validation & Metric Calculation
    # ---------------------------------------------------------
    print("\nStarting Validation Evaluation...")
    device = Config.DEVICE

    # Load validation metadata
    df_val = load_dataframe(Config.VAL_CSV, debug=Config.DEBUG)

    # Setup validation dataset and loader
    val_dataset = BirdDataset(
        df=df_val,
        transforms=get_transforms("val"),
        img_dir=Config.FILTERED_SPECTROGRAM_DIR,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize model structure
    model = BirdClassifier(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # Weights will be loaded from checkpoint
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # Load the best checkpoint saved during training
    checkpoint_path = Config.MODEL_SAVE_PATH
    if not os.path.exists(checkpoint_path):
        print(f"Critical Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Run Inference on Validation Set
    all_targets = []
    all_preds = []
    all_rec_ids = []

    with torch.no_grad():
        for images, labels, rec_ids in val_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Collect results
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())
            all_rec_ids.extend(rec_ids.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Final Metric (Area Under ROC Curve)
    val_auc = calculate_roc_auc(all_targets, all_preds)

    # Print exactly as required
    print(f"Final Validation Metric: {val_auc}")

    # ---------------------------------------------------------
    # 3. Failure Analysis
    # ---------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate Error Magnitude per sample
    # Metric: Mean Absolute Error averaged across all 19 classes for each sample
    per_sample_mae = np.mean(np.abs(all_targets - all_preds), axis=1)

    df_error = pd.DataFrame({"rec_id": all_rec_ids, "error_magnitude": per_sample_mae})

    # Load Supplemental Features (Histogram of Segments) for correlation analysis
    hist_path = os.path.join(
        Config.INPUT_ROOT, "supplemental_data", "histogram_of_segments.txt"
    )

    if os.path.exists(hist_path):
        try:
            # Read CSV, skipping the header line to avoid parsing issues with the weird header format
            # We assume the first column is rec_id and the rest are features
            df_feats = pd.read_csv(hist_path, header=None, skiprows=1)

            # Generate column names
            feat_cols = [f"feat_{i}" for i in range(df_feats.shape[1] - 1)]
            df_feats.columns = ["rec_id"] + feat_cols

            # Merge error data with features
            df_analysis = df_error.merge(df_feats, on="rec_id", how="inner")

            if not df_analysis.empty:
                # Calculate correlation between features and error magnitude
                # We drop rec_id as it's an identifier, not a feature
                correlations = df_analysis.drop(columns=["rec_id"]).corrwith(
                    df_analysis["error_magnitude"]
                )

                # Remove the self-correlation of error_magnitude with itself
                correlations = correlations.drop("error_magnitude", errors="ignore")

                # Sort by absolute correlation strength
                abs_corrs = correlations.abs().sort_values(ascending=False)

                print("Top 5 Input Features Correlated with Error Magnitude:")
                for feat_name in abs_corrs.head(5).index:
                    corr_val = correlations[feat_name]
                    print(f"  {feat_name}: {corr_val:.6f}")
            else:
                print(
                    "Warning: No matching rec_ids found between validation set and supplemental features."
                )

        except Exception as e:
            print(
                f"Warning: Failed to process supplemental features for failure analysis. Error: {e}"
            )
    else:
        print(f"Warning: Supplemental feature file not found at {hist_path}")

    # ---------------------------------------------------------
    # 4. Submission Generation
    # ---------------------------------------------------------
    THRESHOLD = 0.8591753473154088

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_predictions()
    else:
        print(
            f"\nValidation metric ({val_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
