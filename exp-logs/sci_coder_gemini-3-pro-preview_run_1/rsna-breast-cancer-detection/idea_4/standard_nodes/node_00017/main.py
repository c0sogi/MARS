import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import Config, set_seed
from library.utils import get_device, probabilistic_f1
from library.data import get_dataloaders
from library.model import EarlyFusionEfficientNet
from library.train import run_training


def load_best_model(device):
    """
    Loads the best model checkpoint saved during training.
    """
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    model = EarlyFusionEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # Weights will be loaded from checkpoint
        dropout_prob=Config.MODALITY_DROPOUT_PROB,
        in_chans=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    )

    if os.path.exists(model_path):
        print(f"Loading best model from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model = model.to(device)
    model.eval()
    return model


def run_inference(model, loader, device, is_test=False):
    """
    Runs inference on a dataloader.
    Returns lists of predictions, targets (if not test), and prediction_ids (if test).
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", disable=True):
            inputs = batch["image"].to(device)

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

            all_preds.extend(probs)

            if is_test:
                all_ids.extend(batch["prediction_id"])
            else:
                targets = batch["label"].cpu().numpy().ravel()
                all_targets.extend(targets)

    return np.array(all_preds), np.array(all_targets), all_ids


def perform_failure_analysis(val_df, preds, targets):
    """
    Analyzes model errors against metadata features.
    """
    print("\n==== Failure Analysis ====")

    # Ensure alignment
    analysis_df = val_df.copy()
    analysis_df["prediction"] = preds
    analysis_df["target"] = targets
    analysis_df["error"] = np.abs(analysis_df["target"] - analysis_df["prediction"])

    # Preprocess features for correlation
    # Map Density A-D to 1-4
    density_map = {"A": 1, "B": 2, "C": 3, "D": 4}
    analysis_df["density_encoded"] = analysis_df["density"].map(density_map)

    # Handle NaNs in density/age for correlation calculation
    analysis_df["density_encoded"] = analysis_df["density_encoded"].fillna(
        analysis_df["density_encoded"].mean()
    )
    analysis_df["age"] = analysis_df["age"].fillna(analysis_df["age"].mean())

    # Features to check
    features = ["age", "implant", "density_encoded", "machine_id"]

    print("Correlation between Absolute Error and Features:")
    for feat in features:
        if feat in analysis_df.columns:
            corr = analysis_df["error"].corr(analysis_df[feat])
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: Not found")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Training
    # We limit epochs to 5 for a fast baseline as per requirements
    print("Starting Training Pipeline...")
    _, best_score = run_training(
        debug=False, num_epochs=5, batch_size=Config.BATCH_SIZE
    )

    # 3. Validation & Analysis
    print("\nStarting Validation & Failure Analysis...")

    # Load the best model weights
    model = load_best_model(device)

    # Get DataLoaders (re-using get_dataloaders to ensure consistent preprocessing)
    _, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, debug=False
    )

    # Validation Inference
    val_preds, val_targets, _ = run_inference(model, val_loader, device, is_test=False)

    # Compute Metric
    final_metric = probabilistic_f1(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Load Validation Metadata for Analysis
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Failure Analysis
    perform_failure_analysis(val_df, val_preds, val_targets)

    # 4. Submission
    THRESHOLD = 0.04437665641307831

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Test Inference
        test_preds, _, test_ids = run_inference(
            model, test_loader, device, is_test=True
        )

        # Create DataFrame
        sub_df = pd.DataFrame({"prediction_id": test_ids, "cancer": test_preds})

        # Aggregate by prediction_id (Max Probability across views)
        # One prediction_id (e.g., 10116_L) can have multiple images (CC, MLO views).
        # We take the max probability as the breast-level prediction.
        submission = sub_df.groupby("prediction_id")["cancer"].max().reset_index()

        # Save
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}. Rows: {len(submission)}")

        # Verify against sample submission
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)
        print(f"Sample submission rows: {len(sample_sub)}")

    else:
        print(
            f"\nValidation metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
