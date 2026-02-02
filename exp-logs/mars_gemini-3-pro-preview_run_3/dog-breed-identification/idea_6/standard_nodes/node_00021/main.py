import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
import torch.nn.functional as F

# Import library components
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_loaders
from library.trainer import fit_model, predict_with_tta, get_labels
from library.model_factory import create_model
from library.calibration import TemperatureScaler

# Initialize Logger
logger = get_logger(name="main_orchestrator")


def analyze_failures(val_loader, val_probs, val_labels):
    """
    Performs failure analysis by correlating error magnitude with image metadata.
    """
    logger.info("Starting Failure Analysis...")

    # Calculate per-sample Log Loss (Cross Entropy)
    # val_probs: (N, C), val_labels: (N,)
    # We need the probability assigned to the true class

    # Convert labels to one-hot or index into probs
    n_samples = len(val_labels)
    true_class_probs = val_probs[np.arange(n_samples), val_labels.numpy()]

    # Clip to avoid log(0)
    epsilon = 1e-15
    true_class_probs = np.clip(true_class_probs, epsilon, 1 - epsilon)

    # Error magnitude = -log(p_true)
    error_magnitudes = -np.log(true_class_probs)

    # Extract Metadata Features
    # val_loader.dataset is DogDataset, which has .df attribute
    val_df = val_loader.dataset.df

    widths = []
    heights = []
    file_sizes = []

    # We need to access the files. The paths in df are relative.
    # We assume the order in val_loader.dataset matches val_labels/val_probs
    # (shuffle=False is set in get_loaders for val)

    for _, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # File Size
        try:
            size = os.path.getsize(full_path)
        except:
            size = 0
        file_sizes.append(size)

        # Dimensions
        try:
            with Image.open(full_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except:
            widths.append(0)
            heights.append(0)

    # Calculate Correlations
    features = {"File Size": file_sizes, "Width": widths, "Height": heights}

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    for name, values in features.items():
        if len(values) != n_samples:
            logger.warning(f"Length mismatch for {name}. Skipping.")
            continue

        corr, p_val = pearsonr(error_magnitudes, values)
        print(f"  {name}: Correlation = {corr:.4f}, p-value = {p_val:.4f}")


def main():
    # 1. Setup
    # Override Config for Fast Baseline
    Config.EPOCHS = 12
    Config.SWA_START_EPOCH = 8
    Config.FREEZE_BACKBONE_EPOCHS = 1

    seed_everything(Config.SEED)

    logger.info("Configuration set for Fast Baseline.")
    logger.info(f"Epochs: {Config.EPOCHS}, SWA Start: {Config.SWA_START_EPOCH}")

    # 2. Data Loading
    train_loader, val_loader, test_loader, class_list = get_loaders(
        load_cached_data=True
    )

    # Get validation labels for calibration and metric calculation
    val_labels = get_labels(val_loader)

    # Storage for ensemble predictions
    ensemble_val_probs = []
    ensemble_test_probs = []

    # 3. Training & Inference Loop
    for model_name in Config.MODEL_ARCHS:
        logger.info(f"Processing Model: {model_name}")

        # A. Train
        # fit_model handles the 2-phase training and SWA saving
        saved_model_path = fit_model(model_name, train_loader, val_loader)

        # B. Load Model for Inference
        logger.info(f"Loading model from {saved_model_path}")
        model = create_model(model_name, num_classes=len(class_list), pretrained=False)
        state_dict = torch.load(saved_model_path, map_location=Config.DEVICE)
        model.load_state_dict(state_dict, strict=False)
        model.to(Config.DEVICE)
        model.eval()

        # C. Validation Inference (with TTA)
        logger.info(f"[{model_name}] generating validation logits...")
        val_logits = predict_with_tta(model, val_loader, device=Config.DEVICE)

        # D. Calibration
        scaler = None
        if Config.USE_TEMP_SCALING:
            logger.info(f"[{model_name}] calibrating...")
            scaler = TemperatureScaler()
            scaler.fit(val_logits, val_labels)

            # Get calibrated val probabilities
            val_probs = scaler.get_probabilities(val_logits).detach().cpu().numpy()
        else:
            val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()

        ensemble_val_probs.append(val_probs)

        # E. Test Inference (with TTA)
        logger.info(f"[{model_name}] generating test logits...")
        test_logits = predict_with_tta(model, test_loader, device=Config.DEVICE)

        # Apply Calibration to Test
        if scaler:
            test_probs = scaler.get_probabilities(test_logits).detach().cpu().numpy()
        else:
            test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()

        ensemble_test_probs.append(test_probs)

        # Cleanup to save memory
        del model, val_logits, test_logits
        torch.cuda.empty_cache()

    # 4. Ensemble Aggregation
    logger.info("Aggregating Ensemble...")
    avg_val_probs = np.mean(ensemble_val_probs, axis=0)
    avg_test_probs = np.mean(ensemble_test_probs, axis=0)

    # 5. Validation Metric
    # Calculate Multi Class Log Loss
    # labels need to be passed; log_loss expects (n_samples,) labels or one-hot
    # val_labels is a torch tensor of class indices
    final_metric = log_loss(
        val_labels.numpy(), avg_val_probs, labels=list(range(len(class_list)))
    )

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    analyze_failures(val_loader, avg_val_probs, val_labels)

    # 7. Submission Generation
    threshold = 0.14004325100369866

    if final_metric < threshold:
        logger.info(
            f"Metric ({final_metric}) is better than threshold ({threshold}). Generating submission."
        )

        # Get Test IDs
        test_ids = []
        for batch in test_loader:
            # Test loader yields (image, id)
            _, batch_ids = batch
            test_ids.extend(batch_ids)

        # Create DataFrame
        df_sub = pd.DataFrame(avg_test_probs, columns=class_list)
        df_sub.insert(0, "id", test_ids)

        # Save
        output_dir = "./submission"
        os.makedirs(output_dir, exist_ok=True)
        sub_path = os.path.join(output_dir, "submission.csv")
        df_sub.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}")
    else:
        logger.info(
            f"Metric ({final_metric}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
