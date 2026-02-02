import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import RSNADataset
from library.model import RSNAModel
from library.train import run_training
from library.inference import predict


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("runfile")
    logger.info("Starting runfile execution...")

    # 2. Training
    # Run training using Config epochs (10) to allow convergence (Cite solution_lesson_node_00007)
    # This will save the best model to Config.MODEL_SAVE_PATH
    logger.info("Initiating training...")
    run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=False)

    # 3. Validation & Metric Calculation
    logger.info("Starting validation assessment...")

    device = torch.device(Config.DEVICE)

    # Load the best model
    model = RSNAModel(pretrained=False)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logger.error("Model checkpoint not found after training!")
        return

    model.to(device)
    model.eval()

    # Load Validation Data
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    val_dataset = RSNADataset(val_df, subset="val", transform=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Inference Loop
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)

            # Use mixed precision for fast inference
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 8)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 8)

    # Calculate Weighted Multi-label Logarithmic Loss
    # Columns: [C1, C2, C3, C4, C5, C6, C7, patient_overall]
    # Weights: 1/14 for C1-C7, 7/14 for patient_overall
    weights = np.array([1 / 14] * 7 + [7 / 14])

    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    preds_clipped = np.clip(all_preds, epsilon, 1 - epsilon)

    # Compute weighted log loss element-wise
    # L_ij = -w_j * [y_ij * log(p_ij) + (1-y_ij) * log(1-p_ij)]
    term1 = all_targets * np.log(preds_clipped)
    term2 = (1 - all_targets) * np.log(1 - preds_clipped)
    loss_matrix = -weights * (term1 + term2)  # Shape (N, 8)

    # "loss is averaged across all rows"
    # Total rows = N_samples * 8
    final_metric = np.sum(loss_matrix) / (loss_matrix.shape[0] * loss_matrix.shape[1])

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    logger.info("Performing failure analysis...")

    # Calculate loss per sample (sum of weighted losses for that exam)
    # We use the sum here to represent the total error "magnitude" for that patient
    sample_losses = np.sum(loss_matrix, axis=1)

    # Extract input feature: Number of slices (Depth)
    # We count files in the directory as a proxy for depth without loading pixels
    num_slices_list = []
    for rel_path in val_df["image_path"]:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        try:
            # Count files in the directory
            count = len(
                [
                    name
                    for name in os.listdir(full_path)
                    if os.path.isfile(os.path.join(full_path, name))
                ]
            )
        except Exception:
            count = 0
        num_slices_list.append(count)

    num_slices_arr = np.array(num_slices_list)

    # Calculate Correlation
    if np.std(sample_losses) > 0 and np.std(num_slices_arr) > 0:
        correlation = np.corrcoef(sample_losses, num_slices_arr)[0, 1]
        print(f"Correlation between error magnitude and num_slices: {correlation}")
    else:
        print("Correlation could not be calculated (zero variance).")

    # 5. Conditional Submission
    threshold = 0.06429807151236185
    if final_metric < threshold:
        logger.info(
            f"Metric {final_metric} is below threshold {threshold}. Generating submission..."
        )
        predict(debug=False)
    else:
        logger.info(
            f"Metric {final_metric} is not below threshold {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
