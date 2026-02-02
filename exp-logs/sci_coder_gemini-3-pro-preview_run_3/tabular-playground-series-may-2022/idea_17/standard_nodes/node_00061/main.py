import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.preprocessing import get_preprocessed_data
from library.dataset import ManufacturingDataset
from library.engine import train_model, evaluate, generate_submission, set_seed
from library.model import TreeFunnelEnsemble


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Override Config for Fast Baseline
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 2048

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Set seeds for reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading & Subsampling
    # ==========================================
    print("Loading preprocessed data...")
    train_data, val_data, test_data, vocab_sizes = get_preprocessed_data(
        load_from_cache=True
    )

    # Subsample training data for speed (Fast Baseline requirement)
    MAX_TRAIN_SAMPLES = 100000
    total_train = len(train_data["target"])

    if total_train > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {total_train} to {MAX_TRAIN_SAMPLES} samples..."
        )
        indices = np.random.choice(total_train, MAX_TRAIN_SAMPLES, replace=False)
        train_data["cat"] = train_data["cat"][indices]
        train_data["cont"] = train_data["cont"][indices]
        train_data["target"] = train_data["target"][indices]

    # ==========================================
    # 3. Dataset & DataLoader Creation
    # ==========================================
    print("Creating DataLoaders...")
    train_dataset = ManufacturingDataset(
        train_data["cat"], train_data["cont"], train_data["target"]
    )
    val_dataset = ManufacturingDataset(
        val_data["cat"], val_data["cont"], val_data["target"]
    )
    test_dataset = ManufacturingDataset(
        test_data["cat"], test_data["cont"], targets=None
    )

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

    # ==========================================
    # 4. Model Training
    # ==========================================
    cont_dim = train_data["cont"].shape[1]
    print("Starting training...")

    # train_model handles initialization, loop, and saving best model
    best_auc = train_model(train_loader, val_loader, vocab_sizes, cont_dim)

    # ==========================================
    # 5. Final Evaluation
    # ==========================================
    print("Performing final validation evaluation...")

    # Load the best model saved during training
    model = TreeFunnelEnsemble(vocab_sizes, cont_dim)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Calculate metric on full validation set
    criterion = torch.nn.BCEWithLogitsLoss()
    val_loss, final_val_auc = evaluate(model, val_loader, criterion, device)

    # Print exactly as required
    print(f"Final Validation Metric: {final_val_auc}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("Performing failure analysis...")

    all_targets = []
    all_preds = []
    all_cont_inputs = []

    # Inference loop for analysis
    with torch.no_grad():
        for cat_x, cont_x, targets in val_loader:
            cat_x = cat_x.to(device)
            cont_x_dev = cont_x.to(device)

            outputs = model(cat_x, cont_x_dev)

            # Average predictions across heads
            probs_sum = 0
            for out in outputs:
                probs_sum += torch.sigmoid(out)
            avg_probs = probs_sum / len(outputs)

            all_targets.append(targets.numpy())
            all_preds.append(avg_probs.cpu().numpy().flatten())
            all_cont_inputs.append(cont_x.numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_cont_inputs = np.concatenate(all_cont_inputs, axis=0)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # Calculate correlations
    print("Correlation between Error Magnitude and Continuous Features:")
    correlations = []
    for i in range(all_cont_inputs.shape[1]):
        feat_vals = all_cont_inputs[:, i]
        # Avoid division by zero if feature is constant
        if np.std(feat_vals) > 0 and np.std(errors) > 0:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
        else:
            corr = 0.0
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Print top 10 correlations
    for idx, corr in correlations[:10]:
        print(f"Feature cont_{idx:02d}: {corr:.6f}")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    THRESHOLD = 0.9971550270448856

    if final_val_auc > THRESHOLD:
        print(
            f"Validation metric {final_val_auc} > {THRESHOLD}. Generating submission..."
        )
        generate_submission(test_loader, test_data["ids"], vocab_sizes, cont_dim)
    else:
        print(f"Validation metric {final_val_auc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
