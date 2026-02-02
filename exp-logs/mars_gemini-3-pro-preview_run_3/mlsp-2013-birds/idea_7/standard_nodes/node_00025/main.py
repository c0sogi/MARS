import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import warnings

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import load_or_generate_data, BirdDataset
from library.model import MILResNet18
from library.trainer import run_fold

warnings.filterwarnings("ignore")


def predict_ensemble(models, loader, device):
    """
    Runs inference using an ensemble of models.
    Returns averaged probabilities, targets, and recording IDs.
    """
    for m in models:
        m.eval()

    all_preds = []
    all_targets = []
    all_rec_ids = []

    # Access the dataframe directly to retrieve rec_ids sequentially
    dataset_df = loader.dataset.df

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)

            # Aggregate predictions from all models
            batch_preds = []
            for model in models:
                logits = model(inputs)
                probs = torch.sigmoid(logits)
                batch_preds.append(probs.cpu().numpy())

            # Average probabilities across the ensemble
            avg_preds = np.mean(batch_preds, axis=0)

            all_preds.append(avg_preds)
            all_targets.append(targets.numpy())

            # Get corresponding rec_ids for this batch
            start = batch_idx * loader.batch_size
            end = start + inputs.size(0)
            batch_ids = dataset_df.iloc[start:end]["rec_id"].values
            all_rec_ids.extend(batch_ids)

    return np.concatenate(all_preds), np.concatenate(all_targets), np.array(all_rec_ids)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    # 2. Training Loop
    print("Starting 5-Fold Cross-Validation Training...")
    trained_models = []

    # Pre-generate/load data once to ensure cache is populated
    data_dict = load_or_generate_data(load_cached_data=True)

    for fold in range(Config.N_FOLDS):
        print(f"\nTraining Fold {fold}...")
        # Run training for the fold
        run_fold(fold, load_cached_data=True)

        # Load the best checkpoint for this fold
        model = MILResNet18()
        ckpt_path = os.path.join(Config.CHECKPOINTS_DIR, f"fold_{fold}_best.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        trained_models.append(model)

    # 3. Validation Evaluation
    print("\nEvaluating on Hold-out Validation Set...")
    val_df = pd.read_csv(Config.VAL_CSV)
    val_ds = BirdDataset(val_df, data_dict, augment=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_probs, val_targets, _ = predict_ensemble(trained_models, val_loader, device)

    # Compute Metric
    aucs = []
    n_classes = val_targets.shape[1]
    for i in range(n_classes):
        if len(np.unique(val_targets[:, i])) > 1:
            try:
                score = roc_auc_score(val_targets[:, i], val_probs[:, i])
                aucs.append(score)
            except ValueError:
                pass

    if len(aucs) > 0:
        final_val_auc = np.mean(aucs)
    else:
        final_val_auc = 0.5

    print(f"Final Validation Metric: {final_val_auc}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Error metric: Mean Absolute Error per sample averaged across classes
    mae_per_sample = np.mean(np.abs(val_probs - val_targets), axis=1)

    # Feature 1: Number of labels (Ground Truth complexity)
    num_labels = np.sum(val_targets, axis=1)

    # Feature 2: Signal Energy (Spectrogram intensity)
    energies = []
    for _, row in val_df.iterrows():
        fpath = row["file_path"]
        if fpath in data_dict:
            spec = data_dict[fpath]
            energies.append(np.mean(spec))
        else:
            energies.append(0.0)

    # Compute Correlations
    if np.std(mae_per_sample) > 1e-9:
        corr_n_labels = np.corrcoef(mae_per_sample, num_labels)[0, 1]
        corr_energy = np.corrcoef(mae_per_sample, energies)[0, 1]
        print(f"Correlation (Error vs Num Labels): {corr_n_labels}")
        print(f"Correlation (Error vs Signal Energy): {corr_energy}")
    else:
        print("Insufficient variance in errors for correlation analysis.")

    # 5. Submission
    threshold = 0.9072993371210134
    if final_val_auc > threshold:
        print(f"\nMetric {final_val_auc} > {threshold}. Generating submission...")
    else:
        print(
            f"\nMetric {final_val_auc} <= {threshold}. Generating submission anyway to satisfy task requirements..."
        )

    test_df = pd.read_csv(Config.TEST_CSV)
    test_ds = BirdDataset(test_df, data_dict, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_probs, _, test_rec_ids = predict_ensemble(trained_models, test_loader, device)

    # Format for submission: Id,Probability
    # Id = rec_id * 100 + species_id
    submission_data = []
    for i, rec_id in enumerate(test_rec_ids):
        probs = test_probs[i]
        for species_idx, prob in enumerate(probs):
            submission_data.append(
                {"Id": int(rec_id * 100 + species_idx), "Probability": prob}
            )

    submission_df = pd.DataFrame(submission_data)
    submission_df = submission_df.sort_values("Id")

    save_path = "./submission/submission.csv"
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


if __name__ == "__main__":
    main()
