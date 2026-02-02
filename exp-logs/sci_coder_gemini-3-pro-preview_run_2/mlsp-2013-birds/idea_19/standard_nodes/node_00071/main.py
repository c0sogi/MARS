import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import library modules
from library.utils import set_seed, worker_init_fn, compute_metric
from library.loss import AsymmetricLoss
from library.dataset import BirdDataset
from library.models import get_model
from library.engine import train_one_epoch, validate, inference_fn


def load_tabular_features(filepath):
    """
    Loads the histogram of segments features for failure analysis.
    """
    if not os.path.exists(filepath):
        return None

    data_rows = []
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        start_idx = 0
        if "rec_id" in lines[0]:
            start_idx = 1

        for line in lines[start_idx:]:
            parts = line.strip().split(",")
            if len(parts) > 1:
                rec_id = int(parts[0])
                features = [float(x) for x in parts[1:]]
                data_rows.append([rec_id] + features)

        num_features = len(data_rows[0]) - 1
        cols = ["rec_id"] + [f"feat_{i}" for i in range(num_features)]
        return pd.DataFrame(data_rows, columns=cols)
    except Exception as e:
        print(f"Error loading tabular features: {e}")
        return None


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    # Dataset is ~206 samples. 16 batch size -> ~13 steps per epoch.
    # Target 1000 steps -> ~75 epochs. Let's do 60 epochs to be safe on time.
    NUM_EPOCHS = 60

    # Architectures for the ensemble
    ARCHITECTURES = ["resnet18", "efficientnet_b0", "densenet121"]

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = BirdDataset(
        csv_file=TRAIN_CSV,
        mode="train",
        load_cached_data=True,
        cache_dir=os.path.join(WORKING_DIR, "cache"),
        height=224,
        width=448,
    )

    val_dataset = BirdDataset(
        csv_file=VAL_CSV,
        mode="val",
        load_cached_data=True,
        cache_dir=os.path.join(WORKING_DIR, "cache"),
        height=224,
        width=448,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn,
    )

    # Store trained models for ensemble
    trained_models = []

    # 3. Training Loop (Ensemble)
    for arch in ARCHITECTURES:
        print(f"\nTraining Architecture: {arch}")

        model = get_model(arch, num_classes=19, pretrained=True)
        model = model.to(device)

        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        best_val_loss = float("inf")
        best_model_path = os.path.join(WORKING_DIR, f"best_model_{arch}.pth")

        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, optimizer, None, train_loader, device, criterion
            )

            # Simple validation check (without TTA for speed during training)
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

            # Optional: Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f}"
                )

        # Load best weights
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        trained_models.append(model)
        print(f"Finished training {arch}. Best Val Loss: {best_val_loss:.4f}")

    # 4. Final Validation & Ensemble Aggregation
    print("\nPerforming Ensemble Validation with TTA...")

    # We need ground truth for validation
    # val_dataset returns (image, label)
    # We can extract labels directly from the dataset
    y_true = val_dataset.labels

    ensemble_preds = np.zeros_like(y_true)

    for model in trained_models:
        # inference_fn uses TTA
        preds = inference_fn(model, val_loader, device)
        ensemble_preds += preds

    # Average predictions
    ensemble_preds /= len(trained_models)

    final_metric = compute_metric(y_true, ensemble_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Mean Absolute Error per sample
    # Shape: (N_samples, N_classes) -> Mean over classes -> (N_samples,)
    sample_errors = np.mean(np.abs(y_true - ensemble_preds), axis=1)

    # Load tabular features
    feat_path = os.path.join(
        INPUT_DIR, "supplemental_data", "histogram_of_segments.txt"
    )
    df_feats = load_tabular_features(feat_path)

    if df_feats is not None:
        # Merge errors with features based on rec_id
        # val_dataset.df contains rec_id
        df_val = val_dataset.df.copy()
        df_val["error"] = sample_errors

        # Merge
        df_analysis = df_val[["rec_id", "error"]].merge(
            df_feats, on="rec_id", how="inner"
        )

        if len(df_analysis) > 0:
            correlations = []
            feat_cols = [c for c in df_analysis.columns if c.startswith("feat_")]

            for col in feat_cols:
                if df_analysis[col].std() > 0:  # Avoid constant columns
                    corr, _ = pearsonr(df_analysis["error"], df_analysis[col])
                    correlations.append((col, corr))

            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            print("Top 5 Feature Correlations with Error:")
            for name, corr in correlations[:5]:
                print(f"{name}: {corr:.4f}")
        else:
            print("No matching records for failure analysis.")
    else:
        print("Could not load tabular features for failure analysis.")

    # 6. Submission
    THRESHOLD = 0.9129501920716607

    if final_metric > THRESHOLD:
        print("\nMetric exceeds threshold. Generating submission...")

        test_dataset = BirdDataset(
            csv_file=TEST_CSV,
            mode="test",
            load_cached_data=True,
            cache_dir=os.path.join(WORKING_DIR, "cache"),
            height=224,
            width=448,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            worker_init_fn=worker_init_fn,
        )

        # Ensemble Inference
        test_preds_sum = None

        for model in trained_models:
            preds = inference_fn(model, test_loader, device)
            if test_preds_sum is None:
                test_preds_sum = preds
            else:
                test_preds_sum += preds

        avg_test_preds = test_preds_sum / len(trained_models)

        # Format Submission
        # Format: Id, Probability
        # Id = rec_id * 100 + species_id

        submission_rows = []
        df_test = test_dataset.df

        for idx, row in df_test.iterrows():
            rec_id = int(row["rec_id"])
            probs = avg_test_preds[idx]

            for species_id, prob in enumerate(probs):
                sub_id = rec_id * 100 + species_id
                submission_rows.append([sub_id, prob])

        df_sub = pd.DataFrame(submission_rows, columns=["Id", "Probability"])

        # Sort by Id just in case
        df_sub = df_sub.sort_values("Id")

        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric {final_metric} did not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
