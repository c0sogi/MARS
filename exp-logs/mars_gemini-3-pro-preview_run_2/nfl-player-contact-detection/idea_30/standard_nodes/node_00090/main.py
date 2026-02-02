import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.utils import seed_everything
from library.data_loader import NFLDataLoader
from library.dataset import ContactDataset
from library.model import TDSRVNet
from library.trainer import Trainer


def run():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)

    # Adjust Config for Fast Baseline to ensure completion within time limits
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 2048

    print("Initializing Data Loader...")
    data_loader = NFLDataLoader()

    # 2. Load Data
    # Load Train (Sampled for speed)
    print("Loading training data...")
    train_data = data_loader.load_split("train", load_cached_data=True)

    # Downsample training data to ensure fast baseline execution
    N_SAMPLES = 200000
    if len(train_data["y"]) > N_SAMPLES:
        print(
            f"Downsampling training data from {len(train_data['y'])} to {N_SAMPLES}..."
        )
        indices = np.random.choice(len(train_data["y"]), size=N_SAMPLES, replace=False)
        X_num_train = train_data["X_num"][indices]
        X_cat_train = train_data["X_cat"][indices]
        y_train = train_data["y"][indices]
    else:
        X_num_train = train_data["X_num"]
        X_cat_train = train_data["X_cat"]
        y_train = train_data["y"]

    # Load Val (Full set required for accurate metric)
    print("Loading validation data...")
    val_data = data_loader.load_split("val", load_cached_data=True)

    # Load Test (Full set required for submission)
    print("Loading test data...")
    test_data = data_loader.load_split("test", load_cached_data=True)

    # 3. Create Datasets and Loaders
    train_dataset = ContactDataset(X_num_train, X_cat_train, y_train)
    val_dataset = ContactDataset(val_data["X_num"], val_data["X_cat"], val_data["y"])
    test_dataset = ContactDataset(test_data["X_num"], test_data["X_cat"], None)

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

    # 4. Initialize Model
    # Determine embedding sizes from the data (max index + 1)
    # X_cat columns: [pos1, team1, pos2, team2]
    max_pos_idx = max(train_data["X_cat"][:, 0].max(), train_data["X_cat"][:, 2].max())
    max_team_idx = max(train_data["X_cat"][:, 1].max(), train_data["X_cat"][:, 3].max())

    # Ensure we cover potential indices in validation that might be higher if not seen in sample
    # (Though encoders map to fixed range, taking max of train/val is safest)
    max_pos_idx = max(
        max_pos_idx, val_data["X_cat"][:, 0].max(), val_data["X_cat"][:, 2].max()
    )
    max_team_idx = max(
        max_team_idx, val_data["X_cat"][:, 1].max(), val_data["X_cat"][:, 3].max()
    )

    num_positions = int(max_pos_idx) + 1
    num_teams = int(max_team_idx) + 1

    print(f"Model config: num_positions={num_positions}, num_teams={num_teams}")

    model = TDSRVNet(num_positions=num_positions, num_teams=num_teams)

    # 5. Train
    trainer = Trainer(model)
    trainer.train(train_loader, val_loader)

    # 6. Evaluation & Metrics
    print("Evaluating on full validation set...")
    # Load the best model saved during training
    trainer.model.load_state_dict(
        torch.load(trainer.model_path, map_location=trainer.device)
    )

    val_loss, val_mcc, val_thresh = trainer.evaluate(val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_mcc}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    trainer.model.eval()
    all_probs = []
    all_targets = []

    # Inference without gradients
    with torch.no_grad():
        for batch in val_loader:
            (x_kin, x_vis, x_cat), targets = batch
            x_kin = x_kin.to(trainer.device)
            x_vis = x_vis.to(trainer.device)
            x_cat = x_cat.to(trainer.device)

            logits = trainer.model(x_kin, x_vis, x_cat)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # Calculate error magnitude
    errors = np.abs(all_probs - all_targets)

    # Correlate with numerical features
    feature_names = data_loader.get_feature_columns()
    X_val = val_data["X_num"]

    correlations = []
    print("Calculating feature correlations with error...")

    # Calculate correlation for each feature
    # Using a loop over columns (vectorized over samples)
    for i, name in enumerate(feature_names):
        feat_col = X_val[:, i]
        if np.std(feat_col) == 0:
            continue
        corr = np.corrcoef(feat_col, errors)[0, 1]
        correlations.append((name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features associated with Error:")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.4f}")

    # 8. Submission
    THRESHOLD_SCORE = 0.6634847318478787

    if val_mcc > THRESHOLD_SCORE:
        print(
            f"\nValidation MCC ({val_mcc}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        # Predict on Test
        test_probs = trainer.predict(test_loader, load_best_model=True)

        # Apply optimized threshold
        test_preds = (test_probs >= val_thresh).astype(int)

        # Create Submission DataFrame
        sub_df = pd.DataFrame({"contact_id": test_data["ids"], "contact": test_preds})

        # Save
        sub_path = Config.SUBMISSION_PATH
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nValidation MCC ({val_mcc}) <= Threshold ({THRESHOLD_SCORE}). Skipping submission generation."
        )


if __name__ == "__main__":
    run()
