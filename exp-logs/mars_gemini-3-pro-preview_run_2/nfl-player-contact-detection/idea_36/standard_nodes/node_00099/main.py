import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Import provided library modules
import library.config as config
import library.utils as utils
import library.feature_engineering as fe
import library.dataset as ds
import library.model as model_lib


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading & Feature Engineering
    # Load metadata
    train_meta = pd.read_csv(config.TRAIN_META_PATH)
    val_meta = pd.read_csv(config.VAL_META_PATH)

    # Load raw data for feature engineering context
    train_tracking = pd.read_csv(config.TRAIN_TRACKING_PATH)
    train_helmets = pd.read_csv(config.TRAIN_HELMETS_PATH)

    # Generate Features (utilizing caching)
    # We load the full feature set first, then subsample for speed if necessary
    df_train = fe.prepare_data(
        train_meta, train_tracking, train_helmets, load_cached_data=True, split="train"
    )
    df_val = fe.prepare_data(
        val_meta,
        train_tracking,
        train_helmets,
        load_cached_data=True,
        split="validation",
    )

    # Free up raw data memory
    del train_tracking, train_helmets, train_meta, val_meta
    gc.collect()

    # Subsample training data for fast baseline execution
    MAX_TRAIN_SAMPLES = 500000
    if len(df_train) > MAX_TRAIN_SAMPLES:
        df_train = df_train.sample(
            n=MAX_TRAIN_SAMPLES, random_state=config.SEED
        ).reset_index(drop=True)

    # 3. Feature Scaling
    # Identify columns to scale
    vis_cols = [f"{c}_1" for c in config.VISUAL_FEATURES] + [
        f"{c}_2" for c in config.VISUAL_FEATURES
    ]

    exclude_cols = set(config.META_COLUMNS) | {
        "contact_id",
        "contact",
        "nfl_player_id_1",
        "nfl_player_id_2",
    }
    exclude_cols.update(vis_cols)

    kin_cols = [
        c
        for c in df_train.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_train[c])
    ]

    scaler_kin = StandardScaler()
    scaler_vis = StandardScaler()

    # Fit on train, transform train and val
    # Note: Using float32 to save memory and match model precision
    df_train[kin_cols] = scaler_kin.fit_transform(df_train[kin_cols].astype(np.float32))
    df_train[vis_cols] = scaler_vis.fit_transform(df_train[vis_cols].astype(np.float32))

    df_val[kin_cols] = scaler_kin.transform(df_val[kin_cols].astype(np.float32))
    df_val[vis_cols] = scaler_vis.transform(df_val[vis_cols].astype(np.float32))

    # 4. Dataset & DataLoader
    train_dataset = ds.ContactDataset(df_train, split="train")
    val_dataset = ds.ContactDataset(df_val, split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Model Initialization
    # Determine input dimensions from a sample
    sample = train_dataset[0]
    kin_dim = sample["kinematic"].shape[0]
    vis_dim = sample["visual"].shape[0]

    model = model_lib.NRPIRVNet(kin_input_dim=kin_dim, vis_input_dim=vis_dim)
    model.to(device)

    # 6. Training Loop
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = utils.FocalLoss(**config.FOCAL_LOSS_PARAMS)

    best_mcc = -1.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        model.train()

        for batch in train_loader:
            kin = batch["kinematic"].to(device)
            vis = batch["visual"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            logits = model(kin, vis).squeeze()
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        # Validation Step
        model.eval()
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                kin = batch["kinematic"].to(device)
                vis = batch["visual"].to(device)
                targets = batch["target"]

                logits = model(kin, vis).squeeze()
                probs = torch.sigmoid(logits).cpu().numpy()

                val_probs.extend(probs)
                val_targets.extend(targets.numpy())

        val_probs = np.array(val_probs)
        val_targets = np.array(val_targets)

        # Fast threshold search for monitoring
        _, val_mcc = utils.optimize_threshold(val_targets, val_probs, steps=20)

        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            break

    # 7. Final Evaluation & Threshold Optimization
    model.load_state_dict(best_model_state)
    model.eval()

    val_probs = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            kin = batch["kinematic"].to(device)
            vis = batch["visual"].to(device)
            targets = batch["target"]

            logits = model(kin, vis).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()

            val_probs.extend(probs)
            val_targets.extend(targets.numpy())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    best_thresh, final_mcc = utils.optimize_threshold(val_targets, val_probs)
    print(f"Final Validation Metric: {final_mcc}")

    # 8. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate continuous error magnitude
    errors = np.abs(val_targets - val_probs)

    # Select key features for correlation (lag 0 distance and speed)
    # We look for columns in df_val that match these concepts
    analysis_feats = [
        c for c in df_val.columns if "lag_0" in c and ("dist" in c or "speed" in c)
    ]

    correlations = {}
    for feat in analysis_feats:
        if feat in df_val.columns:
            corr = np.corrcoef(df_val[feat].values, errors)[0, 1]
            correlations[feat] = corr

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top Feature Correlations with Error:")
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.4f}")

    # 9. Submission
    THRESHOLD_SCORE = 0.6634847318478787
    if final_mcc > THRESHOLD_SCORE:
        test_meta = pd.read_csv(config.TEST_META_PATH)
        test_tracking = pd.read_csv(config.TEST_TRACKING_PATH)
        test_helmets = pd.read_csv(config.TEST_HELMETS_PATH)

        df_test = fe.prepare_data(
            test_meta, test_tracking, test_helmets, load_cached_data=True, split="test"
        )

        # Apply scaling
        df_test[kin_cols] = scaler_kin.transform(df_test[kin_cols].astype(np.float32))
        df_test[vis_cols] = scaler_vis.transform(df_test[vis_cols].astype(np.float32))

        test_dataset = ds.ContactDataset(df_test, split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        test_probs = []
        with torch.no_grad():
            for batch in test_loader:
                kin = batch["kinematic"].to(device)
                vis = batch["visual"].to(device)

                logits = model(kin, vis).squeeze()
                probs = torch.sigmoid(logits).cpu().numpy()
                test_probs.extend(probs)

        test_probs = np.array(test_probs)
        predictions = (test_probs >= best_thresh).astype(int)

        submission = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": predictions}
        )

        submission.to_csv(config.SUBMISSION_FILE, index=False)


if __name__ == "__main__":
    main()
