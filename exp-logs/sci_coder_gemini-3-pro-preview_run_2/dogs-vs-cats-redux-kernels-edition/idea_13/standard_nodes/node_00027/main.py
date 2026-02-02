import os
import gc
import cv2
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Import library modules
from library.config import Config
from library.utils import seed_everything, save_checkpoint, calc_log_loss
from library.dataset import DogCatDataset, get_transforms
from library.models import get_model
from library.engine import train_one_epoch, inference_fn
from library.soup import generate_soup_model
from library.stacking import (
    load_aggregated_predictions,
    fit_meta_learner,
    predict_meta_learner,
    create_submission,
)


def run():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for Fast Baseline Execution
    Config.EPOCHS = 5
    Config.SOUP_EPOCHS = [3, 4, 5]  # Epochs to soup (1-based index in loop)

    # Subset size to ensure execution within 2 hours
    # We use a subset of the training data for the baseline run
    TRAIN_SUBSET_SIZE = 2000

    seed_everything(Config.SEED)

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.OOF_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Subset Training Data
    if len(train_df) > TRAIN_SUBSET_SIZE:
        train_df = train_df.sample(
            n=TRAIN_SUBSET_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)

    # Assign global ID for tracking in OOF
    train_df["global_id"] = train_df.index

    # Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
        train_df.loc[val_idx, "fold"] = fold

    # Dictionary to store hold-out validation predictions for final metric
    # We use the index of val_df as the ID
    val_preds_dict = {"id": val_df.index.tolist(), "target": val_df["label"].tolist()}

    # -------------------------------------------------------------------------
    # 3. Training & Inference Loop
    # -------------------------------------------------------------------------
    for model_name in Config.MODEL_ARCHS:
        # Containers for this architecture
        arch_oof_dfs = []
        arch_val_preds = []
        arch_test_preds = []

        for fold in range(Config.NUM_FOLDS):
            # --- Data Split ---
            fold_train_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
            fold_val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

            # --- DataLoaders ---
            train_dataset = DogCatDataset(
                fold_train_df, transforms=get_transforms("train"), mode="train"
            )
            valid_dataset = DogCatDataset(
                fold_val_df, transforms=get_transforms("val"), mode="val"
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # --- Model Setup ---
            model = get_model(model_name, pretrained=True, num_classes=1)
            model.to(Config.DEVICE)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS
            )

            # --- Training ---
            checkpoint_paths = []
            for epoch in range(1, Config.EPOCHS + 1):
                train_loss = train_one_epoch(
                    model, optimizer, train_loader, Config.DEVICE, epoch
                )
                scheduler.step()

                if epoch in Config.SOUP_EPOCHS:
                    ckpt_path = os.path.join(
                        Config.CHECKPOINT_DIR, f"{model_name}_fold_{fold}_ep{epoch}.pth"
                    )
                    save_checkpoint(model.state_dict(), ckpt_path)
                    checkpoint_paths.append(ckpt_path)

            # --- Souping ---
            # Initialize fresh model for souping
            soup_model = get_model(model_name, pretrained=False, num_classes=1)
            # Generate soup (averaging on CPU)
            soup_model = generate_soup_model(soup_model, checkpoint_paths, device="cpu")
            soup_model.to(Config.DEVICE)

            # --- Inference: OOF (Validation Fold) ---
            # inference_fn returns (ids, preds). For OOF, we rely on the order of valid_loader
            _, preds_oof = inference_fn(soup_model, valid_loader, Config.DEVICE)

            fold_oof_df = fold_val_df.copy()
            fold_oof_df[model_name] = preds_oof
            arch_oof_dfs.append(fold_oof_df)

            # --- Inference: Hold-out Validation ---
            holdout_dataset = DogCatDataset(
                val_df, transforms=get_transforms("val"), mode="val"
            )
            holdout_loader = torch.utils.data.DataLoader(
                holdout_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )
            _, preds_val = inference_fn(soup_model, holdout_loader, Config.DEVICE)
            arch_val_preds.append(preds_val)

            # --- Inference: Test Set ---
            test_dataset = DogCatDataset(
                test_df, transforms=get_transforms("test"), mode="test"
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )
            _, preds_test = inference_fn(soup_model, test_loader, Config.DEVICE)
            arch_test_preds.append(preds_test)

            # --- Cleanup ---
            del model, soup_model, optimizer, scheduler
            torch.cuda.empty_cache()
            gc.collect()

        # --- Aggregate Architecture Results ---

        # 1. Save OOF (Concatenated Folds)
        full_oof_df = (
            pd.concat(arch_oof_dfs).sort_values("global_id").reset_index(drop=True)
        )
        # Rename global_id to id for stacking compatibility
        full_oof_df = full_oof_df.rename(columns={"global_id": "id", "label": "target"})
        full_oof_df[["id", "target", model_name]].to_csv(
            os.path.join(Config.OOF_DIR, f"{model_name}_oof.csv"), index=False
        )

        # 2. Store Validation Preds (Averaged Folds)
        val_preds_dict[model_name] = np.mean(arch_val_preds, axis=0)

        # 3. Save Test Preds (Averaged Folds)
        avg_test_preds = np.mean(arch_test_preds, axis=0)
        test_res_df = test_df.copy()
        test_res_df[model_name] = avg_test_preds
        test_res_df[["id", model_name]].to_csv(
            os.path.join(Config.OOF_DIR, f"{model_name}_test.csv"), index=False
        )

    # -------------------------------------------------------------------------
    # 4. Stacking & Meta-Learning
    # -------------------------------------------------------------------------
    # Load aggregated OOF predictions
    # This will read the {model_name}_oof.csv files we just created
    oof_df = load_aggregated_predictions(mode="oof", load_cached_data=False)

    # Train Meta-Learner
    meta_model, feature_cols = fit_meta_learner(oof_df, target_col="target")

    # -------------------------------------------------------------------------
    # 5. Validation Assessment
    # -------------------------------------------------------------------------
    # Prepare Validation Data for Meta-Learner
    val_pred_df = pd.DataFrame(val_preds_dict)

    # Predict
    val_meta_preds = predict_meta_learner(meta_model, val_pred_df, feature_cols)

    # Calculate Final Metric
    final_val_metric = calc_log_loss(val_pred_df["target"].values, val_meta_preds)
    print(f"Final Validation Metric: {final_val_metric}")

    # --- Failure Analysis ---
    print("Performing Failure Analysis...")
    val_pred_df["pred"] = val_meta_preds
    val_pred_df["error"] = np.abs(val_pred_df["target"] - val_pred_df["pred"])

    # Extract meta-features for validation images
    meta_features = []
    for _, row in val_df.iterrows():
        filepath = os.path.join(Config.INPUT_DIR, row["filepath"])
        try:
            img = cv2.imread(filepath)
            if img is not None:
                h, w, _ = img.shape
                meta_features.append({"width": w, "height": h, "aspect_ratio": w / h})
            else:
                meta_features.append({"width": 0, "height": 0, "aspect_ratio": 0})
        except:
            meta_features.append({"width": 0, "height": 0, "aspect_ratio": 0})

    meta_df = pd.DataFrame(meta_features)
    analysis_df = pd.concat([val_pred_df.reset_index(drop=True), meta_df], axis=1)

    # Calculate Correlations
    correlations = analysis_df[["error", "width", "height", "aspect_ratio"]].corr()[
        "error"
    ]
    print("Correlation between Error and Meta-features:")
    print(correlations)

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.01366509944361823

    if final_val_metric < THRESHOLD:
        print("Metric check passed. Generating submission...")
        # Load Test Predictions
        test_pred_df = load_aggregated_predictions(mode="test", load_cached_data=False)

        # Predict with Meta-Learner
        final_test_preds = predict_meta_learner(meta_model, test_pred_df, feature_cols)

        # Save Submission
        create_submission(
            test_pred_df,
            final_test_preds,
            os.path.join(Config.SUBMISSION_DIR, "submission.csv"),
        )
    else:
        print(
            f"Metric {final_val_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
