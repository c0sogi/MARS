import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
from scipy import stats
from sklearn.metrics import log_loss

# Import library modules
from library.config import CFG
from library.utils import seed_everything, get_logger
from library.data import prepare_data, CatDogDataset, get_transforms, get_test_loader
from library.modeling import get_model
from library.engine import train_fold, inference


def main():
    # 1. Configuration and Setup
    seed_everything(CFG.seed)

    # Override CFG for fast baseline execution within 4 hours
    # 15 models (3 archs * 5 folds) * 3 epochs fits comfortably on A100
    CFG.epochs = 3
    CFG.debug = False

    # Create output directories
    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(CFG.submission_dir, exist_ok=True)

    # Logger
    logger = get_logger(os.path.join(CFG.output_dir, "run.log"))
    logger.info("Starting runfile.py execution...")
    logger.info(f"Device: {CFG.device}")

    # 2. Data Preparation
    # Load train data with folds
    df_train_folds = prepare_data(load_cached_data=True)

    # Load hold-out validation set
    df_val = pd.read_csv(os.path.join(CFG.metadata_dir, "val.csv"))

    # Create Validation Dataset and Loader
    val_dataset = CatDogDataset(df_val, transforms=get_transforms("valid"), mode="val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # 3. Training Loop (Heterogeneous Ensemble)
    # Store paths to best models
    trained_models = []

    for model_name in CFG.model_names:
        for fold in range(CFG.n_fold):
            logger.info(f"Training {model_name} - Fold {fold}")

            # Train the model
            model, best_score = train_fold(
                df_train_folds, fold, model_name, device=CFG.device
            )

            # Record model path for later reloading (to save memory)
            model_path = os.path.join(
                CFG.output_dir, model_name, f"model_fold_{fold}.pth"
            )
            trained_models.append(
                {"name": model_name, "fold": fold, "path": model_path}
            )

            # Clear memory
            del model
            torch.cuda.empty_cache()

    # 4. Validation & Failure Analysis
    logger.info("Starting Validation on Hold-out Set...")

    # Accumulate predictions from all models
    val_preds_accum = np.zeros(len(df_val))

    for tm in trained_models:
        logger.info(f"Predicting with {tm['name']} Fold {tm['fold']}...")
        model = get_model(tm["name"], pretrained=False)
        model.load_state_dict(torch.load(tm["path"], map_location=CFG.device))
        model.to(CFG.device)

        preds = inference(model, val_loader, device=CFG.device)
        val_preds_accum += preds.flatten()

        del model
        torch.cuda.empty_cache()

    # Average predictions
    val_preds_avg = val_preds_accum / len(trained_models)

    # Calculate Metric
    val_labels = df_val["label"].values
    final_metric = log_loss(val_labels, val_preds_avg)

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate error magnitude
    errors = np.abs(val_labels - val_preds_avg)

    # Extract metadata features for correlation
    widths = []
    heights = []
    file_sizes = []

    for filepath in df_val["filepath"]:
        full_path = os.path.join("./input", filepath)
        try:
            size = os.path.getsize(full_path)
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
                file_sizes.append(size)
            else:
                widths.append(0)
                heights.append(0)
                file_sizes.append(0)
        except:
            widths.append(0)
            heights.append(0)
            file_sizes.append(0)

    # Calculate correlations
    features = {"width": widths, "height": heights, "file_size": file_sizes}

    print("Correlation between Error Magnitude and Input Features:")
    for name, values in features.items():
        if len(values) == len(errors):
            # Handle potential constant input which causes pearsonr warning/error
            if np.std(values) > 0 and np.std(errors) > 0:
                corr, _ = stats.pearsonr(errors, values)
                print(f"{name}: {corr:.4f}")
            else:
                print(f"{name}: 0.0000 (Constant values)")

    # 5. Submission
    THRESHOLD = 0.00879242848677879

    if final_metric < THRESHOLD:
        logger.info(
            f"Validation metric {final_metric} < {THRESHOLD}. Generating submission..."
        )

        test_loader = get_test_loader()
        test_preds_accum = np.zeros(len(test_loader.dataset))

        for tm in trained_models:
            logger.info(f"Inference with {tm['name']} Fold {tm['fold']} on Test Set...")
            model = get_model(tm["name"], pretrained=False)
            model.load_state_dict(torch.load(tm["path"], map_location=CFG.device))
            model.to(CFG.device)

            preds = inference(model, test_loader, device=CFG.device)
            test_preds_accum += preds.flatten()

            del model
            torch.cuda.empty_cache()

        test_preds_avg = test_preds_accum / len(trained_models)

        # Load test metadata to ensure correct ID mapping
        df_test = pd.read_csv(CFG.test_csv)

        submission = pd.DataFrame({"id": df_test["id"], "label": test_preds_avg})

        sub_path = os.path.join(CFG.submission_dir, "submission.csv")
        submission.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}")

    else:
        logger.info(
            f"Validation metric {final_metric} >= {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
