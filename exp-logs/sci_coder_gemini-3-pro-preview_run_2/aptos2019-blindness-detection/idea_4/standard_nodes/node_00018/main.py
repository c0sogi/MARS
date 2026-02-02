import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import RetinopathyDataset, get_transforms
from library.models import RetinopathyModel
from library.train import train_models
from library.inference import predict_and_submit
from library.utils import seed_everything, quadratic_weighted_kappa

# Suppress warnings
warnings.filterwarnings("ignore")


def compute_image_stats(df):
    """
    Computes image statistics for failure analysis.
    """
    stats = []
    for idx, row in df.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(path)
        if img is None:
            continue
        h, w, c = img.shape
        mean_intensity = np.mean(img)
        std_intensity = np.std(img)
        stats.append(
            {
                "id_code": row["id_code"],
                "width": w,
                "height": h,
                "aspect_ratio": w / h if h > 0 else 0,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
            }
        )
    return pd.DataFrame(stats)


def evaluate_ensemble(val_csv_path):
    """
    Evaluates the trained ensemble on the validation set.
    """
    device = Config.DEVICE
    df_val = pd.read_csv(val_csv_path)

    num_samples = len(df_val)
    accumulated_scores = torch.zeros(num_samples, dtype=torch.float32)
    models_executed = 0

    # Ground truth
    y_true = df_val["diagnosis"].values

    print("Evaluating ensemble on validation set...")

    for model_name, image_size in Config.MODEL_SPECS.items():
        # Use test mode transforms (Resize + Normalize) without augmentation
        ds = RetinopathyDataset(
            csv_path=val_csv_path,
            transform=get_transforms(image_size, mode="test"),
            mode="val",  # Returns image, label
        )

        loader = DataLoader(
            ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        for fold in range(Config.NUM_FOLDS):
            ckpt_path = os.path.join(Config.OUTPUT_DIR, f"{model_name}_fold_{fold}.pth")
            if not os.path.exists(ckpt_path):
                continue

            model = RetinopathyModel(model_name=model_name, pretrained=False)
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            fold_preds = []

            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)

                    # TTA: Original + Horizontal Flip
                    out1 = model(images)
                    out2 = model(torch.flip(images, dims=[3]))
                    avg = (out1 + out2) / 2.0
                    fold_preds.append(avg.cpu())

            accumulated_scores += torch.cat(fold_preds)
            models_executed += 1

            del model
            torch.cuda.empty_cache()

    if models_executed == 0:
        return 0.0, None, None

    final_scores = accumulated_scores / models_executed

    # Convert to class labels
    y_pred_continuous = final_scores.numpy()
    y_pred_labels = np.round(y_pred_continuous).astype(int)
    y_pred_labels = np.clip(y_pred_labels, 0, 4)

    qwk = quadratic_weighted_kappa(y_true, y_pred_labels)

    return qwk, y_pred_labels, y_pred_continuous


def run():
    seed_everything(Config.SEED)

    # 1. Configure for Fast Baseline
    # Limit epochs to ensure completion within the time limit while allowing convergence
    Config.EPOCHS = 5

    # 2. Train
    print("Starting Training...")
    train_models(debug=False)

    # 3. Validation
    val_csv = Config.VAL_CSV
    qwk, y_pred_labels, y_pred_cont = evaluate_ensemble(val_csv)

    print(f"Final Validation Metric: {qwk}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    df_val = pd.read_csv(val_csv)

    # Calculate error
    df_val["pred"] = y_pred_labels
    df_val["pred_cont"] = y_pred_cont
    df_val["error"] = np.abs(df_val["diagnosis"] - df_val["pred"])
    df_val["error_cont"] = np.abs(df_val["diagnosis"] - df_val["pred_cont"])

    # Compute image stats
    print("Computing image statistics for validation set...")
    df_stats = compute_image_stats(df_val)

    # Merge
    df_analysis = pd.merge(df_val, df_stats, on="id_code")

    # Correlations
    features = ["width", "height", "aspect_ratio", "mean_intensity", "std_intensity"]
    print("\nCorrelation between Error Magnitude and Image Features:")
    for feat in features:
        if feat in df_analysis.columns:
            corr, _ = spearmanr(df_analysis["error_cont"], df_analysis[feat])
            print(f"{feat}: {corr:.4f}")

    # 5. Submission
    THRESHOLD = 0.9207435978935975
    if qwk > THRESHOLD:
        print(
            f"\nValidation metric ({qwk}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(debug=False)
    else:
        print(
            f"\nValidation metric ({qwk}) does not meet threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    run()
