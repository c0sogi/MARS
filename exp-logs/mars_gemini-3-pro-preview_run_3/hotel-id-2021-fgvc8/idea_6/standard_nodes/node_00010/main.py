import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, apk, mapk
from library.dataset import get_dataloaders
from library.model import HotelRecognitionModel
from library.train import run_training
from library.inference import predict_and_submit


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)

    # Adjust Config for optimized training on A100 within time limits
    # Increasing batch size allows faster iteration on the A100 40GB GPU
    Config.BATCH_SIZE = 96
    Config.EPOCHS = 5
    Config.NUM_WORKERS = 12

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    # run_training handles the training loop, validation, and saving the best model
    run_training(debug=False, load_cached_data=True, epochs=Config.EPOCHS)

    # --------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    device = torch.device(Config.DEVICE)

    # Load DataLoaders (we need val_loader and classes)
    # debug=False ensures we use the full validation set
    _, val_loader, _, classes = get_dataloaders(debug=False, load_cached_data=True)

    # Load the best model saved during training
    model = HotelRecognitionModel(
        n_classes=len(classes),
        model_name=Config.BACKBONE,
        pretrained=False,  # Weights will be loaded from checkpoint
        embedding_size=Config.EMBEDDING_SIZE,
    )

    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: Model checkpoint not found. Using initialized weights.")

    model.to(device)
    model.eval()

    # Perform Inference on Validation Set
    all_preds = []
    all_targets = []

    # Pre-compute class centers for fast similarity search
    class_centers = model.get_class_centers().detach()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Get embeddings
            embeddings = model(images, labels=None)
            embeddings = F.normalize(embeddings)

            # Compute Cosine Similarity
            sims = torch.matmul(embeddings, class_centers.T)

            # Get Top 5 Predictions
            _, topk_indices = torch.topk(sims, k=5, dim=1)

            topk_indices = topk_indices.cpu().numpy()
            labels_np = labels.numpy()

            for i in range(len(labels_np)):
                all_preds.append(topk_indices[i].tolist())
                all_targets.append([labels_np[i]])

    # Compute Final Metric
    val_map5 = mapk(all_targets, all_preds, k=5)
    print(f"Final Validation Metric: {val_map5}")

    # --- Failure Analysis ---
    # 1. Calculate AP@5 per sample
    ap_scores = [apk(t, p, k=5) for t, p in zip(all_targets, all_preds)]

    # 2. Get Class Frequencies from Training Data
    train_df = pd.read_csv(Config.TRAIN_CSV)
    class_counts = train_df["hotel_id"].value_counts().to_dict()

    # 3. Map target indices back to hotel_ids to lookup frequency
    # all_targets contains lists like [[idx]], take [0]
    target_indices = [t[0] for t in all_targets]
    target_hotel_ids = [classes[idx] for idx in target_indices]

    target_freqs = [class_counts.get(hid, 0) for hid in target_hotel_ids]

    # 4. Compute Correlation
    # We correlate the AP score with the frequency of the class
    corr, _ = pearsonr(ap_scores, target_freqs)

    print("Failure Analysis:")
    print(f"Correlation between AP@5 and Class Frequency: {corr}")

    # --------------------------------------------------------------------------
    # 4. Submission
    # --------------------------------------------------------------------------
    threshold = 0.5589516758918762

    if val_map5 > threshold:
        print(
            f"Validation metric meets threshold ({threshold}). Generating submission..."
        )
        predict_and_submit(
            model_path=Config.MODEL_PATH,
            output_file=Config.SUBMISSION_FILE,
            device=Config.DEVICE,
            batch_size=Config.BATCH_SIZE,
        )
    else:
        print(
            f"Validation metric {val_map5} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
