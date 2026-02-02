import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, WhaleLabelEncoder
from library.dataset import WhaleDataset, get_transforms
from library.model import WhaleDenseNet
from library.train import get_logits_inference


def run_inference(debug=Config.DEBUG, load_cached_data=True):
    """
    Executes the inference pipeline using the ensemble of trained DenseNet models.

    Args:
        debug (bool): If True, runs on a subset of the test data for quick validation.
        load_cached_data (bool): Whether to attempt loading the cached LabelEncoder.
    """
    # Ensure reproducibility
    seed_everything(Config.ENSEMBLE_SEEDS[0])
    device = torch.device(Config.DEVICE)

    print("Initializing Inference Pipeline...")

    # -------------------------------------------------------------------------
    # 1. Load Label Encoder
    # -------------------------------------------------------------------------
    # We must use the exact same class mapping as the training phase.
    # The encoder is expected to have been cached to 'classes.parquet' during training.
    label_encoder = WhaleLabelEncoder()
    label_encoder.fit([], load_cached_data=True)

    num_classes = label_encoder.num_classes()
    if num_classes == 0:
        raise RuntimeError(
            "LabelEncoder has 0 classes. Ensure that the training phase has completed "
            "and generated the 'classes.parquet' cache file."
        )

    print(f"Loaded {num_classes} classes from cache.")

    # -------------------------------------------------------------------------
    # 2. Setup Test Data
    # -------------------------------------------------------------------------
    # Use Stage 2 resolution (320x320) as the models were fine-tuned on this.
    test_dataset = WhaleDataset(
        Config.TEST_CSV,
        label_encoder=None,  # Test set has no labels
        transform=get_transforms("val", Config.STAGE_2_IMG_SIZE),
        debug=debug,
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test Dataset Size: {len(test_dataset)} images")

    # -------------------------------------------------------------------------
    # 3. Ensemble Inference
    # -------------------------------------------------------------------------
    # We will accumulate logits from all ensemble members here.
    # Shape: (Num_Samples, Num_Classes)
    aggregated_logits = None

    # Keep track of image names for the submission file
    image_names = test_dataset.df["Image"].values

    for seed in Config.ENSEMBLE_SEEDS:
        print(f"Processing with Ensemble Member (Seed {seed})...")

        # Initialize Model
        model = WhaleDenseNet(
            num_classes=num_classes,
            embedding_size=Config.EMBEDDING_SIZE,
            pretrained=False,  # Loading custom weights, no need for ImageNet download
            dropout_rate=Config.DROPOUT_RATE,
            s=Config.ARCFACE_SCALE,
            m=Config.ARCFACE_MARGIN,
        )
        model.to(device)
        model.eval()

        # Load Checkpoint
        ckpt_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found at {ckpt_path}. Skipping this model.")
            continue

        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading checkpoint for seed {seed}: {e}")
            continue

        # Inference Loop
        model_logits = []

        with torch.no_grad():
            for images, _, _ in test_loader:
                images = images.to(device)

                # get_logits_inference computes Cosine Similarity * Scale
                # It also handles Horizontal Flip TTA if Config.TTA_FLIP is True
                logits = get_logits_inference(
                    model, images, device, tta=Config.TTA_FLIP
                )
                model_logits.append(logits.cpu())

        # Concatenate all batches for this model
        if len(model_logits) > 0:
            full_logits = torch.cat(model_logits, dim=0)

            if aggregated_logits is None:
                aggregated_logits = full_logits
            else:
                aggregated_logits += full_logits

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    if aggregated_logits is None:
        raise RuntimeError(
            "No models were successfully executed. Cannot generate predictions."
        )

    # -------------------------------------------------------------------------
    # 4. Generate Predictions
    # -------------------------------------------------------------------------
    print("Aggregating results and generating submission...")

    # Find Top-K indices (K=5)
    # Note: We don't strictly need to divide aggregated_logits by N_models
    # because the ranking order is invariant to positive scaling.
    topk_vals, topk_indices = torch.topk(aggregated_logits, Config.TOP_K, dim=1)

    topk_indices = topk_indices.numpy()

    submission_rows = []

    for i in range(len(image_names)):
        img_name = image_names[i]
        indices = topk_indices[i]

        # Decode integer indices back to string IDs (e.g., "w_12345")
        pred_labels = label_encoder.inverse_transform(indices)

        # Join into space-separated string
        pred_string = " ".join(pred_labels)

        submission_rows.append({"Image": img_name, "Id": pred_string})

    # -------------------------------------------------------------------------
    # 5. Save Submission
    # -------------------------------------------------------------------------
    df_sub = pd.DataFrame(submission_rows)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission successfully saved to {Config.SUBMISSION_FILE}")
    print(f"Total predictions generated: {len(df_sub)}")
