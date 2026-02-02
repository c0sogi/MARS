import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import CervicalSpineDataset
from library.model import DynamicDepthConvNeXt


def run_inference(
    model_path=None,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
):
    """
    Runs inference on the test set using Multi-Offset Test-Time Augmentation (TTA).
    Generates the submission file.

    Args:
        model_path (str): Path to the trained model weights. If None, uses best_model.pth.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on.
    """
    if model_path is None:
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Initializing inference on device: {device}")

    # --- 1. Load Model ---
    model = DynamicDepthConvNeXt(
        backbone_name=Config.BACKBONE,
        pretrained=False,  # No need to download pretrained weights, we load state_dict
        num_classes=Config.NUM_CLASSES,
    )

    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model path {model_path} not found. Using random initialization (for debugging only)."
        )

    model.to(device)
    model.eval()

    # --- 2. Multi-Offset TTA Loop ---
    # We run inference 3 times with shifted sampling grids to catch sparse fractures
    offsets = [0.0, 0.3, -0.3]
    aggregated_preds = None
    study_uids = None

    for offset in offsets:
        print(f"Running inference with TTA offset: {offset}")

        # Initialize dataset for this offset
        dataset = CervicalSpineDataset(
            mode="test",
            transform=None,  # Uses default test transform
            load_cached_data=True,
            tta_offset=offset,
            seq_length=Config.SEQ_LENGTH,
        )

        # Capture UIDs from the first pass (order is deterministic)
        if study_uids is None:
            study_uids = dataset.df["StudyInstanceUID"].values

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Prediction Loop
        offset_preds = []

        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)

                # Forward pass
                logits = model(inputs)
                probs = torch.sigmoid(logits)

                offset_preds.append(probs.cpu().numpy())

        # Concatenate batches: (N_samples, 8)
        if len(offset_preds) > 0:
            offset_preds = np.concatenate(offset_preds, axis=0)
        else:
            offset_preds = np.zeros((0, Config.NUM_CLASSES))

        # Aggregation: Element-wise Max
        if aggregated_preds is None:
            aggregated_preds = offset_preds
        else:
            aggregated_preds = np.maximum(aggregated_preds, offset_preds)

    # --- 3. Format Submission ---
    print("Formatting submission...")

    # Create a lookup map: UID -> {Target: Prob}
    # Targets: C1, C2, C3, C4, C5, C6, C7, patient_overall
    preds_map = {}

    if aggregated_preds is not None and len(study_uids) == len(aggregated_preds):
        for idx, uid in enumerate(study_uids):
            preds = aggregated_preds[idx]
            preds_map[uid] = {
                col: float(val) for col, val in zip(Config.TARGET_COLS, preds)
            }
    else:
        print("Warning: Mismatch between Study UIDs and Predictions count.")

    # Load sample submission to get the exact row IDs required
    if os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    else:
        # Fallback if sample submission is missing (unlikely in competition)
        print("Sample submission not found. Creating empty dataframe.")
        sub_df = pd.DataFrame(columns=["row_id", "fractured"])

    # Helper to extract probability for a specific row_id
    def get_prediction(row_id):
        # row_id format: [UID]_[Target]
        # Iterate backwards through known targets to find the split point
        for target in Config.TARGET_COLS:
            suffix = f"_{target}"
            if row_id.endswith(suffix):
                uid = row_id[: -len(suffix)]
                if uid in preds_map:
                    return preds_map[uid][target]
        return 0.5  # Default fallback

    # Apply mapping
    sub_df["fractured"] = sub_df["row_id"].apply(get_prediction)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
