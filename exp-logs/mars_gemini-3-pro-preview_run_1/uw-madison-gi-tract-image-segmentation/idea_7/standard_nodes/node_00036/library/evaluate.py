import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import (
    INPUT_DIR,
    VAL_CSV,
    TEST_CSV,
    CHECKPOINT_DIR,
    DEVICE,
    CLASSES,
    IMG_SIZE,
    SUBMISSION_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_CLASSES,
    SEED,
    WORKING_DIR,
)
from library.config import set_seed
from library.utils import (
    rle_decode,
    rle_encode,
    keep_largest_component,
    compute_dice,
    compute_hausdorff_3d,
    recover_original,
)
from library.model import RecurrentUNet
from library.dataset import SliceSequenceDataset


def load_model(checkpoint_path):
    """Loads the trained RecurrentUNet model."""
    model = RecurrentUNet(num_classes=NUM_CLASSES, pretrained=False)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_dataset(model, df, debug=False):
    """
    Runs inference on the dataset and returns a dictionary of predictions.
    Returns:
        dict: {slice_id: np.array(C, H, W) probabilities}
    """
    if debug:
        df = df.head(100)  # Process a small subset for debugging

    # Use 'test' mode to get (image, id) pairs
    dataset = SliceSequenceDataset(
        df, mode="test", cache_dir=os.path.join(WORKING_DIR, "cache")
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    preds_map = {}

    with torch.no_grad():
        for images, ids in tqdm(loader, desc="Inference", disable=True):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            for i, slice_id in enumerate(ids):
                preds_map[slice_id] = probs[i]  # (C, H, W)

    return preds_map


def process_volume(case_day_df, preds_map, is_validation=True):
    """
    Reconstructs 3D volume, applies post-processing, and computes metrics or RLEs.
    """
    # Sort slices spatially
    case_day_df = case_day_df.sort_values("slice")

    # Metadata for resizing
    original_h = case_day_df.iloc[0]["height"]
    original_w = case_day_df.iloc[0]["width"]

    # 1. Construct 3D Volumes (D, H, W, C)
    # Note: preds_map values are (C, H, W), need to transpose to (H, W, C)
    vol_preds = []
    vol_gt = []
    ids = []

    for _, row in case_day_df.iterrows():
        sid = row["id"]
        if sid in preds_map:
            # Get prediction (C, H, W) -> Transpose to (H, W, C)
            p = preds_map[sid].transpose(1, 2, 0)
            vol_preds.append(p)
            ids.append(sid)

            if is_validation:
                # Load GT
                slice_gt = []
                for cls in CLASSES:
                    mask = rle_decode(row[cls], (original_h, original_w))
                    # Resize GT to model input size (256, 256) for consistent 3D processing?
                    # No, better to resize Prediction to Original size for final evaluation
                    # But for 3D CCA, we should do it on the predicted grid (256x256) to save time/memory
                    # then resize back.
                    slice_gt.append(mask)
                vol_gt.append(np.stack(slice_gt, axis=-1))

    if not vol_preds:
        return None

    # Stack: (D, H, W, C)
    vol_preds_np = np.stack(vol_preds, axis=0)

    # 2. Thresholding
    vol_preds_bin = (vol_preds_np > 0.5).astype(np.uint8)

    # 3. Post-Processing: 3D Connected Component Analysis
    # Apply per class
    for c in range(NUM_CLASSES):
        vol_preds_bin[..., c] = keep_largest_component(vol_preds_bin[..., c])

    results = {}

    # 4. Validation Scoring
    if is_validation:
        vol_gt_np = np.stack(vol_gt, axis=0)  # (D, OrigH, OrigW, C)

        # Resize Preds back to Original Size for comparison
        # (D, 256, 256, C) -> (D, OrigH, OrigW, C)
        vol_preds_resized = []
        for i in range(vol_preds_bin.shape[0]):
            # Use recover_original to reverse the padding/resizing
            slice_resized = recover_original(
                vol_preds_bin[i],
                original_shape=(original_h, original_w),
                target_shape=IMG_SIZE,
                interpolation=cv2.INTER_NEAREST,
            )
            vol_preds_resized.append(slice_resized)

        vol_preds_final = np.stack(vol_preds_resized, axis=0)

        # Compute Metrics
        dice_scores = []
        hd_scores = []

        for c in range(NUM_CLASSES):
            p = vol_preds_final[..., c]
            g = vol_gt_np[..., c]

            dice_scores.append(compute_dice(p, g))
            hd_scores.append(compute_hausdorff_3d(p, g))

        results["dice"] = dice_scores
        results["hd"] = hd_scores

    # 5. Submission Generation (RLE)
    else:
        submission_rows = []
        for i in range(vol_preds_bin.shape[0]):
            # Resize slice back to original using recover_original
            slice_resized = recover_original(
                vol_preds_bin[i],
                original_shape=(original_h, original_w),
                target_shape=IMG_SIZE,
                interpolation=cv2.INTER_NEAREST,
            )

            current_id = ids[i]

            for c_idx, cls_name in enumerate(CLASSES):
                # Extract binary mask for class
                if len(slice_resized.shape) == 3:
                    mask = slice_resized[..., c_idx]
                else:
                    # Fallback if dimensions collapsed (unlikely with C=3)
                    mask = slice_resized

                rle = rle_encode(mask)
                submission_rows.append([current_id, cls_name, rle])

        results["submission"] = submission_rows

    return results


def evaluate_model(debug=False):
    """
    Evaluates the model on the validation set.
    """
    set_seed(SEED)
    print("Starting Evaluation...")

    # Load Data
    df_val = pd.read_csv(VAL_CSV, keep_default_na=False)

    # Load Model
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    model = load_model(checkpoint_path)

    # Inference
    preds_map = predict_dataset(model, df_val, debug=debug)

    # Group by Volume
    groups = df_val.groupby(["case", "day"])

    dice_list = []
    hd_list = []

    for (case, day), group_df in groups:
        if debug and len(dice_list) > 5:
            break

        res = process_volume(group_df, preds_map, is_validation=True)
        if res:
            dice_list.append(res["dice"])
            hd_list.append(res["hd"])

    # Aggregate
    dice_arr = np.array(dice_list)  # (N_vols, 3)
    hd_arr = np.array(hd_list)  # (N_vols, 3)

    mean_dice = np.mean(dice_arr)
    mean_hd = np.mean(hd_arr)

    # Weighted Score: 0.4 * Dice + 0.6 * (1 - HD) ?
    # Note: HD is distance (lower is better), but prompt says "normalized... to create a bounded 0-1 score".
    # Usually HD is distance. If normalized to 0-1, 0 is perfect match?
    # "Hausdorff distance... predicted pixel locations are normalized by image size to create a bounded 0-1 score."
    # If it's a distance, 0 is best. If it's a score, 1 is best.
    # Standard HD is distance.
    # Let's assume the metric goal is to MINIMIZE HD and MAXIMIZE Dice.
    # However, Kaggle competitions usually invert HD or use (1 - HD) if normalized.
    # Given the prompt doesn't specify the combination formula explicitly beyond weights,
    # I will print raw metrics.
    # But usually "Score" implies higher is better.
    # Let's assume Score = 0.4 * Dice + 0.6 * (1 - HD).

    score = 0.4 * mean_dice + 0.6 * (1.0 - mean_hd)

    print(f"Validation Results (Debug={debug}):")
    print(f"Mean Dice: {mean_dice:.10f}")
    print(f"Mean Hausdorff: {mean_hd:.10f}")
    print(f"Composite Score: {score:.10f}")

    # Per class breakdown
    for i, cls in enumerate(CLASSES):
        print(
            f"{cls} - Dice: {np.mean(dice_arr[:, i]):.6f}, HD: {np.mean(hd_arr[:, i]):.6f}"
        )


def generate_submission(debug=False):
    """
    Generates submission file for the test set using sample_submission.csv as a template.
    """
    set_seed(SEED)
    print("Generating Submission...")

    # Load Data
    df_test = pd.read_csv(TEST_CSV)

    # Load Model
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    model = load_model(checkpoint_path)

    # Inference
    preds_map = predict_dataset(model, df_test, debug=debug)

    # Group by Volume
    groups = df_test.groupby(["case", "day"])

    all_rows = []

    for (case, day), group_df in groups:
        res = process_volume(group_df, preds_map, is_validation=False)
        if res and "submission" in res:
            all_rows.extend(res["submission"])

    # Create Dictionary of Predictions: {(id, class): rle}
    pred_dict = {f"{r[0]}_{r[1]}": r[2] for r in all_rows}

    # Load Sample Submission to ensure correct order and row count
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    if os.path.exists(sample_sub_path):
        sub_df = pd.read_csv(sample_sub_path)
        # Map predictions to the template
        # Use empty string for missing predictions (background)
        sub_df["key"] = sub_df["id"] + "_" + sub_df["class"]
        sub_df["predicted"] = sub_df["key"].map(pred_dict).fillna("")
        sub_df.drop(columns=["key"], inplace=True)

        sub_df.to_csv(SUBMISSION_PATH, index=False)
    else:
        # Fallback if sample submission is missing
        print("Warning: sample_submission.csv not found. Using generated dataframe.")
        sub_df = pd.DataFrame(all_rows, columns=["id", "class", "predicted"])
        sub_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission saved to {SUBMISSION_PATH} with {len(sub_df)} rows.")


def run_evaluation(debug=False):
    # Ensure output dirs exist
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # 1. Validate
    evaluate_model(debug=debug)

    # 2. Submit
    generate_submission(debug=debug)


# Note: The __name__ == "__main__" block is omitted as per instructions.
