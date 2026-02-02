import os
import torch
import pandas as pd
import numpy as np
import pydicom
from torch.utils.data import DataLoader
from library.config import Config, seed_everything
from library.dataset import SIIMDataset, get_transforms
from library.model import MultiTaskUNet
from library.utils import mask2boxes, format_submission, STUDY_CLASSES


def get_original_dims(df, load_cached_data=True):
    """
    Retrieves original image dimensions (width, height) for scaling bounding boxes.
    Caches the result as a parquet file to avoid re-reading DICOM headers.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "test_dims.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached dimensions from {cache_path}")
        dims_df = pd.read_parquet(cache_path)
        # Convert to dictionary: image_id -> (width, height)
        return dict(zip(dims_df["image_id"], zip(dims_df["width"], dims_df["height"])))

    # 2. Compute from scratch
    print("Processing dimensions from DICOM headers...")
    records = []
    for _, row in df.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # Read header only (fast)
            d = pydicom.dcmread(path, stop_before_pixels=True)
            records.append(
                {
                    "image_id": row["image_id"],
                    "width": int(d.Columns),
                    "height": int(d.Rows),
                }
            )
        except Exception as e:
            # Fallback to config size if read fails
            records.append(
                {
                    "image_id": row["image_id"],
                    "width": Config.IMG_SIZE,
                    "height": Config.IMG_SIZE,
                }
            )

    dims_df = pd.DataFrame(records)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    dims_df.to_parquet(cache_path)

    return dict(zip(dims_df["image_id"], zip(dims_df["width"], dims_df["height"])))


def predict(
    debug=Config.DEBUG,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Runs inference on the test set and generates the submission file.
    """
    # Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Starting inference on device: {device}")

    # Load Metadata
    test_df = pd.read_csv(Config.TEST_CSV)
    if debug:
        print(f"Debug mode: subsetting test data to {Config.MAX_VAL_SAMPLES} samples.")
        test_df = test_df.head(Config.MAX_VAL_SAMPLES)

    # Get Original Dimensions for Scaling
    dims_map = get_original_dims(test_df, load_cached_data=load_cached_data)

    # Dataset & Loader
    dataset = SIIMDataset(
        df=test_df,
        split="test",
        transform=get_transforms("test"),
        load_cached_data=load_cached_data,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
    )

    # Model Setup
    # We use pretrained=False to avoid attempting to download weights,
    # as we will load our own checkpoint.
    model = MultiTaskUNet(pretrained=False)

    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        state_dict = torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model checkpoint from {Config.MODEL_CHECKPOINT_PATH}")
    else:
        print(
            f"WARNING: Checkpoint {Config.MODEL_CHECKPOINT_PATH} not found. Using random weights."
        )

    model.to(device)
    model.eval()

    # Storage for predictions
    study_preds_dict = {}  # study_id -> list of probability arrays
    image_preds_dict = {}  # image_id -> list of boxes

    print("Running inference loop...")
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            study_ids = batch["study_id"]
            image_ids = batch["image_id"]

            # Forward Pass
            cls_logits, mask_logits = model(images)

            # Probabilities
            cls_probs = torch.softmax(cls_logits, dim=1).cpu().numpy()
            mask_probs = torch.sigmoid(mask_logits).cpu().numpy()

            # Process Batch
            for i in range(len(images)):
                sid = study_ids[i]
                iid = image_ids[i]

                # --- Study Level ---
                if sid not in study_preds_dict:
                    study_preds_dict[sid] = []
                study_preds_dict[sid].append(cls_probs[i])

                # --- Image Level ---
                # Determine predicted class (for logic)
                pred_cls_idx = np.argmax(cls_probs[i])
                pred_label = STUDY_CLASSES[pred_cls_idx]

                # Get scaling factors
                orig_w, orig_h = dims_map.get(iid, (Config.IMG_SIZE, Config.IMG_SIZE))
                scale_x = orig_w / Config.IMG_SIZE
                scale_y = orig_h / Config.IMG_SIZE

                boxes = []
                # Logic: Only predict boxes if the study is NOT negative
                if pred_label != "negative":
                    # Extract boxes from 512x512 mask (channel 0)
                    # mask2boxes returns [conf, x1, y1, x2, y2]
                    raw_boxes = mask2boxes(mask_probs[i, 0], threshold=0.5)

                    for b in raw_boxes:
                        conf, x1, y1, x2, y2 = b
                        # Scale to original dimensions
                        boxes.append(
                            [
                                conf,
                                x1 * scale_x,
                                y1 * scale_y,
                                x2 * scale_x,
                                y2 * scale_y,
                            ]
                        )

                image_preds_dict[iid] = boxes

    print("Formatting submission...")

    # --- 1. Generate Image Rows ---
    # We use format_submission but filter for valid image rows
    img_ids_list = list(image_preds_dict.keys())
    img_boxes_list = list(image_preds_dict.values())
    # Dummy study preds (not used for image rows)
    dummy_study_preds = [np.array([1, 0, 0, 0]) for _ in range(len(img_ids_list))]

    df_imgs = format_submission(
        test_ids=img_ids_list,
        study_preds=dummy_study_preds,
        image_preds=img_boxes_list,
        save_path=None,
    )
    # Keep only rows ending with _image
    df_imgs = df_imgs[df_imgs["id"].str.endswith("_image")]

    # --- 2. Generate Study Rows ---
    study_ids_list = list(study_preds_dict.keys())
    study_probs_list = []

    for sid in study_ids_list:
        # Average probabilities if multiple images per study
        probs_arr = np.array(study_preds_dict[sid])
        avg_probs = np.mean(probs_arr, axis=0)
        study_probs_list.append(avg_probs)

    # Dummy image preds (not used for study rows)
    dummy_image_preds = [[] for _ in range(len(study_ids_list))]

    df_studies = format_submission(
        test_ids=study_ids_list,
        study_preds=study_probs_list,
        image_preds=dummy_image_preds,
        save_path=None,
    )
    # Keep only rows ending with _study
    df_studies = df_studies[df_studies["id"].str.endswith("_study")]

    # --- 3. Merge and Save ---
    final_df = pd.concat([df_studies, df_imgs], ignore_index=True).sort_values("id")

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    final_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission generated successfully with {len(final_df)} rows.")
    print(f"Saved to: {Config.SUBMISSION_PATH}")
