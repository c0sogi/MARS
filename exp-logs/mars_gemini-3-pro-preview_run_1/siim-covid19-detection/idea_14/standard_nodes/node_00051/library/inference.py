import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import cfg
from library.dataset import SIIMDataset, process_and_cache_data
from library.model import ResNet18D_UNet
from library.train import extract_boxes_from_prob


def run_inference():
    """
    Executes the inference pipeline:
    1. Loads model and test data.
    2. Predicts study labels and segmentation masks (with TTA).
    3. Post-processes predictions (Gating, Box Extraction, Rescaling).
    4. Generates and saves the submission CSV.
    """
    print("Starting Inference Pipeline...")

    # 1. Setup
    device = cfg.device
    os.makedirs(os.path.dirname(cfg.submission_path), exist_ok=True)

    # 2. Load Data
    # process_and_cache_data handles caching logic internally
    print("Loading test data...")
    df_test, images, _, dims = process_and_cache_data("test", load_cached_data=True)

    dataset = SIIMDataset(df_test, images, None, dims, split="test")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # 3. Load Model
    print(f"Loading model from {cfg.model_save_path}...")
    model = ResNet18D_UNet().to(device)

    if os.path.exists(cfg.model_save_path):
        state_dict = torch.load(cfg.model_save_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Model checkpoint not found at {cfg.model_save_path}")
        return

    model.eval()

    # 4. Inference Loop
    # We need to aggregate results because study decisions affect image predictions (Gating)
    study_probs_map = {}  # study_id -> list of probability vectors
    image_raw_preds = {}  # image_id -> list of {box, score} (scaled)
    image_to_study = {}  # image_id -> study_id map

    print(f"Running inference on {len(dataset)} images...")

    with torch.no_grad():
        for i, (images_tensor, meta) in enumerate(loader):
            images_tensor = images_tensor.to(device)

            # --- Test-Time Augmentation (Horizontal Flip) ---
            # Forward pass 1: Original
            cls_logits_1, seg_logits_1 = model(images_tensor)

            # Forward pass 2: Flipped
            images_flip = torch.flip(images_tensor, [3])
            cls_logits_2, seg_logits_2 = model(images_flip)

            # Average Predictions
            # Flip segmentation back before averaging
            seg_logits_2 = torch.flip(seg_logits_2, [3])

            cls_avg = (cls_logits_1 + cls_logits_2) / 2.0
            seg_avg = (seg_logits_1 + seg_logits_2) / 2.0

            # Apply Activations
            p_study = torch.softmax(cls_avg, dim=1).cpu().numpy()[0]  # (4,)
            p_seg = torch.sigmoid(seg_avg).cpu().numpy()[0, 0]  # (H, W)

            # --- Metadata Extraction ---
            # DataLoader returns batched lists for strings/tensors
            study_id = meta["study_id"][0]
            image_id = meta["image_id"][0]
            orig_h = float(meta["orig_size"][0][0])
            orig_w = float(meta["orig_size"][0][1])

            image_to_study[image_id] = study_id

            # Store Study Probabilities
            if study_id not in study_probs_map:
                study_probs_map[study_id] = []
            study_probs_map[study_id].append(p_study)

            # --- Box Extraction & Rescaling ---
            # Extract boxes in 512x512 space
            boxes, scores = extract_boxes_from_prob(p_seg, threshold=0.5)

            # Calculate Scale Factors
            scale_x = orig_w / cfg.image_size
            scale_y = orig_h / cfg.image_size

            scaled_preds = []
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                scaled_box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                scaled_preds.append({"box": scaled_box, "score": score})

            image_raw_preds[image_id] = scaled_preds

    # 5. Post-Processing & Formatting
    print("Post-processing and generating submission strings...")

    submission_rows = []
    final_study_labels = {}  # study_id -> is_negative (bool)

    # Class mapping based on config order
    # 0: Negative, 1: Typical, 2: Indeterminate, 3: Atypical
    class_names = ["negative", "typical", "indeterminate", "atypical"]

    # A. Process Study Level First
    for study_id, probs_list in study_probs_map.items():
        # Average probabilities across images in the study
        avg_probs = np.mean(probs_list, axis=0)

        # Output all classes as dummy boxes (Cite solution_lesson_node_00040)
        study_parts = []
        for idx, name in enumerate(class_names):
            conf = avg_probs[idx]
            study_parts.append(f"{name} {conf:.6f} 0 0 1 1")

        pred_string = " ".join(study_parts)

        submission_rows.append(
            {"Id": f"{study_id}_study", "PredictionString": pred_string}
        )

        # Store negative status for gating (argmax determines the hard label for gating)
        best_idx = np.argmax(avg_probs)
        final_study_labels[study_id] = best_idx == 0

    # B. Process Image Level (with Gating)
    for image_id, preds in image_raw_preds.items():
        study_id = image_to_study[image_id]
        is_negative = final_study_labels.get(study_id, False)

        pred_string = ""

        # Gating Logic
        if is_negative:
            # Force "none" if study is negative
            pred_string = "none 1 0 0 1 1"
        else:
            if len(preds) == 0:
                # No boxes found despite non-negative study
                pred_string = "none 1 0 0 1 1"
            else:
                # Format: "opacity conf xmin ymin xmax ymax ..."
                parts = []
                for p in preds:
                    score = p["score"]
                    b = p["box"]
                    # Ensure coordinates are within bounds (optional but safe)
                    # Note: We don't clip strictly to orig size here as simple float formatting is usually fine
                    part = f"opacity {score:.6f} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}"
                    parts.append(part)
                pred_string = " ".join(parts)

        submission_rows.append(
            {"Id": f"{image_id}_image", "PredictionString": pred_string}
        )

    # 6. Save Submission
    df_sub = pd.DataFrame(submission_rows)
    # Ensure columns are in correct order
    df_sub = df_sub[["Id", "PredictionString"]]

    df_sub.to_csv(cfg.submission_path, index=False)
    print(f"Submission saved to {cfg.submission_path} with {len(df_sub)} rows.")
