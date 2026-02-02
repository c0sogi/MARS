import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import LidarDataset
from library.detector import TwoStagePointPillars


def transform_box_to_global(box, trans_matrix):
    """
    Transforms a single box from Lidar frame to Global frame.
    box: [x, y, z, w, l, h, yaw]
    trans_matrix: 4x4 matrix (Global -> Lidar)
    """
    # We need Lidar -> Global, which is the inverse
    # trans_matrix provided by dataset is Global -> Lidar
    # However, let's verify dataset implementation.
    # Dataset: M_global_to_lidar = inv(M_lidar_to_global)
    # So the matrix in the batch is Global -> Lidar.
    # To go back, we need to invert it again.

    # Invert to get Lidar -> Global
    lidar_to_global = np.linalg.inv(trans_matrix)

    # 1. Transform Center
    x, y, z = box[0], box[1], box[2]
    center_lidar = np.array([x, y, z, 1.0])
    center_global = lidar_to_global @ center_lidar

    # 2. Transform Yaw
    yaw = box[6]
    # Create a unit vector in the direction of yaw in Lidar frame
    # z-component is 0 for yaw rotation around Z-axis
    vec_lidar = np.array([np.cos(yaw), np.sin(yaw), 0.0, 0.0])
    vec_global = lidar_to_global @ vec_lidar

    yaw_global = np.arctan2(vec_global[1], vec_global[0])

    # Update box
    # [x, y, z, w, l, h, yaw]
    # Dimensions (w, l, h) are scalar and invariant to rigid transformation
    return np.array(
        [
            center_global[0],
            center_global[1],
            center_global[2],
            box[3],
            box[4],
            box[5],
            yaw_global,
        ]
    )


def generate_submission(
    checkpoint_path=Config.CHECKPOINT_PATH,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
):
    """
    Generates the submission CSV file for the test set.
    """
    # 1. Setup
    Config.set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Generating submission using model: {checkpoint_path}")
    print(f"Output path: {output_path}")

    # 2. Load Data
    # split='test' loads test_metadata.csv and test_data json
    test_dataset = LidarDataset(split="test", load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=LidarDataset.collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Test samples: {len(test_dataset)}")

    # 3. Load Model
    model = TwoStagePointPillars()
    model.to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 4. Inference Loop
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            # Forward pass
            # Returns lists of tensors: boxes, scores, labels
            pred_boxes_list, pred_scores_list, pred_labels_list = model(
                batch, mode="test"
            )

            sample_tokens = batch["sample_tokens"]
            trans_matrices = batch["trans_matrices"].cpu().numpy()  # (B, 4, 4)

            # Process batch
            for i in range(len(sample_tokens)):
                token = sample_tokens[i]
                boxes = pred_boxes_list[i].cpu().numpy()  # (N, 7)
                scores = pred_scores_list[i].cpu().numpy()  # (N,)
                labels = pred_labels_list[i].cpu().numpy()  # (N,)
                trans_mat = trans_matrices[i]

                prediction_strings = []

                if len(boxes) > 0:
                    for j in range(len(boxes)):
                        # Transform to Global Frame
                        box_global = transform_box_to_global(boxes[j], trans_mat)

                        score = float(scores[j])
                        label_idx = int(labels[j])

                        # Map label index to class name
                        if 0 <= label_idx < len(Config.DETECTED_CLASSES):
                            class_name = Config.DETECTED_CLASSES[label_idx]
                        else:
                            # Fallback if index out of bounds (should not happen)
                            class_name = "car"

                        # Format: confidence x y z w l h yaw class_name
                        # Note: Task description example:
                        # confidence center_x center_y center_z width length height yaw class_name
                        pred_str = (
                            f"{score:.4f} "
                            f"{box_global[0]:.4f} {box_global[1]:.4f} {box_global[2]:.4f} "
                            f"{box_global[3]:.4f} {box_global[4]:.4f} {box_global[5]:.4f} "
                            f"{box_global[6]:.4f} {class_name}"
                        )
                        prediction_strings.append(pred_str)

                # Join all predictions for this image with a space
                full_pred_str = " ".join(prediction_strings)

                results.append({"Id": token, "PredictionString": full_pred_str})

    # 5. Create DataFrame and Save
    submission_df = pd.DataFrame(results)

    # Ensure all test IDs are present (even if 0 predictions)
    # The loader iterates over metadata which is derived from sample_submission.csv
    # so the order and count should match, but sorting ensures consistency.

    # Load original sample submission to ensure correct order/Ids if needed
    # But here we just save what we processed.

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission_df.head())
