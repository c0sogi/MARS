import os
import sys
import torch
import numpy as np
import random
import warnings

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import SIIMDataset
from library.utils import collate_fn, box_cxcywh_to_xyxy, box_iou
from library.model import MultiTaskDINO
from library.loss import DINOLoss, HungarianMatcher
from library.engine import train_one_epoch, evaluate, get_loss_weights

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("=== Starting SIIM-FISABIO-RSNA COVID-19 Detection Demo ===")

    # 1. Configuration Setup
    # Override Config for a fast demonstration (Debug Mode)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 6  # Small subset for speed
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.setup()

    set_seed(Config.SEED)

    # 2. Dataset and DataLoader
    print("\n[1] Initializing Dataset and DataLoader...")
    # Initialize Training Dataset
    train_dataset = SIIMDataset(split="train", load_cached_data=False)

    # Verify Dataset Length
    assert (
        len(train_dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} samples in debug mode, got {len(train_dataset)}"

    # Verify Data Item Structure
    img_tensor, target = train_dataset[0]
    print(f"    Image Shape: {img_tensor.shape}")
    print(f"    Target Keys: {list(target.keys())}")

    assert img_tensor.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image tensor shape."
    assert (
        "boxes" in target and "labels" in target and "study_label" in target
    ), "Missing keys in target dictionary."

    # Initialize DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )
    print("    DataLoader initialized successfully.")

    # 3. Model Initialization
    print("\n[2] Initializing MultiTaskDINO Model...")
    model = MultiTaskDINO()
    model.to(Config.DEVICE)
    print(f"    Model moved to {Config.DEVICE}.")

    # 4. Forward Pass Verification
    print("\n[3] Running Forward Pass...")
    # Fetch a single batch
    samples, targets = next(iter(train_loader))
    samples = samples.to(Config.DEVICE)
    targets = [
        {
            k: v.to(Config.DEVICE) if isinstance(v, torch.Tensor) else v
            for k, v in t.items()
        }
        for t in targets
    ]

    # Forward pass in training mode (includes CDN)
    model.train()
    outputs = model(samples, targets)

    # Verify Output Structure
    expected_keys = [
        "pred_logits",
        "pred_boxes",
        "study_logits",
        "dn_logits",
        "dn_boxes",
    ]
    for key in expected_keys:
        assert key in outputs, f"Missing key '{key}' in model outputs."

    # Verify Shapes
    # pred_logits: (B, Num_Queries, Num_Classes)
    # pred_boxes: (B, Num_Queries, 4)
    # study_logits: (B, Num_Study_Classes)
    B = Config.BATCH_SIZE
    Q = Config.NUM_QUERIES
    C = Config.NUM_CLASSES
    S = Config.NUM_STUDY_CLASSES

    assert outputs["pred_logits"].shape == (
        B,
        Q,
        C,
    ), f"Shape mismatch for pred_logits: {outputs['pred_logits'].shape}"
    assert outputs["pred_boxes"].shape == (
        B,
        Q,
        4,
    ), f"Shape mismatch for pred_boxes: {outputs['pred_boxes'].shape}"
    assert outputs["study_logits"].shape == (
        B,
        S,
    ), f"Shape mismatch for study_logits: {outputs['study_logits'].shape}"

    print("    Forward pass successful. Output shapes verified.")

    # 5. Loss Calculation
    print("\n[4] Calculating Loss...")
    matcher = HungarianMatcher()
    weight_dict = get_loss_weights()
    criterion = DINOLoss(matcher=matcher, weight_dict=weight_dict)

    loss_dict = criterion(outputs, targets)

    # Check for main loss components
    assert "loss_ce" in loss_dict, "Classification loss missing."
    assert "loss_bbox" in loss_dict, "Bounding box loss missing."
    assert "loss_study" in loss_dict, "Study level loss missing."

    # Calculate total loss
    total_loss = sum(loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict.keys())
    print(f"    Total Loss calculated: {total_loss.item():.4f}")
    assert not torch.isnan(total_loss), "Total loss is NaN."

    # 6. Training Loop (One Epoch)
    print("\n[5] Executing Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Run training for one epoch
    avg_train_loss = train_one_epoch(
        model,
        criterion,
        train_loader,
        optimizer,
        Config.DEVICE,
        epoch=1,
        max_norm=Config.CLIP_MAX_NORM,
    )
    print(f"    Training Epoch Complete. Avg Loss: {avg_train_loss:.4f}")

    # 7. Evaluation
    print("\n[6] Executing Evaluation...")
    # Initialize Val Dataset (Debug mode)
    val_dataset = SIIMDataset(split="val", load_cached_data=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    # Run evaluation
    eval_stats = evaluate(model, criterion, val_loader, Config.DEVICE)

    print(f"    Evaluation Complete.")
    print(f"    Eval Loss: {eval_stats['loss']:.4f}")
    print(f"    mAP@0.5:   {eval_stats['map_50']:.4f}")
    print(f"    Study Acc: {eval_stats['study_acc']:.4f}")

    # 8. Utility Verification
    print("\n[7] Verifying Utilities...")
    # Test Box Conversion
    box_cxcywh = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2]]
    )  # Center (0.5, 0.5), width 0.2, height 0.2
    box_xyxy = box_cxcywh_to_xyxy(box_cxcywh)
    expected_xyxy = torch.tensor([[0.4, 0.4, 0.6, 0.6]])
    assert torch.allclose(box_xyxy, expected_xyxy), "Box conversion logic failed."

    # Test IoU
    iou = box_iou(expected_xyxy, expected_xyxy)
    assert torch.isclose(iou, torch.tensor([[1.0]])), "IoU calculation failed."
    print("    Utilities verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
