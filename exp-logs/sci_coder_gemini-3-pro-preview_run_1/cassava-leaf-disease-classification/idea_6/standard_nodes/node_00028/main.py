import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from scipy.stats import pointbiserialr

# Import library modules
from library.config import CFG
from library.utils import seed_everything, get_llrd_params
from library.dataset import get_dataloaders
from library.model import CassavaModel
from library.engine import (
    train_one_epoch,
    valid_one_epoch,
    inference_fn,
    SoftTargetCrossEntropy,
)


def main():
    # 1. Setup & Configuration Override
    seed_everything(CFG.seed)
    CFG.setup()

    # Optimize for A100 GPU and Time Constraints
    print("Configuring for A100 execution...")
    CFG.batch_size = 32  # Increase batch size for A100
    CFG.grad_accumulation = 1  # Reduce accumulation steps
    CFG.num_workers = 12  # Utilize available vCPUs

    # Adjust epochs for a fast but effective baseline
    CFG.epochs_warmup = 1
    CFG.epochs_base = 4
    CFG.epochs_finetune = 2
    CFG.epochs_swa = 2

    device = CFG.device
    print(f"Device: {device}")

    # 2. Phase 1: Data Loading (384x384)
    print("\n=== Phase 1: Loading Data (384x384) ===")
    train_loader_384, val_loader_384, _ = get_dataloaders(img_size=CFG.img_size_p1)

    # 3. Model Initialization
    print("\n=== Initializing Model ===")
    model = CassavaModel(pretrained=True)
    model.to(device)

    loss_fn = SoftTargetCrossEntropy()

    # 4. Stage 1: Warmup (Frozen Backbone)
    print("\n=== Stage 1: Warmup (1 Epoch) ===")
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    for epoch in range(CFG.epochs_warmup):
        train_loss = train_one_epoch(
            epoch, model, loss_fn, optimizer, train_loader_384, device
        )
        val_loss, val_acc = valid_one_epoch(
            epoch, model, loss_fn, val_loader_384, device
        )

    # 5. Stage 2: Base Training (Unfrozen, LLRD)
    print(f"\n=== Stage 2: Base Training ({CFG.epochs_base} Epochs) ===")
    # Unfreeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = True

    # LLRD Optimizer
    optimizer_params = get_llrd_params(
        model,
        base_lr=CFG.lr,
        weight_decay=CFG.weight_decay,
        decay_factor=CFG.llrd_decay,
    )
    optimizer = optim.AdamW(optimizer_params)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs_base, eta_min=CFG.min_lr
    )

    best_acc = 0.0
    best_model_path = os.path.join(CFG.checkpoint_dir, "base_best.pth")

    for epoch in range(CFG.epochs_base):
        train_loss = train_one_epoch(
            epoch, model, loss_fn, optimizer, train_loader_384, device
        )
        val_loss, val_acc = valid_one_epoch(
            epoch, model, loss_fn, val_loader_384, device
        )
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  [Saved Best Base Model] Acc: {best_acc:.6f}")

    # Load best model for next phase
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # 6. Phase 2: Data Loading (512x512)
    print("\n=== Phase 2: Loading Data (512x512) ===")
    train_loader_512, val_loader_512, test_loader = get_dataloaders(
        img_size=CFG.img_size_p2
    )

    # 7. Stage 3: Fine-tuning
    print(f"\n=== Stage 3: Fine-tuning ({CFG.epochs_finetune} Epochs) ===")

    # Lower LR for fine-tuning
    optimizer_params = get_llrd_params(
        model,
        base_lr=CFG.lr * 0.5,
        weight_decay=CFG.weight_decay,
        decay_factor=CFG.llrd_decay,
    )
    optimizer = optim.AdamW(optimizer_params)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs_finetune, eta_min=CFG.min_lr
    )

    ft_best_acc = 0.0
    ft_model_path = os.path.join(CFG.checkpoint_dir, "finetune_best.pth")

    for epoch in range(CFG.epochs_finetune):
        train_loss = train_one_epoch(
            epoch, model, loss_fn, optimizer, train_loader_512, device
        )
        val_loss, val_acc = valid_one_epoch(
            epoch, model, loss_fn, val_loader_512, device
        )
        scheduler.step()

        if val_acc > ft_best_acc:
            ft_best_acc = val_acc
            torch.save(model.state_dict(), ft_model_path)
            print(f"  [Saved Best Fine-tuned Model] Acc: {ft_best_acc:.6f}")

    # Load best fine-tuned model
    if os.path.exists(ft_model_path):
        model.load_state_dict(torch.load(ft_model_path, map_location=device))

    # 8. Stage 4: Stochastic Weight Averaging (SWA)
    print(f"\n=== Stage 4: SWA ({CFG.epochs_swa} Epochs) ===")

    # Constant LR for SWA
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.min_lr * 5, weight_decay=CFG.weight_decay
    )
    swa_snapshots = []

    for epoch in range(CFG.epochs_swa):
        train_loss = train_one_epoch(
            epoch, model, loss_fn, optimizer, train_loader_512, device
        )
        val_loss, val_acc = valid_one_epoch(
            epoch, model, loss_fn, val_loader_512, device
        )

        snap_path = os.path.join(CFG.checkpoint_dir, f"swa_snapshot_{epoch}.pth")
        torch.save(model.state_dict(), snap_path)
        swa_snapshots.append(snap_path)
        print(f"  [Saved SWA Snapshot {epoch}]")

    # Average Weights
    print("Averaging SWA weights...")
    if swa_snapshots:
        avg_state_dict = torch.load(swa_snapshots[0], map_location=device)
        for path in swa_snapshots[1:]:
            state_dict = torch.load(path, map_location=device)
            for key in avg_state_dict:
                avg_state_dict[key] += state_dict[key]

        for key in avg_state_dict:
            avg_state_dict[key] /= len(swa_snapshots)

        model.load_state_dict(avg_state_dict)

    # Update BN Statistics
    print("Updating BN statistics...")
    model.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader_512):
            if i > 100:
                break  # Sufficient subset
            images = images.to(device)
            model(images)

    # 9. Final Validation
    print("\n=== Final Validation ===")
    _, final_acc = valid_one_epoch(0, model, loss_fn, val_loader_512, device)
    print(f"Final Validation Metric: {final_acc}")

    # 10. Failure Analysis
    print("\n=== Failure Analysis ===")
    val_df = pd.read_csv(CFG.val_csv)

    # Generate predictions on validation set
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in val_loader_512:
            images = images.to(device)
            # Use TTA for robust analysis
            out = model(images)
            if CFG.tta:
                out += model(torch.flip(images, [3]))
                out += model(torch.flip(images, [2]))
                out /= 3.0
            preds.append(out.softmax(1).cpu().numpy())

    preds = np.concatenate(preds)
    pred_labels = preds.argmax(1)
    true_labels = val_df["label"].values

    # Binary error (1 if wrong, 0 if correct)
    errors = (pred_labels != true_labels).astype(int)

    # Correlate with File Size
    print("Correlating errors with file size...")
    file_sizes = []
    for rel_path in val_df["file_path"]:
        full_path = os.path.join(CFG.input_root, rel_path)
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))
        else:
            file_sizes.append(0)

    if len(file_sizes) == len(errors):
        corr, p_val = pointbiserialr(errors, file_sizes)
        print(
            f"Correlation between Error and File Size: {corr:.4f} (p-value: {p_val:.4f})"
        )
    else:
        print("Mismatch in metadata length, skipping correlation.")

    # 11. Submission
    threshold = 0.9041388518024032
    if final_acc > threshold:
        print("\n=== Generating Submission ===")
        test_preds = inference_fn(model, test_loader, device)
        test_labels = test_preds.argmax(1)

        sub_df = pd.read_csv(CFG.test_csv)
        sub_df["label"] = test_labels

        # Ensure correct format
        sub_df = sub_df[["image_id", "label"]]
        sub_df.to_csv(CFG.submission_path, index=False)
        print(f"Submission saved to {CFG.submission_path}")
    else:
        print(
            f"Validation score {final_acc} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
