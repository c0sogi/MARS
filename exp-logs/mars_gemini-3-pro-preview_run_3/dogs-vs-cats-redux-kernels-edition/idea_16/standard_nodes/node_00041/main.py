import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import (
    seed_everything,
    get_device,
    save_checkpoint,
    load_checkpoint,
    calculate_log_loss,
)
from library.dataset import DogCatDataset, get_transforms
from library.models import get_model
from library.engine import train_one_epoch, evaluate, predict, EarlyStopping


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()

    # Override Config for Fast Baseline Execution within time limits
    # Reducing epochs to ensure completion within 2 hours while maintaining multi-stage pipeline
    Config.MODELS["resnet50"]["teacher_epochs"] = 3
    Config.MODELS["resnet50"]["student_epochs"] = 4
    Config.MODELS["convnext_small"]["teacher_epochs"] = 3
    Config.MODELS["convnext_small"]["student_epochs"] = 4
    Config.MODELS["maxvit_tiny"]["teacher_epochs"] = 3
    Config.MODELS["maxvit_tiny"]["student_epochs"] = 4

    print(f"Device: {device}")
    print("Loading Metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # -------------------------------------------------------------------------
    # Phase 1: Teacher Training
    # -------------------------------------------------------------------------
    print("\n=== Phase 1: Teacher Training ===")

    model_names = ["resnet50", "convnext_small", "maxvit_tiny"]

    for model_name in model_names:
        print(f"\nTraining Teacher: {model_name}")
        cfg = Config.MODELS[model_name]

        # Data
        train_dataset = DogCatDataset(
            train_df, transform=get_transforms("train", cfg["img_size"]), mode="train"
        )
        val_dataset = DogCatDataset(
            val_df, transform=get_transforms("val", cfg["img_size"]), mode="train"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model
        model = get_model(model_name, pretrained=True)

        # Optimizer & Scheduler
        optimizer = getattr(optim, Config.OPTIMIZER)(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        scheduler = getattr(optim.lr_scheduler, Config.SCHEDULER)(
            optimizer, T_max=cfg["teacher_epochs"], eta_min=Config.MIN_LR
        )

        # Training Loop
        best_loss = np.inf
        save_path = os.path.join(Config.WORKING_DIR, f"{model_name}_teacher_best.pth")

        for epoch in range(cfg["teacher_epochs"]):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, epoch + 1
            )
            val_loss = evaluate(model, val_loader, device)
            scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint({"model_state_dict": model.state_dict()}, save_path)
                print(f"Saved Best Teacher {model_name}: {best_loss:.6f}")

        # Free memory
        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
        )
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Phase 2: Pseudo-Label Generation
    # -------------------------------------------------------------------------
    print("\n=== Phase 2: Pseudo-Label Generation ===")

    # Accumulate predictions from all teachers
    test_preds_accum = np.zeros(len(test_df))

    for model_name in model_names:
        print(f"Generating pseudo-labels with {model_name}...")
        cfg = Config.MODELS[model_name]

        # Load Best Teacher
        model = get_model(model_name, pretrained=False)
        ckpt_path = os.path.join(Config.WORKING_DIR, f"{model_name}_teacher_best.pth")
        load_checkpoint(ckpt_path, model, device=device)
        model.to(device)

        # Test Loader
        test_dataset = DogCatDataset(
            test_df, transform=get_transforms("val", cfg["img_size"]), mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Predict (Returns list of (id, prob))
        preds = predict(model, test_loader, device, tta=Config.TTA_FLIP)

        # Map predictions to dataframe order
        preds_dict = {int(i): p for i, p in preds}
        ordered_probs = np.array(
            [preds_dict[row_id] for row_id in test_df["id"].values]
        )

        test_preds_accum += ordered_probs

        del model, test_loader, test_dataset
        torch.cuda.empty_cache()

    # Average predictions to get soft labels
    avg_test_preds = test_preds_accum / len(model_names)

    # Create Pseudo-Labeled Dataframe
    pseudo_df = test_df.copy()
    pseudo_df["label"] = avg_test_preds
    pseudo_df = pseudo_df[["filepath", "label"]]

    # Combine with Train Data
    combined_df = pd.concat(
        [train_df[["filepath", "label"]], pseudo_df], ignore_index=True
    )
    print(
        f"Combined Dataset Size: {len(combined_df)} (Train: {len(train_df)} + Pseudo: {len(pseudo_df)})"
    )

    # -------------------------------------------------------------------------
    # Phase 3: Student Training
    # -------------------------------------------------------------------------
    print("\n=== Phase 3: Student Training ===")

    for model_name in model_names:
        print(f"\nTraining Student: {model_name}")
        cfg = Config.MODELS[model_name]

        # Data (Combined)
        train_dataset = DogCatDataset(
            combined_df,
            transform=get_transforms("train", cfg["img_size"]),
            mode="train",
        )
        val_dataset = DogCatDataset(
            val_df, transform=get_transforms("val", cfg["img_size"]), mode="train"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model (Re-initialize from ImageNet weights)
        model = get_model(model_name, pretrained=True)

        # Optimizer & Scheduler
        optimizer = getattr(optim, Config.OPTIMIZER)(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        scheduler = getattr(optim.lr_scheduler, Config.SCHEDULER)(
            optimizer, T_max=cfg["student_epochs"], eta_min=Config.MIN_LR
        )

        # Training Loop
        best_loss = np.inf
        save_path = os.path.join(Config.WORKING_DIR, f"{model_name}_student_best.pth")

        for epoch in range(cfg["student_epochs"]):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, epoch + 1
            )
            val_loss = evaluate(model, val_loader, device)
            scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint({"model_state_dict": model.state_dict()}, save_path)
                print(f"Saved Best Student {model_name}: {best_loss:.6f}")

        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
        )
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Phase 4: Final Validation & Analysis
    # -------------------------------------------------------------------------
    print("\n=== Phase 4: Final Validation & Analysis ===")

    val_preds_accum = np.zeros(len(val_df))

    for model_name in model_names:
        print(f"Inferencing Validation with Student {model_name}...")
        cfg = Config.MODELS[model_name]

        model = get_model(model_name, pretrained=False)
        ckpt_path = os.path.join(Config.WORKING_DIR, f"{model_name}_student_best.pth")
        load_checkpoint(ckpt_path, model, device=device)
        model.to(device)

        val_dataset = DogCatDataset(
            val_df, transform=get_transforms("val", cfg["img_size"]), mode="train"
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        model.eval()
        preds = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).view(-1)

                if Config.TTA_FLIP:
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped = model(images_flipped)
                    probs_flipped = torch.sigmoid(logits_flipped).view(-1)
                    probs = (probs + probs_flipped) / 2.0

                preds.extend(probs.cpu().numpy())

        val_preds_accum += np.array(preds)
        del model, val_loader, val_dataset
        torch.cuda.empty_cache()

    final_val_probs = val_preds_accum / len(model_names)
    final_val_loss = calculate_log_loss(val_df["label"].values, final_val_probs)

    print(f"Final Validation Metric: {final_val_loss}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    val_df["pred"] = final_val_probs
    val_df["error"] = np.abs(val_df["label"] - val_df["pred"])

    # Calculate metadata features for correlation
    widths, heights, file_sizes = [], [], []

    for idx, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["filepath"])
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        else:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    val_df["width"] = widths
    val_df["height"] = heights
    val_df["file_size"] = file_sizes

    features = ["width", "height", "file_size"]
    print("Correlation between Error Magnitude and Features:")
    for feat in features:
        if len(val_df) > 1 and val_df[feat].std() > 0:
            corr, _ = pearsonr(val_df["error"], val_df[feat])
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: N/A (Constant or Empty)")

    # -------------------------------------------------------------------------
    # Phase 5: Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.009074434935821756

    if final_val_loss < THRESHOLD:
        print(
            f"\nValidation Loss ({final_val_loss}) < Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_preds_accum = np.zeros(len(test_df))

        for model_name in model_names:
            print(f"Predicting Test with Student {model_name}...")
            cfg = Config.MODELS[model_name]

            model = get_model(model_name, pretrained=False)
            ckpt_path = os.path.join(
                Config.WORKING_DIR, f"{model_name}_student_best.pth"
            )
            load_checkpoint(ckpt_path, model, device=device)
            model.to(device)

            test_dataset = DogCatDataset(
                test_df, transform=get_transforms("val", cfg["img_size"]), mode="test"
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            preds = predict(model, test_loader, device, tta=Config.TTA_FLIP)

            preds_dict = {int(i): p for i, p in preds}
            ordered_probs = np.array(
                [preds_dict[row_id] for row_id in test_df["id"].values]
            )

            test_preds_accum += ordered_probs

            del model, test_loader, test_dataset
            torch.cuda.empty_cache()

        final_test_probs = test_preds_accum / len(model_names)

        submission_df = pd.DataFrame({"id": test_df["id"], "label": final_test_probs})

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Loss ({final_val_loss}) >= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
