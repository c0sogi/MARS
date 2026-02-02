import os
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import CFG
from library.utils import seed_everything
from library.dataset import process_metadata, get_transforms, HotelDataset
from library.model import HotelNet
from library.engine import train_loop
from library.inference import inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Hotel ID Demo ===")

    # 1. Configuration Overrides for Speed and Demo Purposes
    print("Configuring demo parameters...")
    CFG.seed = 42
    CFG.debug = True  # Limits iterations in engine.py
    CFG.epochs = 1
    CFG.batch_size = 8
    CFG.num_workers = 0  # Avoid multiprocessing overhead for small data
    CFG.working_dir = "./working/demo_run"
    CFG.backbone = "resnet18"  # Use lightweight backbone
    CFG.pretrained = False  # Skip downloading weights
    CFG.embedding_size = 128  # Smaller embedding for speed

    # Create working directory
    os.makedirs(CFG.working_dir, exist_ok=True)

    # Set seed
    seed_everything(CFG.seed)

    # 2. Data Preparation
    print("\n[Data] Processing metadata...")
    # process_metadata loads csvs, maps classes, and saves encodings to working_dir
    train_df, val_df, test_df, hotel_classes, chain_classes = process_metadata(
        load_cached_data=False
    )

    print(f"Original Train size: {len(train_df)}")
    print(f"Original Val size: {len(val_df)}")
    print(f"Original Test size: {len(test_df)}")

    # Subset data for demo speed
    train_subset = train_df.head(50).copy().reset_index(drop=True)
    val_subset = val_df.head(20).copy().reset_index(drop=True)
    test_subset = test_df.head(20).copy().reset_index(drop=True)

    print(f"Subset Train size: {len(train_subset)}")

    # Initialize Datasets
    print("\n[Data] Initializing Datasets...")
    train_ds = HotelDataset(
        train_subset, transform=get_transforms("train"), mode="train"
    )
    val_ds = HotelDataset(val_subset, transform=get_transforms("val"), mode="val")
    test_ds = HotelDataset(test_subset, transform=get_transforms("test"), mode="test")

    # Verify Dataset Output
    img, hotel_lbl, chain_lbl = train_ds[0]
    assert img.shape == (
        3,
        CFG.image_size,
        CFG.image_size,
    ), f"Unexpected image shape: {img.shape}"
    assert isinstance(hotel_lbl, torch.Tensor), "Hotel label must be a tensor"
    assert isinstance(chain_lbl, torch.Tensor), "Chain label must be a tensor"
    print("Dataset verification passed.")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers
    )

    # 3. Model Initialization
    print("\n[Model] Initializing HotelNet...")
    # Update CFG num_classes based on actual data loaded (though process_metadata handles full set)
    CFG.num_classes = len(hotel_classes)
    CFG.num_chains = len(chain_classes)

    device = torch.device(CFG.device)
    model = HotelNet()
    model.to(device)

    # Verify Model Forward Pass
    dummy_input = torch.randn(2, 3, CFG.image_size, CFG.image_size).to(device)
    dummy_hotel = torch.tensor([0, 1]).to(device)
    dummy_chain = torch.tensor([0, 1]).to(device)

    # Train mode forward
    model.train()
    h_logits, c_logits = model(dummy_input, dummy_hotel, dummy_chain)
    assert h_logits.shape == (
        2,
        CFG.num_classes,
    ), f"Hotel logits shape mismatch: {h_logits.shape}"
    assert c_logits.shape == (
        2,
        CFG.num_chains,
    ), f"Chain logits shape mismatch: {c_logits.shape}"

    # Eval mode forward
    model.eval()
    emb = model(dummy_input)
    assert emb.shape == (
        2,
        CFG.embedding_size,
    ), f"Embedding shape mismatch: {emb.shape}"
    print("Model verification passed.")

    # 4. Training Loop
    print("\n[Training] Starting Training Loop...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
    )

    # Run training (Debug mode will limit steps)
    model = train_loop(
        train_loader, val_loader, model, optimizer, scheduler, device, epochs=CFG.epochs
    )
    print("Training loop completed.")

    # 5. Inference
    print("\n[Inference] Generating predictions...")
    # Ensure submission directory exists (inference function creates it, but good to be safe)
    os.makedirs("submission", exist_ok=True)

    submission_df = inference(test_loader, model, device, hotel_classes)

    # 6. Final Validation
    print("\n[Validation] Checking submission...")
    sub_path = "./submission/submission.csv"

    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Submission file not found at {sub_path}")

    df_sub = pd.read_csv(sub_path)
    print(f"Submission shape: {df_sub.shape}")

    # Check columns
    assert "image" in df_sub.columns, "Submission missing 'image' column"
    assert "hotel_id" in df_sub.columns, "Submission missing 'hotel_id' column"

    # Check row count matches test subset
    assert len(df_sub) == len(
        test_subset
    ), f"Submission row count {len(df_sub)} != Test subset {len(test_subset)}"

    # Check format of prediction string
    sample_pred = df_sub.iloc[0]["hotel_id"]
    assert isinstance(sample_pred, str), "Prediction must be a string"
    pred_ids = sample_pred.split(" ")
    assert len(pred_ids) == 5, f"Expected 5 predictions per image, got {len(pred_ids)}"

    print("Submission validation passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
