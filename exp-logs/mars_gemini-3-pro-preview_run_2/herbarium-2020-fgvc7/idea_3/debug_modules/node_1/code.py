import os
import sys
import shutil
import pandas as pd
import torch
import torch.optim as optim
import warnings

# Import provided library modules
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import HierarchicalEfficientNet
from library.trainer import Trainer

# Suppress potential warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting Demo Script...")

    # 1. Setup and Configuration
    # --------------------------
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # Define paths
    # We create a proxy input directory to satisfy the path assumptions in library/dataset.py
    # (which expects metadata to be inside input_dir) while allowing us to modify the
    # test metadata (truncate it) for a fast inference demonstration.

    base_work_dir = os.path.abspath("./working")
    proxy_input_dir = os.path.join(base_work_dir, "input_proxy")

    # Clean up previous run if exists
    if os.path.exists(proxy_input_dir):
        shutil.rmtree(proxy_input_dir)
    os.makedirs(proxy_input_dir)

    print("Setting up proxy input directory...")

    # Symlink the read-only image data: ./input/nybg2020 -> ./working/input_proxy/nybg2020
    # This allows the dataset loader to find images at input_dir/nybg2020/...
    original_data_dir = os.path.abspath("./input/nybg2020")
    proxy_data_dir = os.path.join(proxy_input_dir, "nybg2020")
    os.symlink(original_data_dir, proxy_data_dir)

    # Copy the metadata: ./metadata -> ./working/input_proxy/metadata
    # We copy instead of symlink so we can modify test.csv
    original_meta_dir = os.path.abspath("./metadata")
    proxy_meta_dir = os.path.join(proxy_input_dir, "metadata")
    shutil.copytree(original_meta_dir, proxy_meta_dir)

    # Truncate test.csv to 10 rows for rapid inference demonstration
    test_csv_path = os.path.join(proxy_meta_dir, "test.csv")
    df_test = pd.read_csv(test_csv_path)
    df_test_subset = df_test.head(10)
    df_test_subset.to_csv(test_csv_path, index=False)
    print(f"Truncated test metadata to {len(df_test_subset)} rows.")

    # 2. Data Loading Demonstration
    # -----------------------------
    print("\n[Demo] Loading DataLoaders...")
    # We use debug_size=50 to load only a tiny fraction of the training data
    train_loader, val_loader, num_species, num_genera, num_families, label_map = (
        get_dataloaders(
            input_dir=proxy_input_dir,
            batch_size=4,
            image_size=224,  # Smaller size for speed
            num_workers=2,
            debug_size=50,  # Only use 50 images for training/val
            load_cached_data=False,  # Force recompute for demo purposes
        )
    )

    print(
        f"Classes detected - Species: {num_species}, Genera: {num_genera}, Families: {num_families}"
    )

    # Verify Train Loader
    batch = next(iter(train_loader))
    images = batch["image"]
    targets = batch["species"]

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Species Targets: {targets}")

    assert images.shape == (4, 3, 224, 224), "Incorrect image batch shape"
    assert "genus" in batch and "family" in batch, "Missing hierarchical labels"

    # 3. Model Instantiation Demonstration
    # ------------------------------------
    print("\n[Demo] Instantiating Model...")
    # We use pretrained=False to avoid downloading weights during this time-constrained demo
    model = HierarchicalEfficientNet(
        num_species=num_species,
        num_genera=num_genera,
        num_families=num_families,
        pretrained=False,
        dropout_p=0.2,
    )
    model = model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        # Move a sample batch to device
        imgs_dev = images.to(device)
        s_logits, g_logits, f_logits = model(imgs_dev)

    print(
        f"Output Shapes - Species: {s_logits.shape}, Genus: {g_logits.shape}, Family: {f_logits.shape}"
    )
    assert s_logits.shape == (4, num_species)
    assert g_logits.shape == (4, num_genera)
    assert f_logits.shape == (4, num_families)

    # 4. Training Loop Demonstration
    # ------------------------------
    print("\n[Demo] Starting Training Loop...")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    run_dir = os.path.join(base_work_dir, "demo_run_script")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        label_map=label_map,
        save_dir=run_dir,
    )

    # Train for 1 epoch
    trainer.fit(num_epochs=1)

    # Verify model checkpoint or log
    # Note: If validation F1 doesn't improve (possible with random init/tiny data),
    # best_model.pth might not be saved. We check log existence as proof of run.
    log_path = os.path.join(run_dir, "train.log")
    assert os.path.exists(log_path), "Training log not found."
    print("Training completed.")

    # 5. Inference Demonstration
    # --------------------------
    print("\n[Demo] Running Inference...")

    # Get test loader (uses the truncated test.csv)
    test_loader = get_test_dataloader(
        input_dir=proxy_input_dir, batch_size=4, image_size=224, num_workers=2
    )

    submission_dir = os.path.join(base_work_dir, "submission")
    trainer.predict(test_loader, output_dir=submission_dir)

    submission_file = os.path.join(submission_dir, "submission.csv")
    assert os.path.exists(submission_file), "Submission file was not generated."

    # Verify submission content
    df_sub = pd.read_csv(submission_file)
    print(f"Submission Head:\n{df_sub.head()}")

    assert len(df_sub) == 10, f"Expected 10 predictions, got {len(df_sub)}"
    assert list(df_sub.columns) == ["Id", "Predicted"], "Incorrect submission columns"

    print("\n=== All Demonstrations Passed Successfully ===")


if __name__ == "__main__":
    main()
