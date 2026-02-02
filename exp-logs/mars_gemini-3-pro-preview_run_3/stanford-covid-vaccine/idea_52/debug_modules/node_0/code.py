import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library components
from library.config import Config
from library.utils import set_seed, mcrmse_loss, calculate_metric
from library.data import get_dataloaders
from library.model import DeepStabilizedBiGRU
from library.train import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class PatchedTrainer(Trainer):
    """
    A subclass of Trainer that fixes the validation logic to ensure
    targets are sliced correctly before being passed to calculate_metric.
    """

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, pair_indices, targets in self.val_loader:
                inputs = inputs.to(self.device)
                pair_indices = pair_indices.to(self.device)

                # Forward pass
                outputs = self.model(inputs, pair_indices)

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # FIX: Slice targets to Config.SEQ_SCORED to match calculate_metric expectations
        # calculate_metric expects targets to be of length SEQ_SCORED (68)
        # while the DataLoader provides padded targets of length SEQ_LEN (107).
        all_targets_sliced = all_targets[:, : Config.SEQ_SCORED, :]

        score = calculate_metric(all_preds, all_targets_sliced)
        return score


def main():
    # 1. Setup and Reproducibility
    print(">>> Setting up environment...")
    set_seed(42)

    # Define temporary working directory
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # 2. Override Configuration for Speed and Demo Purposes
    print(">>> Configuring parameters for rapid execution...")
    Config.WORKING_DIR = DEMO_DIR
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "demo_model.pth")
    Config.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "train_cache.npy")
    Config.VAL_CACHE_PATH = os.path.join(DEMO_DIR, "val_cache.npy")
    Config.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "test_cache.npy")

    # Reduce computational load
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.HIDDEN_DIM = 64
    Config.NUM_LAYERS = 2
    Config.CONV_FILTERS = 32
    Config.DROPOUT = 0.0

    # 3. Create Data Subsets
    print(">>> Creating data subsets...")
    # Load original metadata
    df_train = pd.read_parquet("./metadata/train.parquet")
    df_val = pd.read_parquet("./metadata/val.parquet")
    df_test = pd.read_parquet("./metadata/test.parquet")

    # Create tiny subsets (e.g., 32 samples for train, 16 for val/test)
    sub_train = df_train.head(32)
    sub_val = df_val.head(16)
    sub_test = df_test.head(16)

    # Save subsets
    train_sub_path = os.path.join(DEMO_DIR, "train_subset.parquet")
    val_sub_path = os.path.join(DEMO_DIR, "val_subset.parquet")
    test_sub_path = os.path.join(DEMO_DIR, "test_subset.parquet")

    sub_train.to_parquet(train_sub_path)
    sub_val.to_parquet(val_sub_path)
    sub_test.to_parquet(test_sub_path)

    # Point Config to subsets
    Config.TRAIN_METADATA_PATH = train_sub_path
    Config.VAL_METADATA_PATH = val_sub_path
    Config.TEST_METADATA_PATH = test_sub_path

    # 4. Data Loading and Verification
    print(">>> Loading data...")
    # load_cached_data=False forces reprocessing of our new subsets
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Train Batch
    inputs, pair_indices, targets = next(iter(train_loader))
    print(
        f"    Batch shapes: Inputs={inputs.shape}, Pairs={pair_indices.shape}, Targets={targets.shape}"
    )

    assert inputs.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM)
    assert pair_indices.shape == (Config.BATCH_SIZE, Config.SEQ_LEN)
    assert targets.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert not torch.isnan(inputs).any(), "Inputs contain NaNs"

    # 5. Model Initialization and Forward Pass
    print(">>> Initializing model...")
    model = DeepStabilizedBiGRU()
    model.to(Config.DEVICE)

    print(">>> Running dummy forward pass...")
    inputs = inputs.to(Config.DEVICE)
    pair_indices = pair_indices.to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(inputs, pair_indices)

    print(f"    Output shape: {outputs.shape}")
    assert outputs.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert not torch.isnan(outputs).any(), "Model outputs contain NaNs"

    # 6. Metric Verification
    print(">>> Verifying metrics...")
    outputs_cpu = outputs.cpu()
    targets_cpu = targets.cpu()

    # Check Loss Function
    loss = mcrmse_loss(outputs_cpu, targets_cpu)
    assert loss.ndim == 0, "Loss must be a scalar"
    print(f"    Initial Loss: {loss.item():.4f}")

    # Check Competition Metric
    # Note: We must slice targets to SEQ_SCORED (68) for the metric function
    targets_sliced = targets_cpu[:, : Config.SEQ_SCORED, :]
    metric = calculate_metric(outputs_cpu, targets_sliced)
    assert isinstance(metric, float), "Metric must be a float"
    print(f"    Initial Metric: {metric:.4f}")

    # 7. Training Loop
    print(">>> Starting training loop (PatchedTrainer)...")
    # Use PatchedTrainer to handle the validation target slicing
    trainer = PatchedTrainer(model, train_loader, val_loader)
    trainer.fit()

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Best model was not saved."

    # 8. Inference
    print(">>> Performing inference on test set...")
    # Load best model
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    preds = []

    for inputs, pair_indices, _ in test_loader:
        inputs = inputs.to(Config.DEVICE)
        pair_indices = pair_indices.to(Config.DEVICE)
        with torch.no_grad():
            out = model(inputs, pair_indices)
            preds.append(out.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    print(f"    Predictions shape: {preds.shape}")

    # Verify shape matches test subset count
    n_test = len(sub_test)
    assert preds.shape == (n_test, Config.SEQ_LEN, Config.NUM_TARGETS)

    # 9. Generate Sample Submission Entry
    print(">>> Generating sample submission rows...")
    # Submission format requires flattening: id_seqpos
    sample_id = sub_test.iloc[0]["id"]
    sample_preds = preds[0]  # (107, 5)

    # Just show first 3 positions
    print("    id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C")
    for i in range(3):
        row_id = f"{sample_id}_{i}"
        vals = sample_preds[i]
        print(
            f"    {row_id},{vals[0]:.4f},{vals[1]:.4f},{vals[2]:.4f},{vals[3]:.4f},{vals[4]:.4f}"
        )

    print("\n>>> Demo completed successfully.")


if __name__ == "__main__":
    main()
