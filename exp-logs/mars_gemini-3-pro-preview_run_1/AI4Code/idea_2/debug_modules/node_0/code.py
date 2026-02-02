import os
import shutil
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library import data_utils
from library import feature_extraction
from library import dataset
from library import model
from library import loss
from library import inference_utils
from library import train


def run_demo():
    print("=== Starting HAPS Library Demo ===")

    # --- 1. Setup Temporary Directory and Dummy Data ---
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Created temporary directory: {demo_dir}")

    # Load original metadata
    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    orig_val_meta = pd.read_csv("./metadata/val_metadata.csv")
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Create subsets (5 samples each)
    subset_size = 5
    train_meta_subset = orig_train_meta.head(subset_size).copy()
    val_meta_subset = orig_val_meta.head(subset_size).copy()
    test_meta_subset = orig_test_meta.head(subset_size).copy()

    # Save subsets
    demo_train_meta_path = os.path.join(demo_dir, "train_metadata.csv")
    demo_val_meta_path = os.path.join(demo_dir, "val_metadata.csv")
    demo_test_meta_path = os.path.join(demo_dir, "test_metadata.csv")

    train_meta_subset.to_csv(demo_train_meta_path, index=False)
    val_meta_subset.to_csv(demo_val_meta_path, index=False)
    test_meta_subset.to_csv(demo_test_meta_path, index=False)

    print("Created subset metadata files.")

    # --- 2. Override Configuration ---
    # We modify the Config class attributes directly to affect the library modules
    Config.working_dir = demo_dir
    Config.train_metadata_path = demo_train_meta_path
    Config.val_metadata_path = demo_val_meta_path
    Config.test_metadata_path = demo_test_meta_path

    Config.train_features_path = os.path.join(demo_dir, "train_features.parquet")
    Config.val_features_path = os.path.join(demo_dir, "val_features.parquet")
    Config.test_features_path = os.path.join(demo_dir, "test_features.parquet")

    Config.model_save_path = os.path.join(demo_dir, "best_model.pth")
    Config.submission_path = os.path.join(demo_dir, "submission.csv")

    Config.num_epochs = 1
    Config.batch_size = 2
    Config.num_workers = 0  # Avoid multiprocessing overhead for small demo

    # Re-run setup to ensure directories exist
    Config.setup()
    print("Configuration updated for demo run.")

    # --- 3. Verify Data Utils ---
    print("\n--- Testing Data Utils ---")
    sample_nb_id = train_meta_subset.iloc[0]["id"]
    sample_filepath = train_meta_subset.iloc[0]["filepath"]
    sample_order = train_meta_subset.iloc[0]["cell_order"]

    # Test read_notebook
    nb_json = data_utils.read_notebook(sample_filepath)
    assert isinstance(nb_json, dict), "read_notebook should return a dictionary"
    assert "source" in nb_json, "Notebook JSON should contain 'source' field"

    # Test preprocess_text
    raw_text = "  import numpy as np  "
    clean_text = data_utils.preprocess_text(raw_text)
    assert (
        clean_text == "import numpy as np"
    ), "preprocess_text failed to strip whitespace"

    # Test load_notebook_cells
    cells = data_utils.load_notebook_cells(sample_nb_id, sample_filepath, sample_order)
    assert isinstance(cells, list), "load_notebook_cells should return a list"
    assert len(cells) > 0, "Notebook should have cells"
    assert "rel_rank" in cells[0], "Cells should have rank info when order is provided"
    print("Data Utils verification passed.")

    # --- 4. Verify Feature Extraction ---
    print("\n--- Testing Feature Extraction ---")
    # This will generate parquet files in the demo directory
    # We explicitly force re-computation by setting load_cached_data=False
    df_features = feature_extraction.process_dataset(
        Config.train_metadata_path, Config.train_features_path, load_cached_data=False
    )

    assert os.path.exists(
        Config.train_features_path
    ), "Feature parquet file was not created"
    assert not df_features.empty, "Feature DataFrame is empty"
    assert "embedding" in df_features.columns, "Embeddings missing from features"
    # Check embedding dimension (Config.input_dim is 384 for MiniLM)
    assert (
        len(df_features.iloc[0]["embedding"]) == Config.input_dim
    ), "Incorrect embedding dimension"
    print("Feature Extraction verification passed.")

    # Generate val and test features for later steps
    feature_extraction.process_dataset(
        Config.val_metadata_path, Config.val_features_path, load_cached_data=False
    )
    feature_extraction.process_dataset(
        Config.test_metadata_path, Config.test_features_path, load_cached_data=False
    )

    # --- 5. Verify Dataset and DataLoader ---
    print("\n--- Testing Dataset & DataLoader ---")
    ds = dataset.HAPSDataset(Config.train_features_path, mode="train")
    assert (
        len(ds) == subset_size
    ), f"Dataset length mismatch. Expected {subset_size}, got {len(ds)}"

    sample_item = ds[0]
    assert "code_embeddings" in sample_item
    assert "md_embeddings" in sample_item
    assert "anchor_labels" in sample_item
    assert isinstance(sample_item["code_embeddings"], torch.Tensor)

    # Test Collate Function
    dl = DataLoader(ds, batch_size=2, collate_fn=dataset.haps_collate_fn)
    batch = next(iter(dl))

    assert "code_embeddings" in batch
    assert (
        batch["code_embeddings"].dim() == 3
    ), "Code embeddings should be (Batch, Seq, Dim)"
    assert "anchor_labels" in batch
    # Check masking logic: mask should be False for padded areas (or True for valid, depending on implementation)
    # Implementation says: code_mask[i, :n_code] = True. So True = Valid.
    assert batch["code_mask"].dtype == torch.bool
    print("Dataset & DataLoader verification passed.")

    # --- 6. Verify Model and Loss ---
    print("\n--- Testing Model & Loss ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = model.HAPSModel().to(device)
    crit = loss.HAPSLoss()

    # Move batch to device
    code_emb = batch["code_embeddings"].to(device)
    code_mask = batch["code_mask"].to(device)
    md_emb = batch["md_embeddings"].to(device)
    md_mask = batch["md_mask"].to(device)

    # Handle optional pairwise inputs
    pairwise_indices = batch.get("pairwise_indices")
    if pairwise_indices is not None:
        pairwise_indices = pairwise_indices.to(device)

    # Forward Pass
    outputs = net(code_emb, code_mask, md_emb, md_mask, pairwise_indices)
    assert "anchor_logits" in outputs
    # Anchor logits shape: (Batch, MaxMD, MaxCode + 1)
    B, MaxMD, MaxCodePlus1 = outputs["anchor_logits"].shape
    assert B == code_emb.shape[0]
    assert MaxMD == md_emb.shape[1]
    assert MaxCodePlus1 == code_emb.shape[1] + 1

    # Loss Calculation
    # Prepare batch dict for loss (needs labels on device)
    batch_labels = {
        "anchor_labels": batch["anchor_labels"].to(device),
        "pairwise_labels": (
            batch["pairwise_labels"].to(device) if "pairwise_labels" in batch else None
        ),
    }

    loss_out = crit(outputs, batch_labels)
    assert "loss" in loss_out
    assert not torch.isnan(loss_out["loss"]), "Loss is NaN"
    print("Model & Loss verification passed.")

    # --- 7. Verify Inference Utils ---
    print("\n--- Testing Inference Utils ---")
    # Use the outputs from the previous step
    # Mock cell IDs
    dummy_code_ids = [f"code_{i}" for i in range(code_emb.shape[1])]
    dummy_md_ids = [f"md_{i}" for i in range(md_emb.shape[1])]

    # Take first sample from batch
    single_anchor_logits = outputs["anchor_logits"][0:1]  # Keep batch dim 1

    # Global Sort
    sorted_cells = inference_utils.compute_global_sort(
        single_anchor_logits, dummy_code_ids, dummy_md_ids
    )
    assert isinstance(sorted_cells, list)
    assert len(sorted_cells) == len(dummy_code_ids) + len(dummy_md_ids)
    assert "position" in sorted_cells[0]

    # Refinement
    # Just ensure it runs without error
    refined_order = inference_utils.refine_order(
        net, sorted_cells, md_emb[0:1], device, passes=1
    )
    assert isinstance(refined_order, list)
    assert len(refined_order) == len(sorted_cells)
    print("Inference Utils verification passed.")

    # --- 8. Integration Test: Full Training Loop ---
    print("\n--- Running Full Training Loop (Integration Test) ---")
    # This calls the train() function from library.train
    # It will use the Config paths we overrode earlier
    try:
        train.train()
        print("Training loop completed successfully.")
    except Exception as e:
        print(f"Training loop failed: {e}")
        raise e

    # Verify submission file
    assert os.path.exists(Config.submission_path), "Submission file not found"
    df_sub = pd.read_csv(Config.submission_path)
    assert len(df_sub) == subset_size, f"Submission should have {subset_size} rows"
    print("Integration test passed.")

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    run_demo()
