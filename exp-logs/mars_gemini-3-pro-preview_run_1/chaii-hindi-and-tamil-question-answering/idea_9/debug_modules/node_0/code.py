import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, jaccard, get_optimizer_params
from library.model import XLMRobertaForQA
from library.data import prepare_train_features, prepare_test_features
from library.engine import train_fn, predict_fn


def main():
    print("==== Starting Demonstration of QA Pipeline ====")

    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    print("\n[1] Initializing Configuration...")
    # Enable debug mode to use a small subset (50 samples) for speed
    config = Config(debug=True)

    # Override model name to 'base' for faster execution during this demo
    # The logic remains identical to 'large'
    config.model_name = "xlm-roberta-base"
    config.batch_size = 2  # Small batch size for demo
    config.epochs = 1

    # Set fixed seed for reproducibility
    set_seed(config.seed)

    print(f"    Model: {config.model_name}")
    print(f"    Device: {config.device}")
    print(f"    Debug Mode: {config.debug}")

    # 2. Tokenizer and Data Preparation
    # -------------------------------------------------------------------------
    print("\n[2] Preparing Data...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Prepare Training Data
    # This function handles:
    # - Loading train/val metadata
    # - Sliding window tokenization
    # - Label creation (start/end positions)
    # - Negative sampling (hard negatives)
    # - Caching (we disable loading cache to demonstrate processing)
    print("    Processing training features...")
    train_dataset = prepare_train_features(config, tokenizer, load_cached_data=False)

    print(f"    Training samples generated: {len(train_dataset)}")

    # Validation: Check dataset item structure
    sample_item = train_dataset[0]
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "relevance_labels",
    ]
    for key in required_keys:
        assert key in sample_item, f"Missing key {key} in dataset item"

    assert sample_item["input_ids"].shape == (
        config.max_len,
    ), "Incorrect input_ids shape"
    assert (
        sample_item["relevance_labels"].dtype == torch.float
    ), "Relevance label should be float"

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple demo to avoid multiprocessing overhead
    )

    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Model...")
    model = XLMRobertaForQA(config.model_name)
    model.to(config.device)

    # Validation: Check model architecture
    assert hasattr(model, "qa_outputs"), "Model missing span head"
    assert hasattr(model, "relevance_classifier"), "Model missing relevance head"

    # 4. Optimization Setup
    # -------------------------------------------------------------------------
    print("\n[4] Setting up Optimizer...")
    # Use the utility function to apply differential learning rates
    optimizer_grouped_parameters = get_optimizer_params(model, config)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    print(f"    Optimizer param groups: {len(optimizer_grouped_parameters)}")
    assert (
        len(optimizer_grouped_parameters) == 2
    ), "Expected 2 parameter groups (backbone vs head)"

    # 5. Training Loop (Single Epoch)
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (1 Epoch)...")
    # train_fn handles:
    # - Mixed Precision (AMP)
    # - Adversarial Training (FGM)
    # - Loss calculation (Span Loss + Relevance Loss)
    avg_loss = train_fn(train_loader, model, optimizer, config.device, config)

    print(f"    Average Training Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss returned NaN"

    # 6. Inference / Prediction
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    # Prepare Test Data
    # This uses exhaustive sliding windows (no negative sampling)
    test_dataset, test_features = prepare_test_features(config, tokenizer)
    print(f"    Test windows generated: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    # Run prediction
    start_preds, end_preds, relevance_preds = predict_fn(
        test_loader, model, config.device
    )

    # Validation: Check output shapes
    n_samples = len(test_dataset)
    assert start_preds.shape == (
        n_samples,
        config.max_len,
    ), f"Shape mismatch: {start_preds.shape}"
    assert end_preds.shape == (
        n_samples,
        config.max_len,
    ), f"Shape mismatch: {end_preds.shape}"
    assert relevance_preds.shape == (
        n_samples,
    ), f"Shape mismatch: {relevance_preds.shape}"

    print("    Inference shapes verified.")

    # 7. Post-Processing Demonstration
    # -------------------------------------------------------------------------
    print("\n[7] Demonstrating Post-Processing...")

    # Simple decoding strategy: Take the span with the highest start+end score
    # In a real solution, we would aggregate across sliding windows per example_id

    sample_idx = 0
    feature_meta = test_features[sample_idx]

    start_logits = start_preds[sample_idx]
    end_logits = end_preds[sample_idx]

    # Get best start and end
    start_idx = np.argmax(start_logits)
    end_idx = np.argmax(end_logits)

    # Ensure end >= start
    if end_idx < start_idx:
        end_idx = start_idx

    # Decode
    offset_mapping = feature_meta["offset_mapping"]

    # Check if indices are valid context tokens (not None)
    if offset_mapping[start_idx] is None or offset_mapping[end_idx] is None:
        pred_string = ""
    else:
        # Map token indices to character positions in original context
        start_char = offset_mapping[start_idx][0]
        end_char = offset_mapping[end_idx][1]
        pred_string = feature_meta["context"][start_char:end_char]

    print(f"    Sample ID: {feature_meta['example_id']}")
    print(f"    Predicted Span Indices: {start_idx} -> {end_idx}")
    print(f"    Predicted String: '{pred_string}'")

    # 8. Metric Verification
    # -------------------------------------------------------------------------
    print("\n[8] Verifying Metric Function...")

    s1 = "India is a country"
    s2 = "India is country"
    score = jaccard(s1, s2)

    print(f"    Jaccard('{s1}', '{s2}') = {score:.4f}")

    # Manual calculation:
    # set1 = {india, is, a, country} (4)
    # set2 = {india, is, country} (3)
    # intersection = {india, is, country} (3)
    # union = 4 + 3 - 3 = 4
    # score = 3/4 = 0.75
    assert abs(score - 0.75) < 1e-6, "Jaccard calculation incorrect"

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
