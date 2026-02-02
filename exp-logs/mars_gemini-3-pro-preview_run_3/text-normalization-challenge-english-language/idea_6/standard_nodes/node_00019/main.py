import sys
import os
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.utils import set_seed
from library.symbolic_layer import SymbolicMemory
from library.training_agent import Trainer
from library.inference_manager import CascadePredictor
from library.data_factory import DataFactory


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for speed and performance within time limits
    Config.NUM_EPOCHS = 10  # Sufficient for convergence on filtered hard samples
    Config.DEBUG = False

    # Setup directories and seeds
    Config.setup_environment()
    set_seed(Config.SEED)

    print("=== Phase 1: Symbolic Memory Construction ===")
    # Initialize Symbolic Memory
    sym_mem = SymbolicMemory()

    # Load training data to build stats (if not cached)
    # We check if stats exist to avoid loading the large train file if possible
    stats_exist = (
        os.path.exists(sym_mem.trigram_path)
        and os.path.exists(sym_mem.bigram_left_path)
        and os.path.exists(sym_mem.bigram_right_path)
        and os.path.exists(sym_mem.unigram_path)
    )

    if not stats_exist:
        print("Stats not found. Loading training data...")
        if os.path.exists(Config.TRAIN_DATA_PATH):
            df_train_full = pd.read_parquet(Config.TRAIN_DATA_PATH)
            sym_mem.build_stats(df=df_train_full, load_cached_data=False)
            del df_train_full  # Free memory
        else:
            raise FileNotFoundError(
                f"Training data not found at {Config.TRAIN_DATA_PATH}"
            )
    else:
        print("Stats found. Loading from cache...")
        sym_mem.build_stats(load_cached_data=True)

    print("=== Phase 2: Neural Network Training ===")
    # Initialize Trainer
    # Trainer handles data loading, filtering for 'hard' samples, and training
    trainer = Trainer(debug=Config.DEBUG, load_cached_data=True)

    # Train the model
    trainer.fit(epochs=Config.NUM_EPOCHS)

    # Free up trainer memory
    del trainer
    torch.cuda.empty_cache()

    print("=== Phase 3: Validation & Failure Analysis ===")
    # Initialize Predictor (loads model and resources)
    predictor = CascadePredictor()
    predictor._prepare_resources()

    # Load Full Validation Set
    print(f"Loading validation data from {Config.VAL_DATA_PATH}...")
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)

    # Add context (prev/next)
    df_val = predictor.data_factory._add_context(df_val)

    # Ensure types
    df_val["before"] = df_val["before"].astype(str)
    df_val["after"] = df_val["after"].astype(str)
    df_val["prev"] = df_val["prev"].astype(str)
    df_val["next"] = df_val["next"].astype(str)
    if "id" not in df_val.columns:
        # Create id if missing (metadata usually has it)
        df_val["id"] = (
            df_val["sentence_id"].astype(str) + "_" + df_val["token_id"].astype(str)
        )

    # Prepare for Cascade
    total_samples = len(df_val)
    predictions = {}  # Index -> Prediction String
    neural_indices = []

    print("Running Symbolic & Heuristic Layers...")
    # Iterate to apply Symbolic and Heuristic layers
    # Using itertuples for performance
    for row in df_val.itertuples(index=True):
        # 1. Symbolic
        sym_pred = predictor.symbolic_memory.query(row.prev, row.before, row.next)
        if sym_pred is not None:
            predictions[row.Index] = sym_pred
            continue

        # 2. Heuristic (Identity for Alpha OOV)
        if row.before.isalpha():
            predictions[row.Index] = row.before
            continue

        # 3. Neural Candidate
        neural_indices.append(row.Index)

    # Run Neural Inference on the tail
    if neural_indices:
        print(f"Running Neural Inference on {len(neural_indices)} samples...")
        df_neural = df_val.loc[neural_indices].copy()

        # _run_neural_inference returns {id: prediction}
        neural_preds_map = predictor._run_neural_inference(df_neural)

        # Map back to dataframe index
        # We create a mapping from id -> Index for the neural subset
        id_to_index = {row.id: row.Index for row in df_neural.itertuples()}

        for uid, pred in neural_preds_map.items():
            if uid in id_to_index:
                predictions[id_to_index[uid]] = pred

    # Calculate Metrics and Analyze Failures
    print("Calculating metrics...")
    correct_count = 0
    analysis_data = []

    for row in df_val.itertuples(index=True):
        pred = predictions.get(row.Index, "")
        actual = row.after

        if pred == actual:
            correct_count += 1
            is_error = 0
        else:
            is_error = 1

        # Collect data for correlation analysis
        analysis_data.append({"len_before": len(row.before), "is_error": is_error})

    accuracy = correct_count / total_samples
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis: Correlation
    if analysis_data:
        df_analysis = pd.DataFrame(analysis_data)
        if df_analysis["is_error"].sum() > 0:
            corr = df_analysis["is_error"].corr(df_analysis["len_before"])
            print(f"Correlation (Error vs Input Length): {corr}")
        else:
            print("No errors found. Correlation undefined.")

    # ==========================================
    # 4. Submission
    # ==========================================
    THRESHOLD = 0.9943860453286453

    if accuracy > THRESHOLD:
        print(
            f"Accuracy {accuracy} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        # Free memory before submission generation
        del df_val
        del df_analysis
        del predictions
        torch.cuda.empty_cache()

        predictor.generate_submission(load_cached_data=True)
        print("Submission generated successfully.")
    else:
        print(
            f"Accuracy {accuracy} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
