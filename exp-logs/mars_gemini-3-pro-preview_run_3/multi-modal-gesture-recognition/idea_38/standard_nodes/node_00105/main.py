import os
import sys
import random
import numpy as np
import torch
import pandas as pd
import scipy.stats

# Import provided library modules
from library.config import Config
from library.trainer import Trainer
from library.predict import generate_submission
from library.utils import decode_predictions, calculate_levenshtein_distance


# ==========================================
# Setup & Configuration
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Override Config for Fast Baseline Execution
Config.NUM_EPOCHS = 20  # Cite solution_lesson_node_00055: Ensure full convergence
Config.BATCH_SIZE = 32
# Ensure we use the correct working directory
os.makedirs(Config.WORKING_DIR, exist_ok=True)


# ==========================================
# Failure Analysis
# ==========================================
def run_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set to find error correlations.
    """
    print("\n=== Failure Analysis ===")
    model.eval()
    dataset = val_loader.dataset

    # 1. Reconstruct Global Probabilities (Sliding Window Aggregation)
    total_frames = dataset.all_labels.shape[0]
    global_probs = np.zeros((total_frames, Config.NUM_CLASSES), dtype=np.float32)
    global_counts = np.zeros((total_frames,), dtype=np.float32)

    with torch.no_grad():
        for batch_idx, (features, _) in enumerate(val_loader):
            features = features.to(device)
            outputs = model(features)
            # Use Stage 3 probabilities
            probs = outputs["probs3"].cpu().numpy()

            start_window_idx = batch_idx * val_loader.batch_size
            for i in range(features.size(0)):
                window_idx = start_window_idx + i
                if window_idx < len(dataset.windows):
                    start_frame, end_frame = dataset.windows[window_idx]

                    # Handle potential padding at edges
                    valid_len = end_frame - start_frame
                    pred_len = probs.shape[1]
                    actual_len = min(valid_len, pred_len)

                    global_probs[start_frame : start_frame + actual_len] += probs[
                        i, :actual_len
                    ]
                    global_counts[start_frame : start_frame + actual_len] += 1.0

    # Normalize
    mask = global_counts > 0
    global_probs[mask] /= global_counts[mask, None]

    # 2. Compute Metrics & Extract Features per Sample
    errors = []
    durations = []
    avg_motion = []
    avg_audio = []

    for start, end in dataset.sample_indices:
        # --- Prediction ---
        sample_probs = global_probs[start:end]
        pred_seq = decode_predictions(sample_probs)

        # --- Ground Truth ---
        gt_labels = dataset.all_labels[start:end]
        gt_seq = decode_predictions(gt_labels)  # Works for 1D array too

        # --- Metric (Levenshtein Distance) ---
        dist = calculate_levenshtein_distance(pred_seq, gt_seq)
        errors.append(dist)

        # --- Features ---
        # Duration
        durations.append(end - start)

        # Motion Energy (Mean displacement of joints)
        # dataset.all_skeleton: (TotalFrames, 20, 3)
        if hasattr(dataset, "all_skeleton") and dataset.all_skeleton is not None:
            skel_segment = dataset.all_skeleton[start:end]
            if len(skel_segment) > 1:
                # Frame-to-frame displacement
                diff = skel_segment[1:] - skel_segment[:-1]
                # Norm per joint, then mean over joints and time
                disp = np.linalg.norm(diff, axis=2).mean()
            else:
                disp = 0.0
            avg_motion.append(disp)
        else:
            avg_motion.append(0.0)

        # Audio Energy (Mean MFCC value)
        # dataset.all_audio: (TotalFrames, 13)
        if hasattr(dataset, "all_audio") and dataset.all_audio is not None:
            audio_segment = dataset.all_audio[start:end]
            avg_audio.append(np.mean(audio_segment))
        else:
            avg_audio.append(0.0)

    # 3. Compute Correlations
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "duration": durations,
            "motion": avg_motion,
            "audio": avg_audio,
        }
    )

    # Pearson Correlation
    corr_duration = df_analysis["error"].corr(df_analysis["duration"])
    corr_motion = df_analysis["error"].corr(df_analysis["motion"])
    corr_audio = df_analysis["error"].corr(df_analysis["audio"])

    print("Correlation with Error Magnitude:")
    print(f"  Duration: {corr_duration:.4f}")
    print(f"  Motion Energy: {corr_motion:.4f}")
    print(f"  Audio Energy: {corr_audio:.4f}")


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Initialization
    set_seed(Config.SEED)
    print(
        f"Initializing Trainer (Epochs: {Config.NUM_EPOCHS}, Batch: {Config.BATCH_SIZE})..."
    )

    trainer = Trainer(Config)

    # 2. Training Loop
    print("Starting Training...")
    best_val_score = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = trainer.train_epoch(epoch)

        # Validate
        val_score = trainer.validate(trainer.val_loader)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.4f}"
        )

        # Save Best Model
        if val_score < best_val_score:
            best_val_score = val_score
            torch.save(trainer.model.state_dict(), Config.MODEL_SAVE_PATH)

    # 3. Final Evaluation
    print("\nTraining Complete. Loading best model for evaluation...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=trainer.device)
        )

    # Re-compute score on best model to be absolutely sure
    final_score = trainer.validate(trainer.val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_score}")

    # 4. Failure Analysis
    run_failure_analysis(trainer.model, trainer.val_loader, trainer.device)

    # 5. Conditional Submission
    THRESHOLD = 0.2251
    if final_score < THRESHOLD:
        print(
            f"\nValidation score ({final_score:.4f}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        # Use the standalone predict function which handles test data loading and inference
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation score ({final_score:.4f}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )
