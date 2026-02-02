import pandas as pd
import numpy as np
import os
from sklearn.metrics import matthews_corrcoef
from library.training_pipeline import TrainingPipeline
from library.utils import seed_everything
from library.config import ModelConfig


def run():
    # 1. Setup and Reproducibility
    seed_everything(ModelConfig.SEED)
    print("Initializing DSP-EME Pipeline...")
    pipeline = TrainingPipeline()

    # 2. Data Loading
    # Using 50% of data to ensure fast baseline execution within time limits
    # while providing sufficient data for the expert models.
    # FIX: Load FULL dataset first (sample_fraction=None) to preserve Validation integrity.
    # Cite debug_lesson_3: Do Not Apply Training-Specific Data Filtering to Validation Sets.
    print("Loading Full Train/Val Dataset...")
    train_df, val_df = pipeline.loader.prepare_train_val_dataset(
        load_cached=True, sample_fraction=None
    )

    # Manual optimization: Subsample ONLY the training data
    SAMPLE_FRACTION = 0.5
    print(f"Subsampling Training Data to {SAMPLE_FRACTION:.1%}...")
    train_df = train_df.sample(
        frac=SAMPLE_FRACTION, random_state=ModelConfig.SEED
    ).reset_index(drop=True)

    # 3. Stage 1: Train Scout Models
    # Scouts are trained on a balanced subset to learn coarse decision boundaries
    scout_lgbm, scout_xgb = pipeline.train_scouts(train_df)

    # 4. Stage 2: Mine Hard Negatives
    # We disable caching for mining here because we are using a sampled dataset.
    # Cached indices from a full run would be invalid for the sampled dataframe.
    hard_neg_indices = pipeline.mine_hard_negatives(
        train_df, scout_lgbm, scout_xgb, load_cached=False
    )

    # 5. Stage 3: Train Expert Models
    # Experts are trained on Positives + Hard Negatives + Buffer
    expert_lgbm, expert_xgb, best_threshold = pipeline.train_experts(
        train_df, val_df, hard_neg_indices
    )

    # 6. Validation & Metric Calculation
    print("\n--- Validating Model ---")
    X_val, y_val = pipeline.loader.split_features_target(val_df)

    # Inference with Expert Ensemble
    # Note: Models are already fitted. We use the in-memory objects.
    p_val_lgbm = expert_lgbm.predict(X_val)
    p_val_xgb = expert_xgb.predict(X_val)
    p_val_ensemble = (p_val_lgbm + p_val_xgb) / 2.0

    # Apply optimized threshold
    y_pred = (p_val_ensemble >= best_threshold).astype(int)

    # Calculate MCC
    val_mcc = matthews_corrcoef(y_val, y_pred)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_mcc}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude (Absolute difference between Truth and Probability)
    # y_val is binary (0/1), p_val_ensemble is probability (0.0-1.0)
    errors = np.abs(y_val - p_val_ensemble)

    # Create a temporary dataframe for correlation analysis
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlation of features with error magnitude
    # We drop the error column itself from the correlation result
    correlations = analysis_df.corrwith(analysis_df["error_magnitude"]).drop(
        "error_magnitude"
    )

    # Identify top factors associated with errors
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("Top Feature Correlations with Error Magnitude:")
    print(top_correlations)

    # 8. Conditional Submission
    THRESHOLD_SCORE = 0.6782

    if val_mcc > THRESHOLD_SCORE:
        print(
            f"\nValidation metric {val_mcc:.4f} exceeds threshold {THRESHOLD_SCORE}. Generating submission..."
        )
        pipeline.generate_submission(expert_lgbm, expert_xgb, best_threshold)
    else:
        print(
            f"\nValidation metric {val_mcc:.4f} does not exceed {THRESHOLD_SCORE}. Skipping submission generation."
        )


if __name__ == "__main__":
    run()
