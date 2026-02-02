import os
import shutil
import numpy as np
import pandas as pd
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)

# ==========================================
# 1. Setup & Monkey Patching
# ==========================================
# We need to override the default config to use a temporary directory
# and small hyperparameters for this demonstration.

import library.config as config

# Define demo directories
DEMO_DIR = "./working/demo"
DEMO_INPUT_DIR = os.path.join(DEMO_DIR, "input")
DEMO_WORK_DIR = os.path.join(DEMO_DIR, "cache")
DEMO_SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

os.makedirs(DEMO_INPUT_DIR, exist_ok=True)
os.makedirs(DEMO_WORK_DIR, exist_ok=True)
os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

# Override Config Paths
config.INPUT_DIR = DEMO_INPUT_DIR
config.WORK_DIR = DEMO_WORK_DIR
config.SUBMISSION_DIR = DEMO_SUBMISSION_DIR

config.TRAIN_FILE = os.path.join(DEMO_INPUT_DIR, "dummy_train.csv")
config.VAL_FILE = os.path.join(DEMO_INPUT_DIR, "dummy_val.csv")
config.TEST_FILE = os.path.join(DEMO_INPUT_DIR, "dummy_test.csv")
config.SUBMISSION_FILE = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")

# Override Cache Paths in config based on new WORK_DIR
config.TRAIN_VECTORS_PATH = os.path.join(DEMO_WORK_DIR, "train_vectors.npy")
config.TRAIN_LABELS_PATH = os.path.join(DEMO_WORK_DIR, "train_labels.npy")
config.FAISS_INDEX_PATH = os.path.join(DEMO_WORK_DIR, "knn_index.bin")
config.EMBEDDER_PATH = os.path.join(DEMO_WORK_DIR, "subword_embedder.joblib")
config.TEST_VECTORS_PATH = os.path.join(DEMO_WORK_DIR, "test_vectors.npy")

# Override Hyperparameters for small data
config.EMBEDDING_DIM = 5  # Small dimension for dummy data
config.K_NEIGHBORS = 3  # Small K
config.PLAIN_SAMPLE_RATIO = 1.0  # Keep all for this tiny test

# Now import the rest of the library modules which will use the updated config
from library.normalizers import NormalizationRegistry
from library.data_loader import load_train_data, load_val_data, load_test_data
from library.features import (
    SubwordEmbedder,
    load_or_create_train_features,
    create_test_features,
)
from library.retrieval import KNNClassifier, load_or_train_index

# ==========================================
# 2. Create Dummy Data
# ==========================================
print("Creating dummy datasets...")

# Create a small training set with diverse classes and context
train_data = {
    "sentence_id": [0, 0, 0, 1, 1, 1, 2, 2],
    "token_id": [0, 1, 2, 0, 1, 2, 0, 1],
    "class": ["PLAIN", "DATE", "PUNCT", "MONEY", "PLAIN", "PUNCT", "CARDINAL", "PLAIN"],
    "before": ["The", "2012", ".", "$3.50", "price", ".", "123", "items"],
    "after": [
        "The",
        "twenty twelve",
        ".",
        "three dollars fifty cents",
        "price",
        ".",
        "one hundred twenty-three",
        "items",
    ],
    "id": ["0_0", "0_1", "0_2", "1_0", "1_1", "1_2", "2_0", "2_1"],
}
df_train = pd.DataFrame(train_data)
df_train.to_csv(config.TRAIN_FILE, index=False)

# Create a small validation set
val_data = {
    "sentence_id": [10, 10, 10],
    "token_id": [0, 1, 2],
    "class": ["PLAIN", "CARDINAL", "PLAIN"],
    "before": ["I", "10", "apples"],
    "after": ["I", "ten", "apples"],
    "id": ["10_0", "10_1", "10_2"],
}
df_val = pd.DataFrame(val_data)
df_val.to_csv(config.VAL_FILE, index=False)

# Create a small test set
test_data = {
    "sentence_id": [20, 20, 20],
    "token_id": [0, 1, 2],
    "before": ["It", "cost", "$5"],
    "id": ["20_0", "20_1", "20_2"],
}
df_test = pd.DataFrame(test_data)
df_test.to_csv(config.TEST_FILE, index=False)

print("Dummy data created successfully.\n")


# ==========================================
# 3. Demonstrate Normalizers
# ==========================================
print(">>> Testing NormalizationRegistry...")
registry = NormalizationRegistry()

# Test Case 1: Cardinal
raw = "123"
expected = "one hundred twenty-three"
result = registry.normalize(raw, "CARDINAL")
print(f"  CARDINAL: '{raw}' -> '{result}'")
assert result == expected, f"Expected '{expected}', got '{result}'"

# Test Case 2: Money
raw = "$3.50"
expected = "three dollars fifty cents"
result = registry.normalize(raw, "MONEY")
print(f"  MONEY:    '{raw}' -> '{result}'")
assert result == expected, f"Expected '{expected}', got '{result}'"

# Test Case 3: Date (Year)
raw = "2012"
expected = (
    "twenty twelve"  # Based on logic in normalizers.py for 2010-2099 range or heuristic
)
result = registry.normalize(raw, "DATE")
print(f"  DATE:     '{raw}' -> '{result}'")
assert result == expected, f"Expected '{expected}', got '{result}'"

# Test Case 4: PLAIN (Passthrough)
raw = "Hello"
expected = "Hello"
result = registry.normalize(raw, "PLAIN")
print(f"  PLAIN:    '{raw}' -> '{result}'")
assert result == expected

print("Normalization logic verified.\n")


# ==========================================
# 4. Demonstrate Data Loading & Context
# ==========================================
print(">>> Testing Data Loader...")

# Load train data (force no cache to test logic)
df_train_loaded = load_train_data(load_cached_data=False, downsample_ratio=1.0)

# Verify Context Window Logic
# Sentence 0: "The", "2012", "."
# Token 0 ("The"): prev="", next="2012"
# Token 1 ("2012"): prev="The", next="."
# Token 2 ("."): prev="2012", next=""

row_0 = df_train_loaded[
    (df_train_loaded["sentence_id"] == 0) & (df_train_loaded["token_id"] == 0)
].iloc[0]
row_1 = df_train_loaded[
    (df_train_loaded["sentence_id"] == 0) & (df_train_loaded["token_id"] == 1)
].iloc[0]
row_2 = df_train_loaded[
    (df_train_loaded["sentence_id"] == 0) & (df_train_loaded["token_id"] == 2)
].iloc[0]

print(
    f"  Context Check (Start): '{row_0['before']}' | Prev: '{row_0['prev_token']}' | Next: '{row_0['next_token']}'"
)
assert row_0["prev_token"] == "", "Start of sentence should have empty prev_token"
assert row_0["next_token"] == "2012", "Next token mismatch"

print(
    f"  Context Check (Middle): '{row_1['before']}' | Prev: '{row_1['prev_token']}' | Next: '{row_1['next_token']}'"
)
assert row_1["prev_token"] == "The", "Prev token mismatch"
assert row_1["next_token"] == ".", "Next token mismatch"

print(
    f"  Context Check (End):   '{row_2['before']}' | Prev: '{row_2['prev_token']}' | Next: '{row_2['next_token']}'"
)
assert row_2["next_token"] == "", "End of sentence should have empty next_token"

print("Data loader context logic verified.\n")


# ==========================================
# 5. Demonstrate Feature Extraction
# ==========================================
print(">>> Testing Feature Extraction...")

# We need to monkey-patch the EMBEDDER_PATH in features.py indirectly via config or just pass it if supported.
# The features.py uses config.EMBEDDER_PATH. We updated config, so it should be fine.

# Fit and Transform Training Data
# Note: We use load_cached_data=False to force generation
train_vectors, train_labels, embedder = load_or_create_train_features(
    load_cached_data=False
)

print(f"  Train Vectors Shape: {train_vectors.shape}")
print(f"  Train Labels Shape: {train_labels.shape}")

# Expected shape: (n_samples, embedding_dim * 2) -> (8, 5 * 2) = (8, 10)
assert train_vectors.shape == (
    8,
    10,
), f"Expected shape (8, 10), got {train_vectors.shape}"
assert len(train_labels) == 8

# Transform Test Data
test_vectors = create_test_features(embedder, load_cached_data=False)
print(f"  Test Vectors Shape: {test_vectors.shape}")
assert test_vectors.shape == (
    3,
    10,
), f"Expected shape (3, 10), got {test_vectors.shape}"

print("Feature extraction verified.\n")


# ==========================================
# 6. Demonstrate Retrieval / Classification
# ==========================================
print(">>> Testing KNN Classifier...")

# Train Index
knn = load_or_train_index(train_vectors, train_labels, load_cached_model=False)

# Predict on Test Vectors
preds = knn.predict(test_vectors)
print(f"  Test Predictions: {preds}")
assert len(preds) == 3

# Evaluate on Validation Data
# First, process validation data to get vectors
print("  Processing validation data for evaluation...")
df_val_loaded = load_val_data(load_cached_data=False)
val_vectors = embedder.transform(df_val_loaded)
val_labels = df_val_loaded["class"].values

accuracy = knn.evaluate(val_vectors, val_labels)
print(f"  Validation Accuracy: {accuracy:.4f}")

# Since dummy data is tiny and random, we don't assert high accuracy,
# just that the function runs and returns a float.
assert isinstance(accuracy, float)
assert 0.0 <= accuracy <= 1.0

print("KNN Classifier verified.\n")


# ==========================================
# 7. End-to-End Simulation
# ==========================================
print(">>> Running End-to-End Simulation on Test Sample...")

# Let's take the 3rd token from test set: "$5" (id: 20_2)
# Vectors are already computed in `test_vectors`
sample_idx = 2
sample_token = df_test.iloc[sample_idx]["before"]
sample_vector = test_vectors[sample_idx].reshape(1, -1)

# 1. Predict Class
pred_class = knn.predict(sample_vector)[0]
print(f"  Token: '{sample_token}'")
print(f"  Predicted Class: {pred_class}")

# 2. Normalize
normalized_text = registry.normalize(sample_token, pred_class)
print(f"  Normalized: '{normalized_text}'")

# Check if it makes sense (even if prediction is wrong due to tiny data, logic should hold)
# If it predicted MONEY, it should normalize. If PLAIN, it stays "$5".
if pred_class == "MONEY":
    # $5 -> five dollars
    assert "dollar" in normalized_text
elif pred_class == "PLAIN":
    assert normalized_text == sample_token

print("End-to-End simulation completed.\n")


# ==========================================
# 8. Cleanup
# ==========================================
print("Cleaning up temporary files...")
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
print("Cleanup done.")

print("\nAll demonstrations and verifications passed successfully.")
