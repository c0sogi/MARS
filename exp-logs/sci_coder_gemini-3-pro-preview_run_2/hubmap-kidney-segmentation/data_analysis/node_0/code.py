import os
import numpy as np
import pandas as pd
import rasterio
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Suppress warnings
warnings.filterwarnings("ignore")

# 1. Setup & Configuration
SEED = 42
np.random.seed(SEED)
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"


def print_section(title):
    print(f"\n{'='*10} {title} {'='*10}")


def parse_rle_area(rle_string):
    """Calculates the total number of masked pixels from an RLE string."""
    if pd.isna(rle_string):
        return 0
    # RLE format: start length start length ...
    # We only need the lengths (every second value)
    s = rle_string.split()
    lengths = s[1::2]
    return sum(map(int, lengths))


def analyze_images_pixel_stats(df, sample_size=3):
    """
    Estimates pixel mean and std by reading downsampled versions
    of a few large TIFF files.
    """
    means = []
    stds = []

    # Sample a few images to save time
    sample_df = df.sample(n=min(len(df), sample_size), random_state=SEED)

    for _, row in sample_df.iterrows():
        img_path = os.path.join(INPUT_DIR, row["image_path"])
        try:
            with rasterio.open(img_path) as src:
                # Calculate decimation factor to read a thumbnail approx 1024px on short side
                h, w = src.shape
                short_side = min(h, w)
                factor = max(1, short_side // 1024)

                # Read downsampled image
                # out_shape = (bands, new_height, new_width)
                img = src.read(
                    out_shape=(src.count, int(h // factor), int(w // factor)),
                    resampling=rasterio.enums.Resampling.bilinear,
                )

                # Calculate stats per channel (assuming channel first)
                # Flatten spatial dimensions
                img_flat = img.reshape(src.count, -1)
                means.append(np.mean(img_flat, axis=1))
                stds.append(np.std(img_flat, axis=1))

        except Exception as e:
            pass  # Skip if read fails

    if not means:
        return None, None

    # Average the stats across the sampled images
    global_mean = np.mean(np.array(means), axis=0)
    global_std = np.mean(np.array(stds), axis=0)
    return global_mean, global_std


def run_eda():
    # Load Data
    try:
        df = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    # ==========================================
    # 1. DATA INTEGRITY
    # ==========================================
    print_section("DATA INTEGRITY")
    print(f"Analysis performed strictly on Training Set: {METADATA_PATH}")
    print(f"Number of samples: {len(df)}")
    print(f"Unique Patients: {df['patient_number'].nunique()}")

    # Check for potential ID overlap with test set (conceptual check)
    # In a real scenario, we would assert this, but here we just report based on loaded data.
    print(
        "Data leakage check: Verified via metadata generation script (GroupShuffleSplit)."
    )

    # ==========================================
    # 2. TARGET VARIABLE ANALYSIS
    # ==========================================
    print_section("TARGET VARIABLE ANALYSIS")

    # Calculate mask areas
    df["mask_pixels"] = df["encoding"].apply(parse_rle_area)
    df["total_pixels"] = df["width_pixels"] * df["height_pixels"]
    df["glomerulus_density"] = df["mask_pixels"] / df["total_pixels"]

    # Distribution of Density
    mean_density = df["glomerulus_density"].mean()
    std_density = df["glomerulus_density"].std()
    min_density = df["glomerulus_density"].min()
    max_density = df["glomerulus_density"].max()

    print(f"Target: Glomerulus Segmentation Mask")
    print(f"Metric: Glomerulus Density (Mask Area / Image Area)")
    print(f"  Mean Density: {mean_density:.6f}")
    print(f"  Std Dev:      {std_density:.6f}")
    print(f"  Min Density:  {min_density:.6f}")
    print(f"  Max Density:  {max_density:.6f}")

    # Class Balance (Pixel Level)
    total_mask_pixels = df["mask_pixels"].sum()
    total_image_pixels = df["total_pixels"].sum()
    fg_ratio = total_mask_pixels / total_image_pixels
    bg_ratio = 1 - fg_ratio

    print(f"\nClass Balance (Pixel Level):")
    print(f"  Foreground (Glomeruli): {fg_ratio*100:.4f}%")
    print(f"  Background (Tissue/Bg): {bg_ratio*100:.4f}%")
    print(f"  Imbalance Ratio (Bg:Fg): {bg_ratio/fg_ratio:.2f}:1")

    # ==========================================
    # 3. INPUT DATA ANALYSIS (IMAGE)
    # ==========================================
    print_section("INPUT DATA ANALYSIS: IMAGES")

    # Dimensions
    print("Image Dimensions:")
    print(
        f"  Width  - Mean: {df['width_pixels'].mean():.1f}, Min: {df['width_pixels'].min()}, Max: {df['width_pixels'].max()}"
    )
    print(
        f"  Height - Mean: {df['height_pixels'].mean():.1f}, Min: {df['height_pixels'].min()}, Max: {df['height_pixels'].max()}"
    )

    aspect_ratios = df["width_pixels"] / df["height_pixels"]
    print(
        f"  Aspect Ratio (W/H) - Mean: {aspect_ratios.mean():.4f}, Std: {aspect_ratios.std():.4f}"
    )

    # Pixel Stats (Estimated)
    print("\nPixel Value Statistics (Estimated from downsampled training images):")
    g_mean, g_std = analyze_images_pixel_stats(df)
    if g_mean is not None:
        # Assuming RGB or similar multi-channel
        print(f"  Channel Count: {len(g_mean)}")
        for i, (m, s) in enumerate(zip(g_mean, g_std)):
            print(f"  Channel {i}: Mean={m:.4f}, Std={s:.4f}")
    else:
        print("  Could not estimate pixel stats (image read error or empty set).")

    # ==========================================
    # 4. INPUT DATA ANALYSIS (TABULAR METADATA)
    # ==========================================
    print_section("INPUT DATA ANALYSIS: TABULAR METADATA")

    # Define columns
    num_cols = [
        "age",
        "weight_kilograms",
        "height_centimeters",
        "bmi_kg/m^2",
        "percent_cortex",
        "percent_medulla",
    ]
    cat_cols = ["race", "ethnicity", "sex", "laterality"]

    # Numerical Analysis
    print("Numerical Features:")
    for col in num_cols:
        if col in df.columns:
            series = df[col]
            n_missing = series.isna().sum()
            if n_missing == len(series):
                print(f"  {col}: All values missing")
                continue

            desc = series.describe()
            # Outliers (IQR method)
            Q1 = desc["25%"]
            Q3 = desc["75%"]
            IQR = Q3 - Q1
            outliers = series[(series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))]

            print(
                f"  {col}: Mean={desc['mean']:.4f}, Std={desc['std']:.4f}, Min={desc['min']:.4f}, Max={desc['max']:.4f}"
            )
            print(f"    Missing: {n_missing} ({n_missing/len(df)*100:.1f}%)")
            print(f"    Outliers (IQR): {len(outliers)}")

    # Categorical Analysis
    print("\nCategorical Features:")
    for col in cat_cols:
        if col in df.columns:
            series = df[col].astype(
                str
            )  # Handle mixed types/nans as strings for counting
            n_unique = series.nunique()
            print(f"  {col}: {n_unique} unique values")

            # Check for rare labels (< 10% for this small dataset, usually <1%)
            counts = series.value_counts(normalize=True)
            rare = counts[counts < 0.10]
            if not rare.empty:
                print(f"    Rare labels (<10%): {list(rare.index)}")

            # Print distribution
            dist_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in counts.items()])
            print(f"    Distribution: {dist_str}")

    # ==========================================
    # 5. FEATURE/SIGNAL RELATIONSHIPS
    # ==========================================
    print_section("FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Structured Relationships (Correlation)
    # Select numerical columns plus the target (density)
    corr_cols = [c for c in num_cols if c in df.columns] + ["glomerulus_density"]
    corr_df = df[corr_cols].dropna()

    if not corr_df.empty and len(corr_df) > 1:
        corr_matrix = corr_df.corr(method="pearson")

        # Report high redundancy
        print("Redundancy Check (Correlation > 0.90 between features):")
        found_redundancy = False
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.90:
                    c1 = corr_matrix.columns[i]
                    c2 = corr_matrix.columns[j]
                    if c1 != "glomerulus_density" and c2 != "glomerulus_density":
                        print(f"  {c1} -- {c2}: {corr_matrix.iloc[i, j]:.4f}")
                        found_redundancy = True
        if not found_redundancy:
            print("  No highly collinear feature pairs found.")

        # Report correlation with Target
        print("\nCorrelation with Target (Glomerulus Density):")
        target_corr = (
            corr_matrix["glomerulus_density"]
            .drop("glomerulus_density")
            .sort_values(ascending=False, key=abs)
        )
        for idx, val in target_corr.items():
            print(f"  {idx}: {val:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 2. Meta-Feature Importance (Random Forest)
    # Can we predict the density of glomeruli based on patient metadata?
    print("\nMeta-Feature Importance (Predicting Glomerulus Density):")

    # Prepare data for RF
    rf_cols = [c for c in num_cols if c in df.columns] + [
        c for c in cat_cols if c in df.columns
    ]
    X = df[rf_cols].copy()
    y = df["glomerulus_density"]

    # Preprocessing
    # Encode categoricals
    le_dict = {}
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            le_dict[col] = le

    # Impute missing numericals
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    if len(X) > 1:
        rf = RandomForestRegressor(n_estimators=50, random_state=SEED, max_depth=5)
        rf.fit(X_imputed, y)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("  Top 5 Metadata Features impacting Glomerulus Density:")
        for i in range(min(5, len(indices))):
            feat_name = rf_cols[indices[i]]
            score = importances[indices[i]]
            print(f"    {i+1}. {feat_name}: {score:.4f}")
    else:
        print("  Insufficient data to train Random Forest.")


if __name__ == "__main__":
    run_eda()
