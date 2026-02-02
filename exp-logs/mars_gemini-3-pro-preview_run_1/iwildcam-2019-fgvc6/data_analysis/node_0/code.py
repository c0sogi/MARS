import os
import pandas as pd
import numpy as np
import cv2
import random
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train_meta.csv"
SAMPLE_SIZE = 5000  # Number of images to sample for pixel/dimension analysis
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_image_stats(file_info):
    """
    Reads an image and returns its dimensions and pixel statistics.
    """
    idx, row = file_info
    full_path = os.path.join(INPUT_DIR, row["file_path"])

    try:
        # Read image
        img = cv2.imread(full_path)
        if img is None:
            return None

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, channels = img.shape

        # Normalize to [0, 1] for stats calculation
        img_norm = img / 255.0

        # Calculate mean and std per channel
        # axis=(0, 1) computes over height and width
        mean_ch = np.mean(img_norm, axis=(0, 1))
        std_ch = np.std(img_norm, axis=(0, 1))

        return {
            "Id": row["Id"],
            "Category": row["Category"],
            "Width": width,
            "Height": height,
            "AspectRatio": width / height if height > 0 else 0,
            "Channels": channels,
            "Mean_R": mean_ch[0],
            "Mean_G": mean_ch[1],
            "Mean_B": mean_ch[2],
            "Std_R": std_ch[0],
            "Std_G": std_ch[1],
            "Std_B": std_ch[2],
        }
    except Exception:
        return None


def main():
    set_seed(SEED)

    # 1. Load Data
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file {METADATA_FILE} not found.")
        return

    df = pd.read_csv(METADATA_FILE)

    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("================================")

    # 2. Target Variable Analysis
    print("\nTARGET VARIABLE ANALYSIS")
    print("------------------------")

    class_counts = df["Category"].value_counts().sort_index()
    total_samples = len(df)

    print(f"Total Training Samples: {total_samples}")
    print(f"Number of Classes: {len(class_counts)}")

    # Class Balance
    print("\nClass Distribution (Top 5):")
    print(f"{'Class ID':<10} {'Count':<10} {'Percentage':<10}")

    sorted_counts = df["Category"].value_counts(ascending=False)
    for cat, count in sorted_counts.head(5).items():
        print(f"{cat:<10} {count:<10} {count/total_samples*100:.4f}%")

    print("\nClass Distribution (Bottom 5):")
    print(f"{'Class ID':<10} {'Count':<10} {'Percentage':<10}")
    for cat, count in sorted_counts.tail(5).items():
        print(f"{cat:<10} {count:<10} {count/total_samples*100:.4f}%")

    # Imbalance Ratio
    max_class_count = class_counts.max()
    min_class_count = class_counts.min()
    imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else 0
    print(f"\nImbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # 3. Input Data Analysis (Image Modality)
    print("\nINPUT DATA ANALYSIS (IMAGE MODALITY)")
    print("------------------------------------")

    # Stratified Sampling
    # We want to sample images to analyze dimensions and pixels without reading all 160k
    # We try to take a stratified sample to ensure all classes are represented in stats

    # Determine samples per class
    samples_per_class = max(1, SAMPLE_SIZE // len(class_counts))

    sampled_indices = []
    for cat in class_counts.index:
        cat_indices = df[df["Category"] == cat].index.tolist()
        # If class has fewer samples than desired, take all, otherwise sample
        n_sample = min(len(cat_indices), samples_per_class)
        # If the class is very large (like empty), we might want more representation,
        # but for pixel stats, uniform class sampling helps avoid bias towards 'empty' background stats
        # However, to get 'global' dataset stats, we should sample proportionally.
        # Let's do proportional sampling for global stats.

    # Proportional sampling
    sampled_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)

    print(
        f"Analyzing a random subset of {len(sampled_df)} images for pixel statistics..."
    )

    # Process images in parallel
    image_stats_list = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(get_image_stats, sampled_df.iterrows())
        for res in results:
            if res is not None:
                image_stats_list.append(res)

    df_img_stats = pd.DataFrame(image_stats_list)

    if df_img_stats.empty:
        print("Error: No images could be processed.")
        return

    # Dimensions
    print("\nImage Dimensions:")
    print(
        f"Width  - Mean: {df_img_stats['Width'].mean():.4f}, Std: {df_img_stats['Width'].std():.4f}, Min: {df_img_stats['Width'].min()}, Max: {df_img_stats['Width'].max()}"
    )
    print(
        f"Height - Mean: {df_img_stats['Height'].mean():.4f}, Std: {df_img_stats['Height'].std():.4f}, Min: {df_img_stats['Height'].min()}, Max: {df_img_stats['Height'].max()}"
    )

    # Aspect Ratios
    print(
        f"Aspect Ratio - Mean: {df_img_stats['AspectRatio'].mean():.4f}, Std: {df_img_stats['AspectRatio'].std():.4f}"
    )

    # Channels
    channel_counts = df_img_stats["Channels"].value_counts()
    print("\nChannel Distribution:")
    for ch, count in channel_counts.items():
        print(f"Channels {ch}: {count} ({count/len(df_img_stats)*100:.2f}%)")

    # Pixel Stats (Global)
    # Averaging the per-image means/stds gives an approximation of the global mean/std
    global_mean_r = df_img_stats["Mean_R"].mean()
    global_mean_g = df_img_stats["Mean_G"].mean()
    global_mean_b = df_img_stats["Mean_B"].mean()

    # Note: Correct global std requires aggregating sum of squares, but average of stds is a common approximation for EDA
    # For more precision, we can use the pooled variance formula.
    # Var_pooled = Mean(Var_i) + Var(Mean_i)

    var_r = (df_img_stats["Std_R"] ** 2).mean() + df_img_stats["Mean_R"].var()
    var_g = (df_img_stats["Std_G"] ** 2).mean() + df_img_stats["Mean_G"].var()
    var_b = (df_img_stats["Std_B"] ** 2).mean() + df_img_stats["Mean_B"].var()

    global_std_r = np.sqrt(var_r)
    global_std_g = np.sqrt(var_g)
    global_std_b = np.sqrt(var_b)

    print("\nGlobal Pixel Statistics (Normalized [0, 1]):")
    print(
        f"Mean - R: {global_mean_r:.4f}, G: {global_mean_g:.4f}, B: {global_mean_b:.4f}"
    )
    print(f"Std  - R: {global_std_r:.4f}, G: {global_std_g:.4f}, B: {global_std_b:.4f}")

    # 4. Feature/Signal Relationships
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("----------------------------")

    # Relationship between Metadata (Image Size) and Target
    # Do certain classes have distinct image sizes?
    # We use the sampled dataframe for this.

    # Check if there is variance in image size
    unique_widths = df_img_stats["Width"].nunique()
    unique_heights = df_img_stats["Height"].nunique()

    if unique_widths > 1 or unique_heights > 1:
        print("\nImage Size vs Category Analysis:")
        # Group by category and get mean area
        df_img_stats["Area"] = df_img_stats["Width"] * df_img_stats["Height"]
        area_by_cat = df_img_stats.groupby("Category")["Area"].mean()

        # Check correlation between Area and Class ID (just as a proxy for structured relationship)
        # A better metric is ANOVA or just reporting the range of means
        min_area_cat = area_by_cat.idxmin()
        max_area_cat = area_by_cat.idxmax()

        print(
            f"Class with Smallest Avg Image Area: {min_area_cat} ({area_by_cat[min_area_cat]:.0f} pixels)"
        )
        print(
            f"Class with Largest Avg Image Area:  {max_area_cat} ({area_by_cat[max_area_cat]:.0f} pixels)"
        )
    else:
        print("\nImage Size vs Category Analysis:")
        print(
            "All sampled images have identical dimensions. No relationship to analyze."
        )

    # Relationship between Location and Target
    # Check if locations are exclusive to certain animals or diverse
    if "location" in df.columns:
        print("\nLocation vs Category Analysis:")
        loc_counts = df.groupby("location")["Category"].nunique()
        print(f"Average number of unique classes per location: {loc_counts.mean():.4f}")
        print(f"Location with most diversity: {loc_counts.max()} classes")
        print(f"Location with least diversity: {loc_counts.min()} classes")

        # Check if 'Empty' (0) is dominant across all locations
        loc_empty_ratio = df.groupby("location")["Category"].apply(
            lambda x: (x == 0).mean()
        )
        print(
            f"Average proportion of 'Empty' (Class 0) per location: {loc_empty_ratio.mean():.4f}"
        )


if __name__ == "__main__":
    main()
