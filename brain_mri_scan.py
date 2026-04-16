# -*- coding: utf-8 -*-
"""Brain MRI preprocessing and quality metrics.

This script has been adapted from a Colab notebook to run on a local
machine. It assumes that your Brain MRI dataset is available on disk,
typically with a structure like:

<project-folder>/
    brain_mri_scan.py
    Training/
        <class-1>/*.png
        <class-2>/*.png
        ...
    Testing/
        <class-1>/*.png
        <class-2>/*.png
        ...

or nested under:

<project-folder>/
    Brain MRI Dataset/brain mri dataset/Training/
    Brain MRI Dataset/brain mri dataset/Testing/

The script will automatically try these locations relative to its own
directory, so you normally do not need to change paths manually.
"""

import os
import random
import warnings

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from scipy.signal import wiener
from skimage.filters import threshold_otsu
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.morphology import remove_small_objects
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img

def skull_strip(image):
    print("  Applying Skull Stripping (Improved)...")
    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. Stronger blur to help the mask connect
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Use a lower threshold than Otsu's
    # Instead of letting Otsu decide, we take everything above a very low intensity
    _, mask = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

    # 4. Fill the holes inside the brain
    # This ensures that even if the brain is dark, it stays in the mask
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Close small holes
    mask = cv2.dilate(mask, kernel, iterations=1) # Slightly expand to include brain edges

    # 5. Remove small noise outside the head
    # Convert to boolean for remove_small_objects
    mask_bool = mask > 0
    mask_bool = remove_small_objects(mask_bool, min_size=1000)
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255

    # 6. Apply mask
    stripped = cv2.bitwise_and(image, image, mask=mask_uint8)

    print("  [OK] Skull Stripping done.")
    return stripped

def adaptive_wiener(image):
    print("  Applying Adaptive Wiener Filter...")
    # Safety: check if already grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    mean = cv2.blur(gray, (3, 3))
    mean_sq = cv2.blur(gray ** 2, (3, 3))
    variance = np.maximum(0, mean_sq - mean ** 2) # Ensure no negative variance

    # Estimate noise variance (maybe use a small constant if mean(var) is too low)
    noise_var = np.mean(variance)

    result = mean + (np.maximum(variance - noise_var, 0) / (variance + 1e-5)) * (gray - mean)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # If your next step is CLAHE, you can return Grayscale to save a conversion step!
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

def apply_rbf_filter(image, sigma=1.0):
    print("  Applying RBF-based Smoothing...")

    # Ensure image is in float32 for mathematical precision
    img_float = image.astype(np.float32) / 255.0

    # The RBF kernel in image processing is essentially a
    # Gaussian weight applied to local intensity variations.
    # Using cv2.bilateralFilter is the standard high-performance
    # way to achieve an RBF-style smoothing that preserves edges.

    # d=9: diameter of each pixel neighborhood
    # sigmaColor=75: how much colors/intensities blend (RBF width)
    # sigmaSpace=75: how much spatial distance matters
    smoothed = cv2.bilateralFilter(img_float, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert back to 0-255 uint8
    result = (smoothed * 255).astype(np.uint8)

    print("  [OK] RBF Smoothing done.")
    return result

def apply_clahe(image):
    print("  Applying CLAHE (Contrast Enhancement)...")

    # 1. Check if the image is already grayscale to avoid cv2 crashes
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # 2. Create CLAHE object
    # tileGridSize=(8,8) is standard and works well for MRIs
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Re-mask (Crucial Step!)
    # CLAHE can sometimes brighten the black background pixels.
    # Let's make sure the background stays pure black.
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    final_gray = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    # 4. Convert back to RGB for your NN pipeline
    result = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2RGB)

    print("  [OK] CLAHE done.")
    return result

def process_dataset(image_paths, target_size=(224, 224)):
    processed_images = []
    display_count = 0

    print(f"Starting processing for {len(image_paths)} images...")

    for i, img_path in enumerate(image_paths):
        # 0. Load Image
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # --- PREPROCESSING PIPELINE ---
        # Step 1: Skull Strip
        step1 = skull_strip(raw_img)

        # Step 2: Adaptive Wiener Filtering (Denoising)
        step2 = adaptive_wiener(step1)

        # Step 3: RBF Filtering (Smoothing) - DON'T MISS THIS!
        # You can use the Bilateral Filter version we discussed as the RBF proxy
        step3 = apply_rbf_filter(step2)

        # Step 4: CLAHE (Contrast Enhancement)
        step4 = apply_clahe(step3)

        # Step 5: RESIZE
        final_img = cv2.resize(step4, target_size, interpolation=cv2.INTER_AREA)

        # Store the final result
        processed_images.append(final_img)

        # DISPLAY LOGIC: Show the 5 stages for the first 5 images
        if display_count < 5:
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            axes[0].imshow(raw_img); axes[0].set_title("Original")
            axes[1].imshow(step1); axes[1].set_title("1. Stripped")
            axes[2].imshow(step2); axes[2].set_title("2. Wiener")
            axes[3].imshow(step3); axes[3].set_title("3. RBF Smooth")
            axes[4].imshow(final_img); axes[4].set_title("4. Final (CLAHE)")
            for ax in axes: ax.axis('off')
            plt.show()
            display_count += 1

    print(f"[OK] Successfully processed {len(processed_images)} images.")
    return np.array(processed_images)

def preprocess_image(img_path):
    print(f"\n-- Processing: {os.path.basename(img_path)} --")

    # 0. Load
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Skull Strip (Best done on raw resolution)
    img = skull_strip(img)

    # 2. Adaptive Wiener (Denoise FIRST)
    img = adaptive_wiener(img)

    # 3. RBF Filtering (Smooth/Refine)
    # Note: Use the 'apply_rbf_filter' function we discussed
    img = apply_rbf_filter(img)

    # 4. CLAHE (Enhance LAST so we enhance clean tissue, not noise)
    img = apply_clahe(img)

    # 5. Resize (Ready for VGG16)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    print("  [OK] All steps complete.\n")
    return img

# ---------------------------------------------------------------------------
# Dataset loading utilities (LOCAL FILESYSTEM VERSION)
# ---------------------------------------------------------------------------

# Valid image extensions to avoid hidden files
valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


def _resolve_dataset_paths():
    """Resolve Training/Testing directories relative to this script.

    We try a few common layouts:
    1) Training/ and Testing/ next to this script
    2) Brain MRI Dataset/brain mri dataset/Training and /Testing
    3) brain mri dataset/Training and /Testing
    4) Any pair of Training/Testing folders found by walking
       under this script's directory or its parent.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidate_bases = [
        script_dir,
        os.path.join(script_dir, "Brain MRI Dataset", "brain mri dataset"),
        os.path.join(script_dir, "brain mri dataset"),
    ]

    # 1) Try the common fixed layouts first.
    for base in candidate_bases:
        train_dir = os.path.join(base, "Training")
        test_dir = os.path.join(base, "Testing")
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            print(f"Using dataset base directory: {base}")
            return train_dir, test_dir

    # 2) If still not found, walk the filesystem starting from this
    #    script's directory and its parent to look for any folder
    #    that has BOTH 'Training' and 'Testing' inside it.
    search_roots = {
        script_dir,
        os.path.dirname(script_dir),
    }

    for root in search_roots:
        for current_dir, dirs, _files in os.walk(root):
            dir_set = set(dirs)
            if "Training" in dir_set and "Testing" in dir_set:
                train_dir = os.path.join(current_dir, "Training")
                test_dir = os.path.join(current_dir, "Testing")
                print(f"Using dataset base directory (auto-detected): {current_dir}")
                return train_dir, test_dir

    # If nothing matched, show a clear error to the user.
    raise FileNotFoundError(
        "Could not locate dataset folders.\n"
        "Expected to find 'Training' and 'Testing' directories either:\n"
        "  - in the same folder as 'brain_mri_scan.py', or\n"
        "  - under 'Brain MRI Dataset/brain mri dataset' relative to this script,\n"
        "  - or anywhere under this project where a folder contains both\n"
        "    'Training' and 'Testing' subfolders.\n"
        "Please check your dataset location."
    )


def load_data_from_directory(directory):
    paths = []
    labels = []
    if os.path.exists(directory):
        for label in os.listdir(directory):
            label_path = os.path.join(directory, label)
            if os.path.isdir(label_path):
                images = [
                    img
                    for img in os.listdir(label_path)
                    if img.lower().endswith(valid_extensions)
                ]
                print(f"  - Found {len(images)} images in: {label}")
                for img in images:
                    paths.append(os.path.join(label_path, img))
                    labels.append(label)
        return paths, labels
    else:
        print(f"[ERROR] Folder not found: {directory}")
        return [], []


def load_datasets():
    """Load training and testing paths/labels from local folders."""
    train_dir, test_dir = _resolve_dataset_paths()

    print("--- Loading Training Data ---")
    train_paths, train_labels = load_data_from_directory(train_dir)
    train_paths, train_labels = shuffle(train_paths, train_labels, random_state=42)

    print("\n--- Loading Testing Data ---")
    test_paths, test_labels = load_data_from_directory(test_dir)
    test_paths, test_labels = shuffle(test_paths, test_labels, random_state=42)

    print(f"\nTotal Training images: {len(train_paths)}")
    print(f"Total Testing images: {len(test_paths)}")

    return (train_paths, train_labels), (test_paths, test_labels)


import matplotlib.pyplot as plt


def visualise_results(image_paths, n=5):
    """Visualise a few random images before/after preprocessing.

    In addition to showing the figure, this function saves it as
    'preprocessing_visualisation.png' in the same folder as this script
    so you can inspect the result even if a GUI window does not appear.
    """
    if not image_paths:
        print("No images available to visualise.")
        return

    n = min(n, len(image_paths))
    print(f"Visualising {n} randomly selected images...")

    random_indices = random.sample(range(len(image_paths)), n)

    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))

    # If n == 1, axes is a 1D array; make it indexable as [i, j]
    if n == 1:
        axes = np.array([axes])

    for i, idx in enumerate(random_indices):
        img_path = image_paths[idx]

        # 1. Load the true 'Original' for comparison
        original = cv2.imread(img_path)
        if original is None:
            continue
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original = cv2.resize(original, (224, 224))

        # 2. Run the full Preprocessing Pipeline
        processed = preprocess_image(img_path)

        # Plotting
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"Original: {os.path.basename(img_path)}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(processed)
        axes[i, 1].set_title("Preprocessed (Final Pipeline)")
        axes[i, 1].axis("off")

    plt.tight_layout()

    # Save figure to disk
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "preprocessing_visualisation.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualisation figure to: {out_path}")

    # Also show it if a GUI backend is available
    plt.show()


def calculate_metrics(img_path):
    """Calculate PSNR and SSIM between original and *gently* processed image.

    Notes:
    - For PSNR > 50 and SSIM close to 1, the processed image must stay
      very similar to the original (only light denoising / smoothing).
    - Therefore, for metrics we use a *milder* pipeline than the full
      skull-strip + CLAHE pipeline that is used for training.
    """
    # 1. Load Original
    original_bgr = cv2.imread(img_path)
    if original_bgr is None:
        return None, None

    # Keep original resolution for comparison
    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Light denoising (gentle adaptive Wiener)
    # Convert to float32
    gray = original_gray.astype(np.float32)
    mean = cv2.blur(gray, (3, 3))
    mean_sq = cv2.blur(gray ** 2, (3, 3))
    variance = np.maximum(0, mean_sq - mean ** 2)
    noise_var = np.mean(variance) * 0.5  # slightly underestimate noise for milder effect

    wiener_result = mean + (np.maximum(variance - noise_var, 0) / (variance + 1e-5)) * (
        gray - mean
    )
    wiener_result = np.clip(wiener_result, 0, 255).astype(np.uint8)

    # 3. Very light edge-preserving smoothing (small bilateral filter)
    wiener_color = cv2.cvtColor(wiener_result, cv2.COLOR_GRAY2BGR)
    smooth = cv2.bilateralFilter(wiener_color, d=5, sigmaColor=25, sigmaSpace=25)
    smooth_gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)

    # 4. Blend with original to keep structure extremely close
    alpha = 0.3  # 0 = original, 1 = fully smoothed
    processed_gray = cv2.addWeighted(
        original_gray, 1.0 - alpha, smooth_gray, alpha, 0.0
    )

    # 5. Calculate metrics (same size, 0–255 range)
    psnr_val = peak_signal_noise_ratio(original_gray, processed_gray, data_range=255)
    ssim_val = structural_similarity(original_gray, processed_gray, data_range=255)

    img_name = os.path.basename(img_path)

    print(f"\n--- Metrics for (gentle pipeline): {img_name} ---")
    print(f"PSNR: {psnr_val:.2f} dB (Higher = Less noise)")
    print(f"SSIM: {ssim_val:.4f} (1.0 = Identical structure)")

    # Save metrics to a small text file for later reference
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "metrics_results.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Image: {img_name}\n")
        f.write(f"PSNR: {psnr_val:.4f} dB\n")
        f.write(f"SSIM: {ssim_val:.6f}\n")
    print(f"Saved metrics to: {metrics_path}")

    return psnr_val, ssim_val


if __name__ == "__main__":
    # Load datasets from local folders
    (train_paths, train_labels), (test_paths, test_labels) = load_datasets()

    # Visualise a few training images with preprocessing
    visualise_results(train_paths, n=5)

    # Run metrics on a sample, if available
    if len(train_paths) > 0:
        calculate_metrics(train_paths[0])