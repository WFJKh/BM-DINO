import os
import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
from tqdm import tqdm


def generate_segmentation_masks(img_dir, mask_dir, plot_dir=None):
    """
    Batch-generate segmentation masks and save them, using the original segmentation algorithms directly
    """
    os.makedirs(mask_dir, exist_ok=True)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(img_files)} images, start processing...")

    for filename in tqdm(img_files, desc="Generating segmentation masks"):
        img_path = os.path.join(img_dir, filename)

        try:
            # Use the original preprocessing method exactly
            img, gray = preprocess_image(img_path)
            if img is None or gray is None:
                continue

            # Use the original embryo-mask generation exactly
            embryo_mask = improved_embryo_mask(gray)

            # Use the original watershed segmentation exactly
            ws_labels = improved_segment_watershed(gray, embryo_mask)

            # Use the original TE-region identification exactly
            te_mask = identify_te_region(ws_labels, gray)

            # Use the original ICM-region identification exactly
            icm_mask = identify_icm_by_subtraction(embryo_mask, te_mask)

            # Create combined mask: 0=background, 1=EXP, 2=TE, 3=ICM
            combined_mask = np.zeros_like(gray, dtype=np.uint8)
            combined_mask[embryo_mask > 0] = 1
            combined_mask[te_mask > 0] = 2
            combined_mask[icm_mask > 0] = 3

            # Save mask (no value scaling, save 0-3 directly)
            mask_filename = os.path.splitext(filename)[0] + "_mask.png"
            mask_path = os.path.join(mask_dir, mask_filename)

            # Save as PNG, preserving original values
            cv2.imwrite(mask_path, combined_mask)

            # Optional: save visualization
            if plot_dir:
                plot_path = os.path.join(plot_dir, os.path.splitext(filename)[0] + "_seg.png")
                save_visualization(img, embryo_mask, te_mask, icm_mask, combined_mask, plot_path)

        except Exception as e:
            print(f"Error processing image {filename}: {str(e)}")

    print(f"All masks saved to: {mask_dir}")


# The following functions are copied exactly from your original segmentation code
def preprocess_image(img_path):
    """Read image and enhance contrast, return RGB and grayscale images"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read image: {img_path}")
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # CLAHE contrast enhancement
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img_rgb, gray


def improved_embryo_mask(gray):
    """Otsu threshold + edge fusion + closing + largest connected component, filters out corner embryos"""
    # Canny edges + Otsu binarization
    edges = cv2.Canny(gray, 50, 150)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = cv2.bitwise_and(thresh, cv2.dilate(edges, np.ones((5, 5), np.uint8)))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    # Fill holes
    mask_bool = binary_fill_holes(closed > 0)
    # Small closing for smoothing
    mask_bool = closing(mask_bool, disk(5))
    # Keep largest connected component
    lab = label(mask_bool)
    if lab.max() > 0:
        regions = regionprops(lab)
        largest_label = max(regions, key=lambda r: r.area).label
        mask_bool = (lab == largest_label)
    return mask_bool.astype(np.uint8) * 255


def improved_segment_watershed(gray, embryo_mask):
    grad = np.uint8(np.abs(cv2.Laplacian(gray, cv2.CV_64F)))
    dist = cv2.distanceTransform(embryo_mask, cv2.DIST_L2, 5)
    local_maxi = peak_local_max(dist, labels=embryo_mask, footprint=np.ones((11, 11)), exclude_border=True)
    markers = np.zeros_like(gray, dtype=np.int32)
    for idx, (y, x) in enumerate(local_maxi):
        markers[y, x] = idx + 1
    ws = watershed(-grad, markers, mask=embryo_mask)
    return ws


def identify_te_region(ws_labels, gray):
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    te_bool = np.zeros_like(gray, dtype=bool)
    for region in regionprops(ws_labels):
        if region.area < 200:
            continue
        coords = (ws_labels == region.label)
        d = np.hypot(region.centroid[1] - cx, region.centroid[0] - cy) / (min(w, h) / 2)
        if d > 0.5 and np.std(gray[coords]) > 6:
            te_bool |= coords
    # Fill holes and smooth
    te_bool = binary_fill_holes(te_bool)
    te_bool = closing(te_bool, disk(5))
    # Ensure TE is inside main embryo
    te_bool &= (ws_labels > 0)
    return te_bool.astype(np.uint8) * 255


def identify_icm_by_subtraction(embryo_mask, te_mask):
    """ICM = EXP minus TE, keeping the largest connected blob"""
    icm_bool = np.logical_and(embryo_mask > 0, te_mask == 0)
    # Dilate + fill holes
    icm_bool = closing(icm_bool, disk(7))
    icm_bool = binary_fill_holes(icm_bool)
    # Largest connected component
    lab_icm = label(icm_bool)
    if lab_icm.max() > 0:
        regions = regionprops(lab_icm)
        largest = max(regions, key=lambda r: r.area).label
        icm_bool = (lab_icm == largest)
    # Mild closing
    icm_bool = closing(icm_bool, disk(5))
    return icm_bool.astype(np.uint8) * 255


def save_visualization(img, embryo_mask, te_mask, icm_mask, combined_mask, save_path):
    """Save visualization results"""
    # Create colored mask visualization
    color_mask = np.zeros((*combined_mask.shape, 3), dtype=np.uint8)

    # Assign colors: background=black, EXP=red, TE=green, ICM=blue
    color_mask[combined_mask == 1] = [255, 0, 0]  # EXP - red
    color_mask[combined_mask == 2] = [0, 255, 0]  # TE - green
    color_mask[combined_mask == 3] = [0, 0, 255]  # ICM - blue

    # Create overlay with transparency
    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    # Draw individual masks
    titles = ["Original Image", "EXP Mask", "TE Mask", "ICM Mask", "Combined Mask", "Overlay"]
    imgs = [
        img,
        cv2.cvtColor(embryo_mask, cv2.COLOR_GRAY2RGB) if embryo_mask.ndim == 2 else embryo_mask,
        cv2.cvtColor(te_mask, cv2.COLOR_GRAY2RGB) if te_mask.ndim == 2 else te_mask,
        cv2.cvtColor(icm_mask, cv2.COLOR_GRAY2RGB) if icm_mask.ndim == 2 else icm_mask,
        color_mask,
        overlay
    ]

    # Create large figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for ax, im, title in zip(axes, imgs, titles):
        if im.ndim == 3:  # RGB image
            ax.imshow(im)
        else:  # Grayscale image
            ax.imshow(im, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Configure paths
    IMAGE_DIR = "Blastocyst_Dataset/Images"  # original image folder
    MASK_DIR = "Blastocyst_Dataset/Masks"  # mask output folder
    PLOT_DIR = "Blastocyst_Dataset/Visualization"  # visualization output folder (optional)

    # Generate all segmentation masks
    generate_segmentation_masks(IMAGE_DIR, MASK_DIR, PLOT_DIR)

    print("Segmentation mask generation complete!")