import cv2
import numpy as np
import os
from glob import glob
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from scipy.stats import skew


def extract_lbp_features(image, P=8, R=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7
    return hist


def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_quantized = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(
        gray_quantized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "homogeneity", "energy", "correlation"]
    feats = []
    for prop in props:
        val = graycoprops(glcm, prop).flatten()
        feats.append(np.mean(val))
        feats.append(np.var(val))
    return np.array(feats)


def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_color_moments(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []
    h, s, v = cv2.split(hsv)
    for channel in [h, s, v]:
        features.append(np.mean(channel))
        features.append(np.var(channel))
        s = skew(channel.flatten())
        if np.isnan(s):
            s = 0
        features.append(s)
    return np.array(features)


def extract_hog_features(image):
    resized = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog_feats = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
    )
    return hog_feats


def extract_all_features(image):
    lbp = extract_lbp_features(image)
    glcm = extract_glcm_features(image)
    color_hist = extract_color_histogram(image)
    color_moments = extract_color_moments(image)
    hog_feats = extract_hog_features(image)
    combined = np.concatenate([lbp, glcm, color_hist, color_moments, hog_feats])
    combined = np.nan_to_num(combined)
    return combined


def process_dataset(source_dir, output_file_X, output_file_y):
    classes = sorted(
        [
            d
            for d in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, d))
        ]
    )
    label_map = {cls: i for i, cls in enumerate(classes)}
    X = []
    y = []
    print(f"Starting Feature Extraction on {source_dir}...")
    img_paths = []
    img_labels = []
    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        imgs = glob(os.path.join(class_dir, "*.jpg")) + glob(
            os.path.join(class_dir, "*.png")
        )
        img_paths.extend(imgs)
        img_labels.extend([label_map[class_name]] * len(imgs))
    print(f"Found {len(img_paths)} images across {len(classes)} classes.")
    for i, img_path in enumerate(img_paths):
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue
        features = extract_all_features(image)
        X.append(features)
        y.append(img_labels[i])
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(img_paths)} images...")
    X = np.array(X)
    y = np.array(y)
    print("Feature Extraction Complete.")
    print(f"Feature Vector Shape: {X.shape}")
    np.save(output_file_X, X)
    np.save(output_file_y, y)
    print(f"Saved features to {output_file_X} and labels to {output_file_y}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    SOURCE_DIR = os.path.join(current_dir, "dataset_augmented")
    OUTPUT_X = os.path.join(current_dir, "features_X.npy")
    OUTPUT_Y = os.path.join(current_dir, "labels_y.npy")
    if os.path.exists(SOURCE_DIR):
        process_dataset(SOURCE_DIR, OUTPUT_X, OUTPUT_Y)
    else:
        print(f"Source directory {SOURCE_DIR} not found.")
