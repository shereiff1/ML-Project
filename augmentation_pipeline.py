import cv2
import numpy as np
import os
import random
from glob import glob

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height))

def flip_image(image, code):
    return cv2.flip(image, code)

def add_noise(image, scale):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss * scale
    return np.clip(noisy, 0, 255).astype(np.uint8)

def adjust_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def augment_dataset(source_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    ROTATION_RANGE = 20
    NOISE_SCALE = 10
    BRIGHTNESS_RANGE = (-20, 20)
    total_original = 0
    total_augmented = 0

    print(f"Found classes: {classes}")
    class_counts = {}
    for class_name in classes:
        p = os.path.join(source_dir, class_name)
        cnt = len(glob(os.path.join(p, "*.jpg")) + glob(os.path.join(p, "*.png")))
        class_counts[class_name] = cnt
    max_count = max(class_counts.values()) 
    TARGET_PER_CLASS = max_count * 2
    print(f"Balancing classes. Target per class: {TARGET_PER_CLASS}")
    aug_types = ['rotate', 'flip', 'brightness', 'noise']
    weights = [30, 30, 20, 20]
    for class_name in classes:
        class_src_dir = os.path.join(source_dir, class_name)
        class_dst_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dst_dir):
            os.makedirs(class_dst_dir)
        images = glob(os.path.join(class_src_dir, "*.jpg")) + glob(os.path.join(class_src_dir, "*.png"))
        original_count = len(images)
        print(f"Processing {class_name}: {original_count} original images...")
        for img_path in images:
            img_name = os.path.basename(img_path)
            image = cv2.imread(img_path)
            if image is None:
                continue
            cv2.imwrite(os.path.join(class_dst_dir, img_name), image)
            total_original += 1
        needed = TARGET_PER_CLASS - original_count
        if needed <= 0:
            continue
        print(f"  - Generating {needed} augmentations...")
        for i in range(needed):
            src_img_path = random.choice(images)
            image = cv2.imread(src_img_path)
            if image is None: continue
            aug_type = random.choices(aug_types, weights=weights, k=1)[0]
            aug_img = image.copy()
            suffix = ""
            if aug_type == 'rotate':
                angle = random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
                aug_img = rotate_image(image, angle)
                suffix = f"_rot{int(angle)}"
                
            elif aug_type == 'flip':
                code = random.choice([-1, 0, 1])
                aug_img = flip_image(image, code)
                suffix = f"_flip{code}"
                
            elif aug_type == 'brightness':
                value = random.randint(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
                aug_img = adjust_brightness(image, value)
                suffix = f"_bright{value}"
                
            elif aug_type == 'noise':
                aug_img = add_noise(image, NOISE_SCALE)
                suffix = f"_noise"
                
            name, ext = os.path.splitext(os.path.basename(src_img_path))
            new_name = f"{name}{suffix}_{i}{ext}"
            try:
                cv2.imwrite(os.path.join(class_dst_dir, new_name), aug_img)
                total_augmented += 1
            except Exception as e:
                print(f"Error saving augmented image: {e}")

    print(f"Done. Original: {total_original}, Augmented: {total_augmented}, Total: {total_original + total_augmented}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    SOURCE_DIR = os.path.join(current_dir, "dataset")
    OUTPUT_DIR = os.path.join(current_dir, "dataset_augmented")
    augment_dataset(SOURCE_DIR, OUTPUT_DIR)
