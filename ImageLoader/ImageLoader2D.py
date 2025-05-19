import glob
import os
import tensorflow as tf
import numpy as np
from PIL import Image  # Retained for potential use if tf.image.decode_image has issues with specific formats, but tf.image.decode_image is preferred.
from skimage.io import imread  # Retained for similar reasons, but tf.io.read_file + tf.image.decode_image is preferred.


# folder_path will be set from the notebook
# Example: folder_path = "/path/to/your/dataset_root_folder/"
# This root folder should contain 'train', 'masks', and 'test' subdirectories.

def get_image_paths(base_folder_path):
    """
    Gets all image and mask paths from the specified base folder.
    Tries two common directory structures.
    Mask filenames are assumed to match training image filenames (possibly with different extensions).
    """
    # Strategy 1: Nested structure (e.g., train/images, train/masks, test/images)
    train_img_dir_s1 = os.path.join(base_folder_path, 'train', 'images')
    # Original assumption for S1 masks: base_folder_path/train/masks
    mask_dir_s1 = os.path.join(base_folder_path, 'train', 'masks')
    test_img_dir_s1 = os.path.join(base_folder_path, 'test', 'images')

    print(f"Attempting Strategy 1 paths:\n  Train Imgs: {train_img_dir_s1}\n  Masks: {mask_dir_s1}\n  Test Imgs: {test_img_dir_s1}")
    
    # Use a generic pattern for globbing, e.g., '*' or specific image types '*.[pjt][npji][gef]*'
    # For simplicity, using "*.*" which was original, but can be refined.
    img_file_pattern = "*.*" 
    train_image_paths = sorted(glob.glob(os.path.join(train_img_dir_s1, img_file_pattern)))

    if train_image_paths:
        print(f"Strategy 1: Found {len(train_image_paths)} training images in {train_img_dir_s1}.")
        effective_train_img_dir = train_img_dir_s1
        effective_mask_dir = mask_dir_s1
        effective_test_img_dir = test_img_dir_s1
    else:
        print(f"Strategy 1 failed to find training images in {train_img_dir_s1}.")
        # Strategy 2: Flatter structure (e.g., train/, masks/, test/)
        train_img_dir_s2 = os.path.join(base_folder_path, 'train')
        mask_dir_s2 = os.path.join(base_folder_path, 'masks') # Masks in a root 'masks' folder
        test_img_dir_s2 = os.path.join(base_folder_path, 'test')
        
        print(f"Attempting Strategy 2 paths:\n  Train Imgs: {train_img_dir_s2}\n  Masks: {mask_dir_s2}\n  Test Imgs: {test_img_dir_s2}")
        train_image_paths = sorted(glob.glob(os.path.join(train_img_dir_s2, img_file_pattern)))

        if train_image_paths:
            print(f"Strategy 2: Found {len(train_image_paths)} training images in {train_img_dir_s2}.")
            effective_train_img_dir = train_img_dir_s2
            effective_mask_dir = mask_dir_s2
            effective_test_img_dir = test_img_dir_s2
        else:
            print(f"Strategy 2 also failed to find training images in {train_img_dir_s2}.")
            print("Using Strategy 1 paths by default for error consistency, though no training images were found.")
            effective_train_img_dir = train_img_dir_s1
            effective_mask_dir = mask_dir_s1
            effective_test_img_dir = test_img_dir_s1
            # train_image_paths is already empty here

    print(f"\nEffective paths determined:\n  Train Img Dir: {effective_train_img_dir}\n  Mask Dir: {effective_mask_dir}\n  Test Img Dir: {effective_test_img_dir}\n")

    train_mask_paths_intermediate = [] # Store found mask paths before filtering
    if not os.path.isdir(effective_mask_dir) and train_image_paths:
        print(f"Warning: Mask directory '{effective_mask_dir}' does not exist. Cannot load masks.")
    elif os.path.isdir(effective_mask_dir) and train_image_paths:
        print(f"Searching for masks in '{effective_mask_dir}'...")
        for img_path in train_image_paths:
            fname = os.path.basename(img_path)
            base_fname = os.path.splitext(fname)[0] # Use os.path.splitext for robust extension removal
            
            potential_mask_extensions = ['.png', '.tif', '.jpg', '.jpeg', '.gif']
            # Also check for mask with same name as image (original behavior)
            potential_mask_fnames_to_check = [fname] + [base_fname + ext for ext in potential_mask_extensions]
            
            found_mask_for_current_image = False
            for p_mask_fname in potential_mask_fnames_to_check:
                current_potential_mask_path = os.path.join(effective_mask_dir, p_mask_fname)
                if os.path.exists(current_potential_mask_path):
                    train_mask_paths_intermediate.append(current_potential_mask_path)
                    found_mask_for_current_image = True
                    break 
            
            if not found_mask_for_current_image:
                # This print can be verbose if many masks are missing.
                # print(f"Warning: Mask not found for image {img_path} (tried variants like {base_fname}.png/tif/jpg etc. in {effective_mask_dir}).")
                pass # Do not add a placeholder; filtering will handle it.
        if not train_mask_paths_intermediate and train_image_paths:
            print(f"Warning: No masks found in '{effective_mask_dir}' for any training images.")
        elif train_image_paths:
            print(f"Found {len(train_mask_paths_intermediate)} potential mask files.")


    test_image_paths = sorted(glob.glob(os.path.join(effective_test_img_dir, img_file_pattern)))
    if not test_image_paths and os.path.isdir(effective_test_img_dir):
        print(f"No test images found in '{effective_test_img_dir}' with pattern '{img_file_pattern}'. Check file names/extensions.")
    elif not os.path.isdir(effective_test_img_dir):
        print(f"Warning: Test image directory '{effective_test_img_dir}' does not exist.")

    # Filter to ensure images and masks are paired correctly.
    # This logic assumes that if a mask is found, its basename (w/o ext) matches the image's basename.
    final_train_image_paths = []
    final_train_mask_paths = []

    if train_image_paths and train_mask_paths_intermediate:
        # Create a dictionary of found masks by their base names for efficient lookup
        mask_map = {}
        for m_path in train_mask_paths_intermediate:
            mask_basename = os.path.splitext(os.path.basename(m_path))[0]
            if mask_basename not in mask_map: # Keep first found mask if multiple exist for same base
                 mask_map[mask_basename] = m_path
        
        for img_path in train_image_paths:
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            if img_basename in mask_map:
                final_train_image_paths.append(img_path)
                final_train_mask_paths.append(mask_map[img_basename])
            else:
                print(f"Note: No corresponding mask found for image {img_path} (basename: {img_basename}). This image will be excluded from training.")
        
        if len(final_train_image_paths) != len(train_image_paths):
            print(f"Filtered training set from {len(train_image_paths)} images to {len(final_train_image_paths)} image/mask pairs due to missing/unmatched masks.")
    elif train_image_paths and not train_mask_paths_intermediate: # Images found, but no masks at all
        print("Warning: Training images were found, but no masks were loaded. Training set will be empty.")
        # final_train_image_paths and final_train_mask_paths remain empty

    return final_train_image_paths, final_train_mask_paths, test_image_paths


def tf_dataset_generator(image_paths, mask_paths, img_height, img_width, is_test_set=False):
    """
    TensorFlow dataset generator.
    Loads, decodes, resizes, and normalizes images.
    For masks, it also binarizes them.
    """
    for i in range(len(image_paths)):
        img_path = image_paths[i]

        img_raw = tf.io.read_file(img_path)
        try:
            # expand_animations=False for TF >= 2.3, handles GIF by taking first frame
            img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
        except tf.errors.InvalidArgumentError:  # Fallback for formats decode_image might struggle with
            print(f"TF decode_image failed for {img_path}, trying PIL.")
            pil_img = Image.open(img_path).convert('RGB')
            img = tf.convert_to_tensor(np.array(pil_img), dtype=tf.uint8)

        img = tf.image.resize(img, [img_height, img_width], method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape([img_height, img_width, 3])

        if is_test_set:
            yield img
        else:
            mask_path = mask_paths[i]
            mask_raw = tf.io.read_file(mask_path)
            try:
                mask = tf.image.decode_image(mask_raw, channels=1, expand_animations=False)
            except tf.errors.InvalidArgumentError:
                print(f"TF decode_image failed for mask {mask_path}, trying PIL.")
                pil_mask = Image.open(mask_path).convert('L')
                mask = tf.convert_to_tensor(np.array(pil_mask), dtype=tf.uint8)
                mask = tf.expand_dims(mask, axis=-1)

            mask = tf.image.resize(mask, [img_height, img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            mask = tf.cast(mask >= 127, tf.uint8)  # Binarize mask to 0 or 1
            mask.set_shape([img_height, img_width, 1])
            yield img, mask
