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
    Assumes 'train' folder for training images, 'masks' for training masks,
    and 'test' for test images. Mask filenames are assumed to match training image filenames.
    """
    train_img_dir = os.path.join(base_folder_path, 'train')
    mask_dir = os.path.join(base_folder_path, 'masks')
    test_img_dir = os.path.join(base_folder_path, 'test')

    train_image_paths = sorted(glob.glob(os.path.join(train_img_dir, "*.*")))

    train_mask_paths = []
    for img_path in train_image_paths:
        fname = os.path.basename(img_path)
        # Attempt common extensions for masks if not directly matching (e.g. .png for .jpg image)
        # This logic might need adjustment based on exact naming conventions.
        # Simplest assumption: mask has same name and is a common mask format like png.
        potential_mask_path_png = os.path.join(mask_dir, fname.split('.')[0] + '.png')
        potential_mask_path_tif = os.path.join(mask_dir, fname.split('.')[0] + '.tif')
        potential_mask_path_jpg = os.path.join(mask_dir, fname.split('.')[0] + '.jpg')

        if os.path.exists(os.path.join(mask_dir, fname)):
            train_mask_paths.append(os.path.join(mask_dir, fname))
        elif os.path.exists(potential_mask_path_png):
            train_mask_paths.append(potential_mask_path_png)
        elif os.path.exists(potential_mask_path_tif):
            train_mask_paths.append(potential_mask_path_tif)
        elif os.path.exists(potential_mask_path_jpg):
            train_mask_paths.append(potential_mask_path_jpg)
        else:
            # If mask is not found, you might want to raise an error or skip the image
            print(f"Warning: Mask not found for image {img_path}")
            # For now, let's assume it must exist or this will cause issues later.
            # A robust solution would filter out images without masks here.
            # For simplicity, assuming direct name match or common extensions.
            # Fallback to expecting exact same filename in mask folder.
            train_mask_paths.append(os.path.join(mask_dir, fname))

    test_image_paths = sorted(glob.glob(os.path.join(test_img_dir, "*.*")))

    # Filter out train images for which masks were not found if lists are unequal
    if len(train_image_paths) != len(train_mask_paths):
        print("Warning: Mismatch in number of training images and masks. Please check paths and naming.")
        # This part needs robust handling, e.g. by creating pairs and filtering
        valid_train_image_paths = []
        valid_train_mask_paths = []
        mask_basenames = {os.path.basename(p).split('.')[0]: p for p in train_mask_paths}
        for img_p in train_image_paths:
            img_basename = os.path.basename(img_p).split('.')[0]
            if img_basename in mask_basenames:
                valid_train_image_paths.append(img_p)
                valid_train_mask_paths.append(mask_basenames[img_basename])
        train_image_paths = valid_train_image_paths
        train_mask_paths = valid_train_mask_paths
        print(f"Filtered to {len(train_image_paths)} image/mask pairs.")

    return train_image_paths, train_mask_paths, test_image_paths


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
