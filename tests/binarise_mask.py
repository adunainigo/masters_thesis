import os
from PIL import Image
import numpy as np

# Hyperparameter
MASKDIR = './data/val_masks' # Placeholder for the actual directory path

# Function to convert images in a directory to binary masks
def binary_mask(mask_dir):
    # Iterate over all files in the directory
    for filename in os.listdir(mask_dir):
        file_path = os.path.join(mask_dir, filename)
        # Check if the file is an image
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(file_path)
            array = np.array(image)
            # Apply threshold to create a binary mask
            binary_array = np.where(array < 128, 0, 254).astype(np.uint8)
            binary_image = Image.fromarray(binary_array)
            # Save the binary mask with a new filename
            binary_image.save(os.path.join(mask_dir, f'{filename}'))

# Since we cannot execute the function without a valid MASKDIR, we comment out the function call
binary_mask(MASKDIR)

# Code is ready for a real directory path to be substituted in place of the placeholder.
