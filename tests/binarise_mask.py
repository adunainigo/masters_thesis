import os
from PIL import Image
import numpy as np

# Define the directory for storing validation masks
MASKDIR = './data/val_masks'  # Placeholder for the actual directory path

def binary_mask(mask_dir):
    """
    Converts all images in the specified directory to binary masks.
    
    This function iterates through each image file within a directory,
    converts it to a grayscale array, and applies a threshold to create
    a binary mask. The binary mask is then saved in the same directory with
    the same filename.
    
    Parameters:
    mask_dir (str): The directory path that contains the image files.
    
    Returns:
    None: The function saves the binary masks in the same directory and does not return any value.
    """
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




