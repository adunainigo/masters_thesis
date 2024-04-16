# Import necessary libraries for image manipulation and file management
from PIL import Image, ImageEnhance
import numpy as np
import os
from glob import glob
import random

# Set directories for images and masks, and define brightness adjustment factors
IMAGE_DIR = "./data/val_images/"
MASK_DIR = "./data/val_masks/"
brightness_factors = [0.8, 1.2]  # Darken by 20%, brighten by 20%

def change_brightness(image, brightness_factor):
    """
    Adjusts the brightness of the specified image by the given factor.
    
    Parameters:
    - image (PIL.Image): The image to adjust.
    - brightness_factor (float): The factor to adjust brightness. >1 brightens the image, <1 darkens it.
    
    Returns:
    - PIL.Image: The adjusted image.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness_factor)

def apply_shift(image, shift_value):
    """
    Applies a spatial shift to the specified image based on the given shift values.
    
    Parameters:
    - image (PIL.Image): The image to shift.
    - shift_value (tuple): A tuple (x_shift, y_shift) specifying the shift amounts in pixels.
    
    Returns:
    - PIL.Image: The shifted image.
    """
    return image.transform(image.size, Image.AFFINE, (1, 0, shift_value[0], 0, 1, shift_value[1]))

def add_noise(image, noise_type):
    """
    Adds specified noise type to the image.
    
    Currently supports Gaussian noise.
    
    Parameters:
    - image (PIL.Image): The image to modify.
    - noise_type (str): Type of noise to add, currently only "gaussian".
    
    Returns:
    - PIL.Image: The noisy image.
    """
    if noise_type == "gaussian":
        np_image = np.array(image)
        row, col, ch = np_image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = np_image + gauss
        return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
    
# Function dataset_augmentation
def dataset_augmentation(imagedir, maskdir, target_count=500):
    # Determine the starting index for the names of the modified images/masks
    start_index = 50

    # List files in the image and mask directories
    image_files = sorted(glob(os.path.join(imagedir, "image_*.jpg")))
    mask_files = sorted(glob(os.path.join(maskdir, "image_*.jpg")))
    print(f"image_files: {len(image_files)}")
    print(f"mask_files: {len(mask_files)}")
    
    # Ensure that each image has its corresponding mask
    print(len(image_files))
    print(len(mask_files))
    assert len(image_files) == len(mask_files), "The number of images and masks does not match."

    # Transformations to apply
    rotations = [30, 60, 90, 120, 150]  # Additional rotations
    shifts = [(10, 10), (-10, -10), (20, 20), (-20, -20)]  # Various shift values
    # Combine rotations and shifts into a single list of transformations
    transformations = [{"type": "rotate", "value": angle} for angle in rotations] + \
                      [{"type": "shift", "value": shift} for shift in shifts]
    transformations += [{"type": "brightness", "value": factor} for factor in brightness_factors]
    # Image augmentation loop
    while start_index < target_count + 50:  # +50 because we start counting from image_50
        for image_path, mask_path in zip(image_files, mask_files):
            if start_index >= target_count + 50:  # Check if we've reached the target count
                break

            image = Image.open(image_path)
            mask = Image.open(mask_path)

            for transformation in transformations:
                # Apply each transformation to both the image and its mask
                transformed_image, transformed_mask = apply_transformation(image, mask, transformation)

                # Save the transformed image and mask
                transformed_image.save(os.path.join(imagedir, f"image_{start_index}.jpg"))
                transformed_mask.save(os.path.join(maskdir, f"image_{start_index}_mask.jpg"))

                # Increment the index for the next file name
                start_index += 1

            # Apply noise to 20% of the images
            if random.random() < 0.2:  # 20% chance
                noisy_image = add_noise(image, "gaussian")
                noisy_image.save(os.path.join(imagedir, f"image_{start_index}.jpg"))
                mask.save(os.path.join(maskdir, f"image_{start_index}_mask.jpg"))
                start_index += 1


def dataset_augmentation_val(imagedir, maskdir, start_index=1000, transformations_per_image=10, total_target=40):
    """
    Applies a series of transformations to each image in the dataset for augmentation, saving the augmented images and masks.
    
    Parameters:
    - imagedir (str): Directory containing images.
    - maskdir (str): Directory containing corresponding masks.
    - start_index (int): Index to start naming augmented files.
    - transformations_per_image (int): Number of transformations to apply per image.
    - total_target (int): Total number of augmented images to generate.
    
    Operations include rotation, spatial shifts, and brightness adjustments. Each transformation is applied to the image, and the corresponding mask is processed similarly except for brightness adjustments.
    """
    # Define transformations
    rotations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    shifts = [(10, 10), (-10, -10), (20, 20), (-20, -20)]
    transformations = [('rotate', r) for r in rotations] + \
                      [('shift', s) for s in shifts] + \
                      [('brightness', b) for b in brightness_factors]

    # Randomize and select a subset of transformations for each image
    random.shuffle(transformations)
    transformations = transformations[:transformations_per_image]

    image_files = sorted(glob(os.path.join(imagedir, "image_*.jpg")))
    mask_files = sorted(glob(os.path.join(maskdir, "image_*.jpg")))
    assert len(image_files) == len(mask_files), "Each image must have a corresponding mask."

    current_index = start_index
    for image_path, mask_path in zip(image_files, mask_files):
        if current_index >= start_index + total_target:
            break

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        for transformation in transformations:
            t_type, t_value = transformation

            if t_type == 'rotate':
                transformed_image = image.rotate(t_value)
                transformed_mask = mask.rotate(t_value)
            elif t_type == 'shift':
                transformed_image = apply_shift(image, t_value)
                transformed_mask = apply_shift(mask, t_value)
            elif t_type == 'brightness':
                transformed_image = change_brightness(image, t_value)
                transformed_mask = mask  # Mask does not change with brightness adjustments

            transformed_image.save(os.path.join(imagedir, f"image_{current_index}.jpg"))
            transformed_mask.save(os.path.join(maskdir, f"image_{current_index}_mask.jpg"))
            current_index += 1
            if current_index >= start_index + total_target:
                break
            

def apply_transformation(image, mask, transformation):
    """
    Applies a specified transformation to an image and its corresponding mask.
    
    This function handles different types of transformations including rotation, spatial shifting, adding noise,
    and adjusting brightness. The mask is transformed similarly to the image except in cases of brightness adjustments
    and noise addition where only the image is affected.
    
    Parameters:
    - image (PIL.Image): The image to transform.
    - mask (PIL.Image): The mask corresponding to the image.
    - transformation (dict): A dictionary specifying the type of transformation ('rotate', 'shift', 'noise', 'brightness')
      and the value associated with that transformation. The value could be an angle for rotation, a tuple for shift, 
      a noise type for adding noise, or a brightness factor for adjusting brightness.
    
    Returns:
    - tuple: A tuple containing the transformed image and mask.
    """
    if transformation["type"] == "rotate":
        # Rotate both the image and mask by the specified angle
        angle = transformation["value"]
        return image.rotate(angle), mask.rotate(angle)
    elif transformation["type"] == "shift":
        # Shift both the image and mask by the specified x and y amounts
        shift = transformation["value"]
        return (
            image.transform(image.size, Image.AFFINE, (1, 0, shift[0], 0, 1, shift[1])),
            mask.transform(mask.size, Image.AFFINE, (1, 0, shift[0], 0, 1, shift[1]))
        )
    elif transformation["type"] == "noise":
        # Add noise only to the image; the mask remains unchanged
        noisy_image = add_noise(image, transformation["value"])
        return noisy_image, mask
    elif transformation['type'] == 'brightness':
        # Adjust brightness only for the image; the mask remains unchanged
        transformed_image = change_brightness(image, transformation['value'])
        return transformed_image, mask

# Example usage:
# dataset_augmentation_val(IMAGE_DIR, MASK_DIR, start_index=1000, transformations_per_image=10, total_target=50)










