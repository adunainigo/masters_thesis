import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class RobotDataset(Dataset):
    """
    A custom dataset class for robot image and corresponding mask data, compatible with PyTorch's Dataset interface.

    This dataset class is designed to handle the loading and preprocessing of images and their corresponding segmentation masks used for training machine learning models. The class supports on-the-fly transformations of the dataset items.

    Attributes:
        image_dir (str): Directory path containing the images.
        mask_dir (str): Directory path containing the corresponding mask images.
        transform (callable, optional): A function/transform that takes in an image and mask and returns a transformed version. Defaults to None.
        images (list): List of filenames in the image directory.

    Methods:
        __len__: Returns the number of images in the dataset.
        __getitem__: Fetches the image and mask by index, applies transformations if any, and returns the processed image and mask.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Initializes the dataset with directory paths and an optional transform.

        Parameters:
            image_dir (str): The directory containing the images.
            mask_dir (str): The directory containing the masks.
            transform (callable, optional): A function/transform to apply to each item.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)  # Load all image filenames from the image directory.

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Retrieves an image and its corresponding mask by index, applies transformations, and returns both.

        Parameters:
            index (int): The index of the item.

        Returns:
            tuple: A tuple containing the transformed image and mask arrays.
        """
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.jpg"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0  # Normalizing mask to be 0 for background and 1 for the object, preparing for sigmoid activation.

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask


