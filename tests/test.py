import torch
import os 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
import numpy as np
from PIL import Image
from utils import load_checkpoint, save_predictions_as_imgs
from torchvision.utils import save_image
import time
import re

# Hyperparameters and configuration settings for model inference.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Set device to CUDA if available, otherwise use CPU.
CHECKPOINT_PATH = "./my_checkpoint.pth.tar"  # Path to the model checkpoint.
TEST_IMAGE_DIR = "./data/test_images"  # Directory containing test images.
PREDICTED_IMAGE_DIR = "./predicted_images"  # Directory to save predicted images.

# Ensure the predicted_images directory exists
os.makedirs(PREDICTED_IMAGE_DIR, exist_ok=True)

# Constants defining the height and width of the images.
IMAGE_HEIGHT = 270  # Set the image height for resizing.
IMAGE_WIDTH  = 480  # Set the image width for resizing.

# Define the transformations for validation images.
test_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),  # Resize images to defined dimensions.
        A.Normalize(
            mean=[0.0, 0.0, 0.0],  # Normalize images with a mean of 0.
            std=[1.0, 1.0, 1.0],    # Standard deviation for normalization.
            max_pixel_value=255.0,  # Maximum pixel value in input images.
        ),
        ToTensorV2(),  # Convert images to tensor format compatible with PyTorch.
    ],
)

model = UNET(in_channels=3, out_channels=1).to(DEVICE)  # Initialize and move the UNET model to the defined device.
load_checkpoint(torch.load(CHECKPOINT_PATH, map_location=DEVICE), model)  # Load model weights from checkpoint.
model.eval()  # Set the model to evaluation mode.

# Process each image in the test image directory.
start_time = time.time()  # Record the start time of processing.
for image_file in os.listdir(TEST_IMAGE_DIR):
    match = re.search(r'image_(\d+).jpg', image_file)  # Extract index from image filename using regex.
    if match:
        index = int(match.group(1))
        
        image_path = os.path.join(TEST_IMAGE_DIR, image_file)  # Path to the image file.
        image = np.array(Image.open(image_path))  # Open and convert the image to a numpy array.
        
        transformed = test_transforms(image=image)  # Apply defined transformations.
        image = transformed["image"].unsqueeze(0).to(DEVICE)  # Add batch dimension and move image to the device.
        
        with torch.no_grad():  # Disable gradient computation.
            predictions = model(image)  # Predict the output using the model.
        predictions = torch.sigmoid(predictions)  # Apply sigmoid to obtain probabilities.
        predictions = (predictions > 0.5).float()  # Threshold the probabilities to generate binary output.
        
        filepath = os.path.join(PREDICTED_IMAGE_DIR, f"pred_mask_{index}.jpg")  # Define the output filepath.
        save_image(predictions, filepath)  # Save the predicted mask image.

end_time = time.time()  # Record the end time of processing.
total_time = end_time - start_time  # Calculate the total processing time.
average_time_per_image = total_time / num_images  # Calculate the average processing time per image.
print(f"Average time per image: {average_time_per_image:.2f} seconds")  # Print the average time per image.
