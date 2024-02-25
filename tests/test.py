import torch
import os 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
import numpy as np
from PIL import Image
from utils import load_checkpoint,save_predictions_as_imgs
from torchvision.utils import save_image

# HYPERPARAMETERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "./my_checkpoint.pth.tar"
TEST_IMAGE_DIR = "./data/test_images"
PREDICTED_IMAGE_DIR = "./predicted_images"

# Ensure the predicted_images directory exists
os.makedirs(PREDICTED_IMAGE_DIR, exist_ok=True)

# Define the image height and width
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

# Define the validation transforms
test_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
load_checkpoint(torch.load(CHECKPOINT_PATH, map_location=DEVICE), model)
model.eval()
index=0
for image_file in os.listdir(TEST_IMAGE_DIR):
    image_path = os.path.join(TEST_IMAGE_DIR, image_file)
    image = np.array(Image.open(image_path))
    
    #Apply transformation
    transformed= test_transforms(image=image)
    image = transformed["image"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        predictions = model(image)
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    
    filepath = os.path.join(PREDICTED_IMAGE_DIR, f"pred_{index}.jpg")
    save_image(predictions, filepath)
    index+=1
    