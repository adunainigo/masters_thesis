import torch
import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from torch.utils.tensorboard import SummaryWriter
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 270 #135,270,540,1080
IMAGE_WIDTH = 480  #240,480,960,1920
PIN_MEMORY = True 
LOAD_MODEL = False
#Directories
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
GRAPH_DIR    = "./graph_dir"

#EARLY STOPPING
BEST_ACCURACY = 0.0
BEST_DICE_SCORE = 0.0
PATIENCE=10
PATIENCE_COUNTER=0

#RUNNING WRITER
RUN_NAME = "u_net"

#graph variables
loss_list=[]
accuracy_list=[]
dice_score_list=[]


def plot_metrics(graph_dir,list1,list2=None,num_epochs=None,lr=None,label1=None,label2=None):
    index=0
    epochs=list(range(1,len(list1) +1))
    plt.plot(epochs, list1, color='blue', label=label1)
    if list2 is not None:
        index=1
        plt.plot(epochs, list2, color='red', label=label2)
    plt.title(f'Metrics over epochs (LR: {lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    plt.savefig(f"{graph_dir}/lr_{lr}_ep_{num_epochs}_{index}.png")
    plt.clf()
    
def train_fn(loader, model, optimizer, loss_fn, scaler,epoch,writer):
    loop = tqdm(loader)
    max_predictions = [] #Delete later
    max_targets = []     #Delete later
    model.train()
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        targets = torch.sigmoid(targets)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            # Escalate predictions to 0 or 254 after sigmoid
            predictions = torch.sigmoid(predictions)
            loss = loss_fn(predictions, targets)
        writer.add_scalar("Training loss", loss.item(), epoch * len(loader) + batch_idx)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        max_predictions.append(predictions.max().item())
        max_targets.append(targets.max().item())
        

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
    loss_epoch = loss_fn(predictions, targets)
    loss_list.append(loss_epoch.item())
    print(f"Max prediction value: {max(max_predictions)}") #Delete later
    print(f"Max target value: {max(max_targets)}") #Delete later

def main():
    writer = SummaryWriter(f"runs/{RUN_NAME}")
    #Definition of image transformations using Albumentations ('A') used for data augmentation  
    train_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=max(IMAGE_HEIGHT, IMAGE_WIDTH), always_apply=True),
        A.CenterCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, always_apply=True),
        A.PadIfNeeded(min_height=IMAGE_HEIGHT, min_width=IMAGE_WIDTH, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
        A.Rotate(limit=35, p=1.0), # Rotation +-35º p=1=> En todas las imágenes
        A.HorizontalFlip(p=0.5),   # Mirror horizontally images with prob 50%
        A.VerticalFlip(p=0.1),     # Mirror vertically images with prob 10%
        A.Normalize(               # Image normalization -mean/std 
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
    )

    val_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=max(IMAGE_HEIGHT, IMAGE_WIDTH), always_apply=True),
        A.CenterCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, always_apply=True),
        A.PadIfNeeded(min_height=IMAGE_HEIGHT, min_width=IMAGE_WIDTH, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
    )

    # For multiclass:
    #    out_channels=N with N=#classes
    #    loss_fn= cross entropy loss

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() #Binary Cross Entropy 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler() #T

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler,epoch, writer)
        global BEST_ACCURACY, BEST_DICE_SCORE,PATIENCE,PATIENCE_COUNTER
        #check accuracy and dice score
        accuracy,dice_score=check_accuracy(val_loader, model, device=DEVICE)
        
        #Save metrics for graph
        accuracy_list.append(accuracy.item())
        dice_score_list.append(dice_score.item())
        
        writer.add_scalar("Accuracy", accuracy, epoch)
        writer.add_scalar("Dice Score", dice_score, epoch)
        
        #check if this is the best model
        if (accuracy > BEST_ACCURACY*0.8) and (dice_score > BEST_DICE_SCORE):
            BEST_ACCURACY=accuracy
            BEST_DICE_SCORE=dice_score
            print(f"Epoch {epoch+1}, New best model with accuracy: {accuracy.item():.4f}, Dice Score: {dice_score.item():.4f}")
            
            #Save model
            checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            }
            
            save_checkpoint(checkpoint)
            PATIENCE_COUNTER=0 #Reset patience
        else:
            PATIENCE_COUNTER+=1
            print(f"Epoch {epoch+1}, No improvement. Patience: {PATIENCE_COUNTER}/{PATIENCE}")
        if PATIENCE_COUNTER >=PATIENCE:
            print("Early stopping triggered. No improvement in {patience} epochs.")
            print(f"Best model: Accuracy= {BEST_ACCURACY:.4f}, Dice_score= {BEST_DICE_SCORE:.4f}.")
            break
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
        
    plot_metrics(GRAPH_DIR,accuracy_list,dice_score_list,NUM_EPOCHS,LEARNING_RATE,"Accuracy","Dice_score")
    plot_metrics(graph_dir=GRAPH_DIR,list1=loss_list,num_epochs=NUM_EPOCHS,lr=LEARNING_RATE,label1="loss_fn")
if __name__ == "__main__":
    main()
    
    
    
    