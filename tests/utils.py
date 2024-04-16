import torch
import torchvision
from dataset import RobotDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = RobotDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = RobotDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_tp = 0
    num_fp=0
    num_tn=0
    num_fn=0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            y = torch.sigmoid(y)
            y = (y > 0.5).float()
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            num_tp+= (preds * y).sum() #y=1 pred=1
            num_tn+= ((1-preds)* (1-y)).sum() #y=0 pred=0
            num_fp += (preds * (1 - y)).sum() #y=0 pred=1
            num_fn += ((1 - preds) * y).sum() #y=1 pred=0
            
    print (f"FP: {num_fp}, FN: {num_fn}, TP: {num_tp}, TN: {num_tn}")
    # Dice score and accuracy calculation
    accuracy = (num_tp + num_tn) / (num_tp + num_tn + num_fp + num_fn)
    dice_score = (2 * num_tp) / ((2 * num_tp) + num_fp + num_fn + 1e-8)
    
    model.train()
    return accuracy*100, dice_score*100

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()