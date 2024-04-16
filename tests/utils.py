import torch
import torchvision
from dataset import RobotDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Saves the current state of the model to a file.

    Parameters:
        state (dict): The state of the model, typically containing model parameters.
        filename (str, optional): The filename where the state will be saved. Defaults to "my_checkpoint.pth.tar".
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """
    Loads the model state from a checkpoint file.

    Parameters:
        checkpoint (dict): The checkpoint containing model state as saved previously.
        model (torch.nn.Module): The model instance where the state will be loaded.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir, train_maskdir, val_dir, val_maskdir, batch_size,
    train_transform, val_transform, num_workers=4, pin_memory=True
):
    """
    Creates data loaders for the training and validation datasets.

    Parameters:
        train_dir (str): Directory with training images.
        train_maskdir (str): Directory with training masks.
        val_dir (str): Directory with validation images.
        val_maskdir (str): Directory with validation masks.
        batch_size (int): Batch size for data loaders.
        train_transform (callable): Transformations to apply to training data.
        val_transform (callable): Transformations to apply to validation data.
        num_workers (int, optional): Number of worker threads for loading data. Defaults to 4.
        pin_memory (bool, optional): If True, the data loader will copy tensors into CUDA pinned memory. Defaults to True.

    Returns:
        tuple: A tuple containing the DataLoader instances for training and validation datasets.
    """
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
    """
    Computes the accuracy and Dice score of the model using a given loader.

    Parameters:
        loader (DataLoader): The DataLoader for the dataset to evaluate.
        model (torch.nn.Module): The model to evaluate.
        device (str, optional): The device to use for computation. Defaults to "cuda".

    Returns:
        tuple: A tuple containing the accuracy and Dice score, both multiplied by 100.
    """
    num_tp = 0
    num_fp = 0
    num_tn = 0
    num_fn = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            y = torch.sigmoid(y)
            y = (y > 0.5).float()
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_tp += (preds * y).sum()
            num_tn += ((1-preds) * (1-y)).sum()
            num_fp += (preds * (1 - y)).sum()
            num_fn += ((1 - preds) * y).sum()

    print(f"FP: {num_fp}, FN: {num_fn}, TP: {num_tp}, TN: {num_tn}")
    accuracy = (num_tp + num_tn) / (num_tp + num_tn + num_fp + num_fn)
    dice_score = (2 * num_tp) / ((2 * num_tp) + num_fp + num_fn + 1e-8)

    model.train()
    return accuracy * 100, dice_score * 100

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    """
    Saves the model's predictions as images for each batch in the loader.

    Parameters:
        loader (DataLoader): The DataLoader to get batches from.
        model (torch.nn.Module): The model to generate predictions.
        folder (str, optional): The directory where the images will be saved. Defaults to "saved_images/".
        device (str, optional): The device to use for computation. Defaults to "cuda".
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
