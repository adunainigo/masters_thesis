import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import json
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from typing import List
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from glob import glob

#HYPERPARAMETERS
TRAIN_GT_LABELS   = './data/train_gt_labels/label_*.txt'
TRAIN_PRED_LABELS = './data/train_pred_labels/label_*.txt'
VAL_GT_LABELS     = './data/val_gt_labels/label_*.txt'
VAL_PRED_LABELS   = './data/val_pred_labels/label_*.txt'
MODEL_PATH =        "./training_run_best_model.tar"
TEST_PRED_LABELS = './data/test_pred_labels/label_*.txt'


BATCH_SIZE = 5
LEARNING_RATE = 1e-3
L2_REGULARIZATION = 1e-5
NUMEPOCHS = 10000
INPUT_SIZE= 8
HIDDEN_SIZE = [200,200]
OUTPUT_SIZE = 3
RUN_NAME = f"lr_1e-3_h_200-200"
    
    
#DATA READING FUNCTIONS
def read_gt_labels(file_path: str) -> np.ndarray:
    return pd.read_csv(file_path, header=None, sep=" ").values

def read_predicted_labels(file_path: str) -> np.ndarray:
    with open(file_path, 'r') as f:
        data = json.load(f)
    transformed_data = []
    for item in data:
        values = [
            item['aspect_ratio'],
            item['eccentricity'],
            item['area'],
            item['perimeter'],
            item['centroid'][0],
            item['centroid'][1],
            item['orientation'],
            1 if item['convexity'] else 0
        ]
        transformed_data.append(values)
    return np.array(transformed_data)
#DATASET DEFINITION (Correct)
class LabelsDataset(Dataset):
    def __init__(self, gt_paths_train, gt_paths_val, pred_paths_train, pred_paths_val):
        gt_train   = sorted(glob(gt_paths_train), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        gt_val     = sorted(glob(gt_paths_val), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        pred_train = sorted(glob(pred_paths_train), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        pred_val   = sorted(glob(pred_paths_val), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        self.gt_labels_train = [read_gt_labels(file) for file in gt_train]
        self.gt_labels_val = [read_gt_labels(file) for file in gt_val]
        
        self.pred_labels_train = [read_predicted_labels(file) for file in pred_train]
        self.pred_labels_val = [read_predicted_labels(file) for file in pred_val]
        
        self.train = True 
        
    def train_mode(self):
        self.train = True

    def val_mode(self):
        self.train = False

    def __len__(self):
        # Devuelve el tamaño del dataset según el modo (entrenamiento o validación)
        return len(self.gt_labels_train) if self.train else len(self.gt_labels_val)

    def __getitem__(self, idx):
        if self.train:
            gt_label = self.gt_labels_train[idx]
            pred_label = self.pred_labels_train[idx]
        else:
            gt_label = self.gt_labels_val[idx]
            pred_label = self.pred_labels_val[idx]

        # Aplicar squeeze para eliminar dimensiones de tamaño 1
        gt_label_tensor = torch.tensor(gt_label, dtype=torch.float).squeeze()
        pred_label_tensor = torch.tensor(pred_label, dtype=torch.float).squeeze()

        return pred_label_tensor, gt_label_tensor
        
#MLP_CLASS  
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # Dropout con una probabilidad de 0.2
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        layers.append(nn.Tanh())
    
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
    
    
#input_size = 8 
#hidden_sizes = [10, 10, 10]  
#output_size = 3  
#mlp = MLP(input_size, hidden_sizes, output_size)

def training_step(model, data_loader, loss_fn, optimizer, writer, epoch):
    model.train()
    for inputs, targets in data_loader:
        # 0.- Clean up gradients from previous iteration
        optimizer.zero_grad()
        # 1.- Pass the training data through the model
        outputs = model(inputs)
        # 2.- Compute the loss value
        loss = loss_fn(outputs, targets)
        # 4.- Compute grad of loss wrt model params
        loss.backward()
        # 5.- Take optimization step
        optimizer.step()
    # 6.- Log the training loss
    writer.add_scalar(f"{loss_fn.__class__.__name__}/Train", loss.item(), epoch)
    return loss.item()

def validation_step(model, data_loader, loss_fn, writer, epoch):
    model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            # 1.- Pass the validation data through the model
            outputs = model(inputs)
            # 2.- Compute the loss value
            val_loss = loss_fn(outputs, targets)
    # 3.- Log the training loss
    writer.add_scalar(f"{loss_fn.__class__.__name__}/Val", val_loss.item(), epoch)
    return val_loss

def training_loop(model, train_loader, val_loader, optimizer, run_name, epochs, loss_fn):
    writer = SummaryWriter(f"runs/{run_name}")
    best_loss = torch.inf
    best_epoch = -1

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)):
    #for epoch in range(epochs):
        training_loss = training_step(model, train_loader, loss_fn, optimizer, writer, epoch)
        train_losses.append(training_loss)

        current_val_loss = validation_step(model, val_loader, loss_fn, writer, epoch)
        val_losses.append(current_val_loss.item())

        if current_val_loss < best_loss:
            best_epoch = epoch
            best_loss = current_val_loss
            torch.save(model.state_dict(), f"./pretrained/{run_name}_best_model.tar")

    return best_loss, best_epoch, train_losses, val_losses
    

def main():
    dataset = LabelsDataset(TRAIN_GT_LABELS, VAL_GT_LABELS, TRAIN_PRED_LABELS, VAL_PRED_LABELS)

    model = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #Train_mode
    dataset.train_mode()  
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    #Validation_mode
    dataset.val_mode()  
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    loss_fn = nn.MSELoss()
    best_loss, best_epoch, train_losses, val_losses = training_loop(model, train_loader, val_loader, optimizer, RUN_NAME,NUMEPOCHS,loss_fn)

    print(f"Best validation loss: {best_loss} at epoch {best_epoch}")

def test_model(test_dir, model_path, input_size, hidden_sizes, output_size):
    train_path   = sorted(glob(test_dir), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    pred_labels_train = [read_predicted_labels(file) for file in train_path]
    pred_labels_train = torch.tensor(pred_labels_train, dtype=torch.float).squeeze()
    model = MLP(input_size, hidden_sizes, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    output= model(pred_labels_train)
    return output


#TRAIN THE MODEL
# main()


#TEST THE IMAGE
# output = test_model(TEST_PRED_LABELS, MODEL_PATH, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
# print(output)
MODEL_PATH_1 =        "./lr_1e-3_h_200-200_best_model.tar"
MODEL_PATH_2 =        "./lr_1e-3_h_250-250_best_model.tar"
MODEL_PATH_3 =        "./lr_1e-4_h_200-200_best_model.tar"
MODEL_PATH_4 =        "./lr_1e-4_h_250-250_best_model.tar"
MODEL_PATH_5 =        "./lr_1e-5_h_200-200_best_model.tar"
MODEL_PATH_6 =        "./lr_1e-5_h_250-250_best_model.tar"
TEST_PRED_LABELS = './data/test_pred_labels/label_*.txt'

train_path   = sorted(glob(TEST_PRED_LABELS), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
pred_labels_train = [read_predicted_labels(file) for file in train_path]
pred_labels_train = torch.tensor(pred_labels_train, dtype=torch.float).squeeze()
HIDDEN_SIZE1 = [200,200]
HIDDEN_SIZE2 = [250,250]

model1 = MLP(INPUT_SIZE, HIDDEN_SIZE1, OUTPUT_SIZE)
model2 = MLP(INPUT_SIZE, HIDDEN_SIZE2, OUTPUT_SIZE)
model3 = MLP(INPUT_SIZE, HIDDEN_SIZE1, OUTPUT_SIZE)
model4 = MLP(INPUT_SIZE, HIDDEN_SIZE2, OUTPUT_SIZE)
model5 = MLP(INPUT_SIZE, HIDDEN_SIZE1, OUTPUT_SIZE)
model6 = MLP(INPUT_SIZE, HIDDEN_SIZE2, OUTPUT_SIZE)

model1.load_state_dict(torch.load(MODEL_PATH_1))
model2.load_state_dict(torch.load(MODEL_PATH_2))
model3.load_state_dict(torch.load(MODEL_PATH_3))
model4.load_state_dict(torch.load(MODEL_PATH_4))
model5.load_state_dict(torch.load(MODEL_PATH_5))
model6.load_state_dict(torch.load(MODEL_PATH_6))

model1.eval() 
model2.eval() 
model3.eval() 
model4.eval() 
model5.eval() 
model6.eval() 

output1= model1(pred_labels_train)
output2= model2(pred_labels_train)
output3= model3(pred_labels_train)
output4= model4(pred_labels_train)
output5= model5(pred_labels_train)
output6= model6(pred_labels_train)

l=[output1,output2,output3,output4,output5,output6]
concatenated_tensor = torch.stack(l)
mean_tensor = torch.mean(concatenated_tensor, dim=0)
std_tensor = torch.std(concatenated_tensor, dim=0)
print(f"Media por dimensión: {mean_tensor}, Desviación estándar por dimensión: {std_tensor}")


