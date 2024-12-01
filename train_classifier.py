import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed

import numpy as np
from glob import glob
from torch.utils.data import Dataset

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

class DopplerClassificationDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = os.path.join(data_path, split, 'doppler')
        self.classes = sorted([d for d in os.listdir(self.data_path) 
                             if os.path.isdir(os.path.join(self.data_path, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.file_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            files = glob(os.path.join(class_path, "*.npy"))
            
            for file_path in files:
                try:
                    data = np.load(file_path)
                    if data.shape == (256, 128):
                        self.file_paths.append(file_path)
                        self.labels.append(self.class_to_idx[class_name])
                except:
                    continue
                    
        print(f"Found {len(self.file_paths)} valid files in {split} split")
        print(f"Number of classes: {len(self.classes)}")
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        try:
            data = np.load(self.file_paths[idx])
            if data.shape != (256, 128):
                raise ValueError(f"Invalid shape {data.shape}")
            data = data.reshape(1, 256, 128)
            data = torch.from_numpy(data).float()
            data = (data - data.mean()) / (data.std() + 1e-6)
            return data, self.labels[idx]
        except Exception as e:
            print(f"Error loading file {self.file_paths[idx]}: {str(e)}")
            return torch.zeros(1, 256, 128), self.labels[idx]


def plot_confusion_matrix(model, dataloader, class_names, device, save_path='demo'):
    # Create demo directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Get predictions and true labels
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(dataloader, desc="Generating confusion matrix"):
            img = img.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=20)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default='vit-t-classifier-from_scratch.pt')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    # Replace the dataset loading section
    data_path = '/home/sxu7/data/heatmap/recorded_heatmap-doppler'
    train_dataset = DopplerClassificationDataset(data_path, split='train')
    val_dataset = DopplerClassificationDataset(data_path, split='eval')
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, 
                                                  shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, 
                                               shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Update the classifier initialization with correct number of classes
    num_classes = len(train_dataset.classes)
    if args.pretrained_model_path is not None:
        model = torch.load(args.pretrained_model_path, map_location='cpu')
        writer = SummaryWriter(os.path.join('logs', 'heatmap', 'pretrain-cls'))
    else:
        model = MAE_ViT(image_size=(256, 128), patch_size=(128, 1))
        writer = SummaryWriter(os.path.join('logs', 'heatmap', 'scratch-cls'))
    model = ViT_Classifier(model.encoder, num_classes=num_classes).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    best_val_acc = 0
    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(val_dataloader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch!')       
            torch.save(model, args.output_model_path)

        writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=e)
        writer.add_scalars('cls/acc', {'train' : avg_train_acc, 'val' : avg_val_acc}, global_step=e)
    
    best_model = torch.load(args.output_model_path)
    plot_confusion_matrix(
        model=best_model,
        dataloader=val_dataloader,
        class_names=val_dataset.classes,
        device=device
    )    
