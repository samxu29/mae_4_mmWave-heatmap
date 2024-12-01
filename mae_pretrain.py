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

# Add these imports at the top
import numpy as np
from glob import glob
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import os


# Add custom dataset class
class DopplerDataset(Dataset):
    def __init__(self, data_path):
        all_files = glob(f"{data_path}/doppler/all/*.npy")
        self.file_paths = []
        
        for file_path in all_files:
            try:
                # Quick validation check
                data = np.load(file_path)
                if data.shape == (256, 128):
                    self.file_paths.append(file_path)
            except:
                continue
                
        print(f"Found {len(self.file_paths)} valid files out of {len(all_files)} total files")
        
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
            return data, 0
        except Exception as e:
            print(f"Error loading file {self.file_paths[idx]}: {str(e)}")
            # Return a zero tensor as fallback
            return torch.zeros(1, 256, 128), 0
        
def visualize_random_reconstruction(model, dataset, device, index=None, save_dir='demo'):
    model.eval()
    with torch.no_grad():
        # Get random index if not specified
        if index is None:
            index = np.random.randint(0, len(dataset))
            
        # Get input image
        input_img, _ = dataset[index]
        input_img = input_img.unsqueeze(0).to(device)
        
        # Get reconstruction
        predicted_img, mask = model(input_img)
        
        # Combine masked input, reconstruction and original
        reconstructed = predicted_img * mask + input_img * (1 - mask)
        masked = input_img * (1 - mask)
        
        # Move tensors to CPU and convert to numpy
        input_img = input_img.cpu().numpy()[0,0]
        masked = masked.cpu().numpy()[0,0]
        reconstructed = reconstructed.cpu().numpy()[0,0]
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a figure with subplots
        plt.figure(figsize=(15,4))
        
        plt.subplot(131)
        plt.imshow(input_img, cmap='viridis')
        plt.title('Original')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.subplot(132)
        plt.imshow(masked, cmap='viridis')
        plt.title('Masked Input')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.subplot(133)
        plt.imshow(reconstructed, cmap='viridis')
        plt.title('Reconstruction')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'pretrain_reconstruction_{index}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=20)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae_heatmap.pt')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = DopplerDataset('/home/sxu7/data/heatmap/simulation-KIT_heatmap-doppler')
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    writer = SummaryWriter(os.path.join('logs', 'heatmap', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(image_size=(256, 128), patch_size=(128, 1), mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for batch_idx, (img, label) in enumerate(tqdm(iter(dataloader))):
            try:
                step_count += 1
                img = img.to(device)
                predicted_img, mask = model(img)
                loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
                loss.backward()
                if step_count % steps_per_update == 0:
                    optim.step()
                    optim.zero_grad()
                losses.append(loss.item())
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                continue        
            
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_imgs = torch.stack([train_dataset[i][0] for i in range(16)])
            val_imgs = val_imgs.to(device)
            predicted_val_img, mask = model(val_imgs)
            predicted_val_img = predicted_val_img * mask + val_imgs * (1 - mask)
            
            # Visualize only first channel for simplicity
            img = torch.cat([val_imgs[:,:1] * (1 - mask), predicted_val_img[:,:1], val_imgs[:,:1]], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_channel1', img, global_step=e)        
        ''' save model '''
        torch.save(model, args.model_path)
        
    model = torch.load(args.model_path)
    visualize_random_reconstruction(model, train_dataset, device)
