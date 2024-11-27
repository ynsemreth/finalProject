import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import *
from utils.dataloader import *

N_STEPS = 10

def load_data(args, path):
    train_data = MovingMNIST(args, is_train=True, root=path, n_frames_input=N_STEPS, n_frames_output=N_STEPS, num_objects=[2])
    val_data = MovingMNIST(args, is_train=False, root=path, n_frames_input=N_STEPS, n_frames_output=N_STEPS, num_objects=[2])
    return train_data, val_data

def main(args):
    start_epoch = 1
    path = "./"
    best_loss = float('inf')
    lr = args.lr
    
    model = get_model(args)
    
    ckpt_path = f'./model_ckpt/{args.model}_layer{args.num_layers}_model.pth'
    ckpt_best_path = f'./model_ckpt/{args.model}_layer{args.num_layers}_best_model.pth'
    
    if args.reload:
        start_epoch, lr, optimizer_state_dict = load_checkpoint(model, args, ckpt_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_data, val_data = load_data(args, path)
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.batch_size)
    
    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    if args.reload:
        optimizer.load_state_dict(optimizer_state_dict)
        
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Eğitim Aşaması
        model.train()
        train_losses = []
        train_progress = tqdm(train_loader, desc="Training", leave=False, position=0)
        
        for x, y in train_progress:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_progress.set_postfix(loss=f'{loss.item():.4f}')
        
        train_loss_avg = sum(train_losses) / len(train_losses)
        print(f"  Training Loss: {train_loss_avg:.4f}")
        
        # Doğrulama Aşaması
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_losses = []
            val_progress = tqdm(val_loader, desc="Validation", leave=False, position=1)
            
            with torch.no_grad():
                for x, y in val_progress:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    val_losses.append(loss.item())
                    val_progress.set_postfix(val_loss=f'{loss.item():.4f}')
            
            val_loss_avg = sum(val_losses) / len(val_losses)
            print(f"  Validation Loss: {val_loss_avg:.4f}")
            
            if val_loss_avg < best_loss:
                best_loss = val_loss_avg
                print(f"  Saving best model with validation loss {best_loss:.4f}")
                save_checkpoint(model, optimizer, epoch, ckpt_best_path)
        
        save_checkpoint(model, optimizer, epoch, ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of hidden dim for ConvLSTM layers')
    parser.add_argument('--input_dim', type=int, default=1, help='input channels')
    parser.add_argument('--model', type=str, default='convlstm', help='name of the model')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers')
    parser.add_argument('--frame_num', type=int, default=10, help='number of frames')
    parser.add_argument('--img_size', type=int, default=64, help='image size')
    parser.add_argument('--gpu_num', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--reload', action='store_true', help='reload model')
    args = parser.parse_args()

    main(args)
