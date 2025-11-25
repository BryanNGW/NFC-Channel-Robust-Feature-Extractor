from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import CosineEmbeddingLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from pytorch_metric_learning.losses import TripletMarginLoss, AngularLoss, NPairsLoss
import torch.nn.functional as F
import numpy as np
import math
import re
import random
import os
import time
import json
from torchmetrics.classification  import BinaryF1Score
# from torchmetrics.functional import precision_recall
from sklearn.metrics import precision_recall_fscore_support ,f1_score
from pytorch_metric_learning.miners import TripletMarginMiner
from sklearn.preprocessing import OneHotEncoder
from typing import Optional, Dict, Iterable, Tuple

def check_device(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")

def fixed_seed(myseed:int):

    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def load_parameters(model, path, optimizer=None, epoch:int=0):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path)
    model.load_state_dict(param)

    if optimizer != None:
        optimizer.load_state_dict(torch.load(os.path.join(f"./save/optimizer_{epoch}.pt")))

    print("End of loading !!!")

def train(model, train_loader, val_loader, initial_epoch: int, num_epoch: int, save_path: str, device, criterion, optimizer, train_miners, val_miners, scheduler, patience: int = 1000, exp_name=""):
    import time
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from pytorch_metric_learning.losses import TripletMarginLoss
    from pytorch_metric_learning.miners import TripletMarginMiner

    start_train = time.time()
    writer = SummaryWriter(os.path.join("./runs", exp_name))
    os.makedirs(save_path, exist_ok=True)

    if isinstance(train_loader.dataset, torch.utils.data.Subset):
        base_dataset = train_loader.dataset.dataset
    else:
        base_dataset = train_loader.dataset

    class_names = sorted(list(set([label for _, label, _ in base_dataset.samples])))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    best_val_epoch = 0
    best_loss = 999
    no_improve_counter = 0

    margin = 0.8
    margin_increment = 0.05
    max_margin = 0.8
    dynamic_margin_patience = 200

    for epoch in range(initial_epoch, num_epoch):
        print(f'Epoch = {epoch}')
        start_time = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("train/margin", float(margin), global_step=epoch)
        writer.add_scalar("train/learning_rate", float(current_lr), global_step=epoch)
        print(f"Current Learning Rate = {current_lr:.6f}")
        print(f"Current Margin = {margin:.2f}")

        model.train()
        print(f"Model is on device: {next(model.parameters()).device}")

        train_loss = 0.0
        num_train_triplet = 0

        for i, (CIS, label, _) in enumerate(tqdm(train_loader)):
            if isinstance(label, list):
                label = torch.tensor([class_to_idx[l] for l in label]).to(device)
            else:
                label = torch.tensor([class_to_idx[label]]).to(device)

            if isinstance(CIS, list):
                anchor = torch.stack([torch.tensor(x) for x in CIS]).to(device)
            else:
                anchor = CIS.to(device)

            outputs = model(anchor)
            triplets = train_miners(outputs, label)
            anchor_idx, positive_idx, negative_idx = triplets[0], triplets[1], triplets[2]

            if anchor_idx.numel() == 0:
                print(f"⚠️ Skipping training batch {i} in epoch {epoch} — no valid triplets")
                continue

            num_triplet = anchor_idx.size(0)
            loss, ap_dist, an_dist = criterion(outputs, label, (anchor_idx, positive_idx, negative_idx))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_train_triplet += num_triplet

        train_loss /= len(train_loader)
        writer.add_scalar("train/loss", float(train_loss), global_step=epoch)
        writer.add_scalar("train/num_triplets", int(num_train_triplet), global_step=epoch)

        with torch.no_grad():
            eval_loss = 0.0
            num_val_triplet = 0

            model.eval()

            for i, (CIS, label, _) in enumerate(tqdm(val_loader)):
                if isinstance(label, list):
                    label = torch.tensor([class_to_idx[l] for l in label]).to(device)
                else:
                    label = torch.tensor([class_to_idx[label]]).to(device)

                if isinstance(CIS, list):
                    anchor = torch.stack([torch.tensor(x) for x in CIS]).to(device)
                else:
                    anchor = CIS.to(device)

                outputs = model(anchor)
                val_triplets = val_miners(outputs, label)
                anchor_idx, positive_idx, negative_idx = val_triplets[0], val_triplets[1], val_triplets[2]

                if anchor_idx.numel() == 0:
                    print(f"⚠️ Skipping validation batch {i} in epoch {epoch} — no valid triplets")
                    continue

                eval_num_triplet = anchor_idx.size(0)
                loss, _, _ = criterion(outputs, label, (anchor_idx, positive_idx, negative_idx))

                eval_loss += loss.item()
                num_val_triplet += eval_num_triplet

            eval_loss /= len(val_loader)
            writer.add_scalar("val/loss", float(eval_loss), global_step=epoch)
            writer.add_scalar("val/num_val_triplets", int(num_val_triplet), global_step=epoch)

            if eval_loss < best_loss:
                best_loss = eval_loss
                best_val_epoch = epoch
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                if no_improve_counter >= dynamic_margin_patience:
                    if margin < max_margin:
                        margin += margin_increment
                        margin = min(margin, max_margin)
                    no_improve_counter = 0
                    criterion = TripletMarginLoss(margin=margin)
                    train_miners = TripletMarginMiner(margin=margin, type_of_triplets='all')
                    val_miners = TripletMarginMiner(margin=margin, type_of_triplets='all')
                    writer.add_scalar("train/margin_change", float(margin), global_step=epoch)
                    print(f"Margin increased to {margin:.2f} after {dynamic_margin_patience} epochs without validation loss improvement.")

        scheduler.step(eval_loss)

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pt'))
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if no_improve_counter >= patience:
            print(f"Validation loss did not improve for {patience} epochs. Early stopping at epoch {epoch}.")
            break

        torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}.pt'))
        torch.save(optimizer.state_dict(), os.path.join(save_path, f'optimizer_{epoch}.pt'))

        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time = end_time - start_train
        print('=' * 24)
        print(f'Time for Epoch: {elapsed_time // 60:.0f} min {elapsed_time % 60:.1f} sec')
        print(f'Total Time: {total_time // 60:.0f} min {total_time % 60:.1f} sec')
        print('=' * 24 + '\n')

    print("End of training!")
    print(f"Best validation loss {best_loss:.6f} at epoch {best_val_epoch}")
    writer.close()

        
    