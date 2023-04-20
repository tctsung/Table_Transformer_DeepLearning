#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random
# pytorch:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# FTT model:
import rtdl
import zero
from rtdl import FTTransformer
# save output:
import copy
from sklearn.metrics import balanced_accuracy_score
def setup_seed(seed): 
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True
    return None
class MyDataset(Dataset):    # inherit from torch.utils.data.DatasetDataset
    def __init__(self, xy, normalize=False):
        # read data & preprocess
        x_num = xy.iloc[:,4:]
        if normalize:    # min-max normalization for each feature in numeric data
            x_num= x_num.apply(lambda x:(x - x.min()) / (x.max()-x.min() ))
        self.x_num = torch.tensor(np.array(x_num), dtype=torch.float32) # numeric dtype
        x_cat = xy.iloc[:,0:4]
        y = x_cat.pop('disease')
        self.x_cat = torch.tensor(np.array(x_cat), dtype=torch.long)    # categorical dtype
        self.y = torch.tensor(y, dtype=torch.long)
        self.n_samples = xy.shape[0]
    def __getitem__(self, index): # allow idxing
        return self.x_num[index], self.x_cat[index], self.y[index]     
    def __len__(self):
        return self.n_samples  # num of row
def FTTransformer_train(model, train_loader, validate_loader, loss_fn, optimizer, num_epochs, device, verbose=10):
    model = model.to(device)
    save_values = {}  # dictionary to save
    train_steps = len(train_loader)
    best_acc, best_bac = 0.0, 0.0   # starting buffer to find optimal bac/acc
    train_losses, validate_losses = [], [] # to save loss per epoch
    train_acc, validate_acc, train_bac, validate_bac = [], [], [], [] # to save accuracy/BAC per epoch
    for epoch in range(num_epochs):
        loss_accum = 0
        y_true = torch.tensor([]).to(device) # true label
        y_pred = torch.tensor([]).to(device) # predicted label
        for i, (x_num, x_cat, y) in enumerate(train_loader): # training
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)
            y_true = torch.cat([y_true, y], dim=0)
            outputs = model(x_num, x_cat)  # forward
            loss = loss_fn(outputs, y) 
            loss_accum += loss.item()
            optimizer.zero_grad()  # empty the gradient from previous step
            loss.backward()        # backpropagation: get gradients
            optimizer.step()       # gradient descent
            _, preds = torch.max(outputs, 1) # raw value & col idx of prediction
            y_pred = torch.cat([y_pred, preds], dim=0)
            if (i+1) % verbose == 0:
                print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{train_steps}, loss: {loss.item():.4f}")
        # training evaluation:
        loss_accum = loss_accum / train_steps  # mean of loss in one epoch
        y_pred = y_pred.detach().cpu().numpy()       # turn to array for calculation
        y_true = y_true.detach().cpu().numpy()
        acc = np.mean(y_pred==y_true)
        bac = balanced_accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch+1}/{num_epochs}, train ACC: {acc:.4f}, BAC:{bac:.4f}", end="/ ")
        # save info:
        train_acc.append(acc) # save accuracy
        train_bac.append(bac) # save BAC
        train_losses.append(loss_accum)        # save mean loss
        model.eval()               # turn to evaluation mode
        with torch.no_grad():      # disable backpropagation in validation set
            loss_accum = 0
            y_true = torch.tensor([]).to(device) # true label
            y_pred = torch.tensor([]).to(device) # predicted label
            for i, (x_num, x_cat, y) in enumerate(validate_loader):
                x_num = x_num.to(device)
                x_cat = x_cat.to(device)
                y = y.to(device)
                y_true = torch.cat([y_true, y], dim=0)
                outputs = model(x_num, x_cat)    # predicted prob. of validation set
                loss = loss_fn(outputs, y)       # validation loss
                loss_accum += loss.item()
                _, preds = torch.max(outputs, 1) 
                y_pred = torch.cat([y_pred, preds], dim=0)
            loss_accum = loss_accum / len(validate_loader)
            y_pred = y_pred.detach().cpu().numpy()       
            y_true = y_true.detach().cpu().numpy()
            acc = np.mean(y_pred==y_true)
            bac = balanced_accuracy_score(y_true, y_pred)
            print(f"validate ACC: {acc:.4f}, BAC:{bac:.4f}")
            # save validate set info:
            validate_acc.append(acc) # save accuracy
            validate_bac.append(bac) # save BAC
            validate_losses.append(loss_accum)        # save mean loss
        if acc > best_acc:    # use accuracy as criteria for best model
            best_acc = acc    # keep the params with best accuracy
            best_acc_wts = copy.deepcopy(model.state_dict())
        if bac > best_bac:
            best_bac = bac
            best_bac_wts = copy.deepcopy(model.state_dict())
            # add early stopping!!!!!     
        model.train()  # turn back to training mode
    print('Best val Acc: {:4f}'.format(best_acc))
    # model.load_state_dict(best_bac_wts)        # load model with the best weights
    ## save info:
    save_values["best_bac_wts"] = best_bac_wts
    save_values["best_acc_wts"] = best_acc_wts
    save_values["validate_acc"] = validate_acc
    save_values["validate_bac"] = validate_bac
    save_values["validate_losses"] = validate_losses
    save_values["train_acc"] = train_acc
    save_values["train_bac"] = train_bac
    save_values["train_losses"] = train_losses
    return save_values
###  code to run:
setup_seed(7)      # random seed
# setup before training:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")
cat_cardinalities = [21, 2, 15]    # number of unique classes 
n_num_features = 328   
d_out = 6
model = FTTransformer.make_default(
    n_num_features=n_num_features,
    cat_cardinalities=cat_cardinalities,
    d_out=d_out,
    last_layer_query_idx=[-1]    # enhance running efficiency
)
optimizer = torch.optim.AdamW(model.parameters()) # , lr=lr, weight_decay=weight_decay
loss_fn = F.cross_entropy
verbose = 10
# load data:
num_epochs = 30
batch_size = 32
train1 = pd.read_pickle('v1_train.pkl')
validate1 = pd.read_pickle('v1_validate.pkl')
train_loader = DataLoader(dataset=MyDataset(train1), batch_size=batch_size, shuffle=True, num_workers=2)
validate_loader = DataLoader(dataset=MyDataset(validate1), batch_size=batch_size, shuffle=True, num_workers=2)
results = FTTransformer_train(model, train_loader, validate_loader, loss_fn, optimizer, num_epochs, device, verbose)
torch.save(results, 'FTT_default_model.pt')




