#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random
import sys
import os
# pytorch:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score
# FTT model:
import rtdl
import zero
from rtdl import FTTransformer
# ray tune
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from math import ceil
# save output:
import copy

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
        self.y = torch.tensor(y.values, dtype=torch.long)
        self.n_samples = xy.shape[0]
    def __getitem__(self, index): # allow idxing
        return self.x_num[index], self.x_cat[index], self.y[index]     
    def __len__(self):
        return self.n_samples  # num of row
def load_data(cv_set, batch_size, abs_path):
    train = pd.read_pickle(f'{abs_path}/data/v{cv_set}_train.pkl')
    validate = pd.read_pickle(f'{abs_path}/data/v{cv_set}_validate.pkl')
    train_loader = DataLoader(dataset=MyDataset(train), batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(dataset=MyDataset(validate), batch_size=batch_size, shuffle=True)
    return train_loader, validate_loader

def train_forRaytune(search_space, cv_set, num_epochs, abs_path):
    """
    cv_set = 1, 2, 3; specify the set to use in 3-fold CV
    """
    # setup:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FTTransformer.make_default(
        n_num_features=328,
        cat_cardinalities=[21, 2, 15],
        d_out=6,
        last_layer_query_idx=[-1]    
        ).to(device)
    loss_fn = F.cross_entropy
    # setup from hyperparams:
    optimizer = torch.optim.AdamW(model.parameters(), lr=search_space["lr"], weight_decay=search_space["wd"])
    train_loader, validate_loader = load_data(cv_set, search_space["batch_size"], abs_path) # load data
    best_acc, best_bac, best_loss = 0.0, 0.0, float('inf')   # starting buffer to find optimal bac/acc
    for epoch in range(num_epochs):
        # Training:
        train_loss = 0
        y_true = torch.tensor([]).to(device) # true label
        y_pred = torch.tensor([]).to(device) # predicted label
        for i, (x_num, x_cat, y) in enumerate(train_loader): # training
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)
            y_true = torch.cat([y_true, y], dim=0)
            outputs = model(x_num, x_cat)  # forward
            loss = loss_fn(outputs, y) 
            train_loss += loss.item()
            optimizer.zero_grad()  # empty the gradient from previous step
            loss.backward()        # backpropagation: get gradients
            optimizer.step()       # gradient descent
            _, preds = torch.max(outputs, 1) # raw value & col idx of prediction
            y_pred = torch.cat([y_pred, preds], dim=0)
        # training evaluation:
        train_loss = train_loss / len(train_loader)  # mean of loss in one epoch
        y_pred = y_pred.detach().cpu().numpy()       # turn to array for calculation
        y_true = y_true.detach().cpu().numpy()
        train_acc = np.mean(y_pred==y_true)
        train_bac = balanced_accuracy_score(y_true, y_pred)
        # print(f"train ACC: {train_acc:.4f}, BAC:{train_bac:.4f}")
        # Evaluation:
        model.eval()               
        with torch.no_grad():      # disable backpropagation in validation set
            val_loss = 0
            y_true = torch.tensor([]).to(device) # true label
            y_pred = torch.tensor([]).to(device) # predicted label
            for i, (x_num, x_cat, y) in enumerate(validate_loader):
                x_num = x_num.to(device)
                x_cat = x_cat.to(device)
                y = y.to(device)
                y_true = torch.cat([y_true, y], dim=0)
                outputs = model(x_num, x_cat)    # predicted prob. of validation set
                loss = loss_fn(outputs, y)       # validation loss
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1) 
                y_pred = torch.cat([y_pred, preds], dim=0)
            val_loss = val_loss / len(validate_loader)
            y_pred = y_pred.detach().cpu().numpy()       
            y_true = y_true.detach().cpu().numpy()
            val_acc = np.mean(y_pred==y_true)
            val_bac = balanced_accuracy_score(y_true, y_pred)
            # print(f"validate ACC: {val_acc:.4f}, BAC:{val_bac:.4f}")
        # metrics to report:
        tune.report(train_loss=train_loss, train_acc=train_acc, train_bac=train_bac, 
                    val_loss=val_loss, val_acc=val_acc, val_bac=val_bac
                                )
        if val_loss < best_loss:   # continue training
            best_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1  # early-stopping
            if early_stop_count == (num_epochs//10):
                print(f"Validation loss hasn't improved in {num_epochs//10} epochs. Stopping early")
                break        
        model.train()  # turn back to training 


def main():
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    abs_path = os.path.abspath("")  # get the absolute path because ray tune couldn't use relative path
    num_epochs = 5  # 100
    num_trials = 5
    cv_set = sys.argv[1]
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),   # learning rate
        "wd": tune.loguniform(1e-4, 1e-1),   # wegith decay
        "batch_size": tune.choice([16, 32, 64, 128])
    }
    scheduler = ASHAScheduler(
        metric="val_loss",            # metric for evaluation
        mode="min",                   # minimize the metric
        max_t=num_epochs,             # max epochs in each trial
        grace_period=ceil(num_epochs/5),   # the interval of doing early-stopping
        reduction_factor=2)           # stop half for every grace_period  
    reporter = CLIReporter(metric_columns=["training_iteration", "train_loss","train_acc", 
                                           "train_bac","val_loss", "val_acc","val_bac"])
    result = tune.run(
        partial(train_forRaytune, cv_set=cv_set, num_epochs=num_epochs, abs_path=abs_path),
        resources_per_trial={"cpu": 8, "gpu": 0.5},
        config=search_space,
        num_samples=num_trials,       # num of total random trials to run
        scheduler=scheduler,
        progress_reporter=reporter  
        )  # checkpoint_at_end=True
    print(result.results_df)
    torch.save(result.results_df, f"cv_{cv_set}_RayTune_df.pt")
    torch.save(result, f"cv_{cv_set}_RayTune.pt")
    best_trial = result.get_best_trial("val_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}' best result: {best_trial.last_result}")
if __name__ == "__main__":
    main()
