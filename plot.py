import torch
torch.set_default_dtype(torch.float32)
SEED = 1616
torch.manual_seed(SEED)
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import time
import json
import os
import tkinter as tk

class PlotLoss:
    def __init__(self,name,output):
        self.name = name
        self.output = output
    def plot_loss(self):
        # Load the h5 file into a pandas DataFrame
        # filename
        
        filename = os.path.splitext(self.name)[0] + '_losses.h5'
        
        #
        with h5py.File(filename, "r") as f:
            #
            plt.figure(figsize=(10, 6))
            plt.xlabel("Epoch")
            plt.ylabel("Loss value")
            #
            title = ""
            #
            i = 0
            for dataset_name in f.keys():
                data = f[dataset_name][:]
                if dataset_name == "loss":
                    mean_last_100_loss = data[-100:].mean()
                #
                plt.plot(data, label = dataset_name)
                #
                if i == 2:
                    title += dataset_name + ": {:.4E}\n".format(data[-100:].mean())
                    i = 0
                else:
                    title += dataset_name + ": {:.4E} - ".format(data[-100:].mean())
                    i += 1
            #
            plt.yscale('log')
            plt.legend()
            plt.title(title)
            plt.grid(which='both',visible=True)
            plot_name = os.path.splitext(self.name)[0] + "_loss.png"
            plt.savefig(plot_name)
        self.output.insert(tk.END, "Plot Executado com sucesso!\n")

        return mean_last_100_loss
        
