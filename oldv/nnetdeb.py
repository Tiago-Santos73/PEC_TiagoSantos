
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
import tkinter as tk
import read_nodes
import os
from tkinter import ttk
import random
import numpy as np


class Plate(torch.nn.Module):
    def __init__(self, root, params, bc, ic,output, verbose = False):
        super(Plate,self).__init__()
        self.parameters = params
        self.bc = bc  # Instância da GUI das condições de fronteira
        self.ic = ic  # Instância da GUI das condições iniciais
        self.xt_pde = None
        self.xt_bc = None
        self.loader = None
        self.output = output
        self.root = root

        self.filename = self.parameters['nodes file']
        tmax=float('-inf')
        tmin =float('inf')
        
        # for key in self.bc.entry_values:
        # if (self.bc.entry_values[key]) != "":
        for key, var in self.bc.var_dict.items():
                if var.get() != 3:
                    if float(self.bc.entry_values[key]) < tmin:
                        tmin = float(self.bc.entry_values[key])
                    if float(self.bc.entry_values[key]) > tmax:
                        tmax = float(self.bc.entry_values[key])
        
        for key in self.ic.ic_values:
            if float(self.ic.ic_values[key]) < tmin:
                tmin = float(self.ic.ic_values[key])
            if float(self.ic.ic_values[key]) > tmax:
                tmax = float(self.ic.ic_values[key])
        
        self.bc_adim = {}
        self.ic_adim = {}

        for key, var in self.bc.var_dict.items():
            if var.get() != 3:
                self.bc_adim[key] = ((float(self.bc.entry_values[key])-tmin)/(tmax-tmin)) * (0.5-0.25)+0.25##### 0.75

        for key in self.ic.ic_values:
            self.ic_adim[key] = ((float(self.ic.ic_values[key])-tmin)/(tmax-tmin)) * (0.5-0.25)+0.25 ##### 0.75

            

        #
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        #
        # Activation function
        self.activation = torch.nn.Tanh
        #
        # Loss function criteria
        self.criteria = torch.nn.MSELoss(reduction='mean')
        #
        # Initialize the neural network
        print("Creating the NN...")
        L = []
        #
        # First hidden layer
        L.append(torch.nn.Linear(self.parameters['inputs'],self.parameters['layers'][0]))
        L.append(self.activation())
        #
        # Internal Hidden layers
        for i in range(len(self.parameters['layers'])-1):
            L.append(torch.nn.Linear(self.parameters['layers'][i], self.parameters['layers'][i+1]))
            L.append(self.activation())
        #
        # Output layer
        L.append(torch.nn.Linear(self.parameters['layers'][-1], self.parameters['outputs']))
        # deploy the neural network
        self.model = torch.nn.Sequential(*L).to(self.device)
        #
        if verbose:
            for name, param in self.model.named_parameters():
                print(name,param.shape)
                print(name, param.requires_grad,"\n")
        #
        # define the optimizer for the model parameters
        self.adam = torch.optim.Adam(params=self.model.parameters(), lr=self.parameters['lr'])
        #
        print("NN Created")

    def define_domain(self):
            print("Defining the domain...")


            """"""""""""""""""""""""""""""""""""""
            # Filtrar as chaves em var_dict que têm valor 3 (PDE)
            pde_keys = [key for key, var in self.bc.var_dict.items() if var.get() == 3]

            
            all_pde_points = []
            for key in pde_keys:
                if key in self.bc.new_sets:
                    points = self.bc.new_sets[key]
                    all_pde_points.append(points)
            
            if all_pde_points:
                all_pde_points_tensor = torch.cat(all_pde_points, dim=0)

                num_points_to_select = int(0.08 * all_pde_points_tensor.shape[0])

                random_indices = random.sample(range(all_pde_points_tensor.shape[0]), num_points_to_select)

                selected_points = all_pde_points_tensor[random_indices]

                self.IC = torch.zeros(size=(selected_points.shape[0], 4))

                # Preencher o tensor IC com os pontos selecionados
                self.IC[:, 0] = selected_points[:, 0]  # Coordenadas x
                self.IC[:, 1] = selected_points[:, 1]  # Coordenadas y
                self.IC[:, 2] = self.parameters['t domain'][0]  # Tempo inicial

                IC_values = torch.zeros(selected_points.shape[0], 1)

                for i, idx in enumerate(random_indices):
                    for key in pde_keys:
                        if key in self.bc.new_sets and torch.all(all_pde_points_tensor[idx, :2] == self.bc.new_sets[key][:, :2], dim=1).any():
                            IC_values[i] = self.ic_adim[key]
                            break
                
                self.IC[:, 3] = IC_values.squeeze()

            
            for key in pde_keys:
                if key in self.bc.new_sets:
                    global Temp0
                    Temp0 = self.ic_adim[key]

          
            # # Inicializando uma lista para armazenar todos os tensores
            bc_tensors = [self.IC]  # Comece com self.IC como primeiro tensor

          
            self.xt = torch.cat(bc_tensors, dim=0)

            #criar dominio com tempos != 0
            all_points = []
            for key, points in self.bc.new_sets.items():
                
                num_points = points.shape[0]
                
                t_values = torch.empty((num_points, 1)).uniform_(self.parameters['t domain'][0],self.parameters['t domain'][1])
                
                temp_values = torch.full((num_points, 1), Temp0)
                
                new_data = torch.cat((points[:, :2], t_values, temp_values), dim=1)
                
                all_points.append(new_data)
            
            new_points_tensor = torch.cat(all_points, dim=0) if all_points else torch.empty((0, 4))
            
            self.xt = torch.cat((self.xt, new_points_tensor), dim=0)
            
            self.xt = self.xt.to(torch.float32)

            # torch.set_printoptions(threshold=5000)
            # print(self.xt)  
            self.output.insert(tk.END, "Domínio criado com sucesso!\n")


            print("Dataset created with {:,d} elements\n".format(self.xt.shape[0]))
            # Create the dataloader
            print("Creating the DATALOADER...")
            self.loader = torch.utils.data.DataLoader(
                list(self.xt),
                batch_size=self.parameters['batch size'],
                shuffle=True
                )
            #
            xt_numpy = self.xt.cpu().numpy()

            #Guardar num ficheiro .txt (com tabulação ou vírgula)
            np.savetxt("dominio_gerado_deb.txt", xt_numpy, delimiter="\t", header="x\ty\tt\ttemp", comments="")
            print("Dataloader created\n")
    

    def trainPINN(self):
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Progresso")
        progress_window.geometry("400x100")

        label = tk.Label(progress_window, text="Treino a decorrer...")
        label.pack(pady=5)

        progress_bar = ttk.Progressbar(
        progress_window,
        orient="horizontal",
        length=300,
        mode="determinate"
        )
        progress_bar.pack(pady=10)

        status_label = tk.Label(progress_window, text="Epoch: 0")
        status_label.pack()

        progress_window.update()

        # set the model into training mode
        self.model.train()
        # create the train loss database
        self.train_loss = []
        # define the databases for the loss function
        self.train_loss_IC = []
        self.train_loss_PDE = []

        
        for key in self.bc.var_dict.keys():
            if self.bc.var_dict[key].get() != 3:  # Ignora chaves do tipo PDE
                setattr(self, f"train_loss_BC_{key}", [])
        #
        t0 = time.time()
        for i in range(self.parameters['epochs']):
            
            
            progress_bar["value"] = (i + 1) / self.parameters['epochs'] * 100
            status_label.config(text=f"Epoch: {i + 1}/{self.parameters['epochs']}")
            progress_window.update()
            
            
            epoch_time = time.time()
            #
            print("Epoch {}/{}".format(i+1, self.parameters['epochs']))
            #
            for data in tqdm(self.loader, ncols=100, disable=True):
                # reset the optimizer gradient
                self.adam.zero_grad()
                # get the loss function value
                train_loss = self.lossFunction(dataset=data)
                # calculate the gradients
                train_loss.backward()
                # calculate and update the new parameters
                self.adam.step()
            # record the loss function
            self.train_loss.append(train_loss.item())
            #
            # print the information related to the epoch
            last_loss = self.train_loss[-1]

            # Inicializa a string de perda com os valores fixos
            loss_str = "Loss: {:.2e} - L0: {:.2e}".format(last_loss, self.train_loss_IC[-1])

            # Adiciona dinamicamente os termos de BC
            for key in self.bc.var_dict.keys():
                if self.bc.var_dict[key].get() != 3:  # Ignorar chaves do tipo PDE
                    loss_list_key = f"train_loss_BC_{key}"
                    if hasattr(self, loss_list_key) and getattr(self, loss_list_key):  # Verifica se a lista existe e não está vazia
                        loss_str += " - L{}: {:.2e}".format(key, getattr(self, loss_list_key)[-1])

            # Adiciona a perda do PDE
            loss_str += " - PDE: {:.2e}".format(self.train_loss_PDE[-1])

            # Imprime o resultado
            print(loss_str)

            #
            print("Epoch time: {:.2f} seconds\n".format(time.time() - epoch_time))
            #
            if i == 110:
                exit()

        progress_window.destroy()

        tf = time.time()
        print("Training took: {:.2f} seconds\n".format(tf-t0))
        # save the data
        print("Recording data...")
        #
        filename = os.path.splitext(self.filename)[0] + '_losses.h5'
        f = h5py.File(filename, "w")
        f.create_dataset("loss", data=self.train_loss, compression="gzip")
        f.create_dataset("loss IC", data=self.train_loss_IC, compression="gzip")

        for key in self.bc.var_dict.keys():
            if self.bc.var_dict[key].get() != 3:  # Ignora chaves do tipo PDE
                loss_list_key = f"train_loss_BC_{key}"
        
        # Tenta acessar a lista de perdas dinamicamente
                loss_list = getattr(self, loss_list_key, None)
        
        # Se a lista de perdas existir e não estiver vazia, crie o dataset
                if loss_list is not None and len(loss_list) > 0:
                    f.create_dataset(f"loss BC {key}", data=loss_list, compression="gzip")


        f.create_dataset("loss PDE", data=self.train_loss_PDE, compression="gzip")
        f.flush()
        f.close()
        print("Recording data...DONE!!!")
        #
        filename = os.path.splitext(self.filename)[0] + "_model.pth"
        torch.save(self.model, filename)
        print("Model saved...")

        # plt.plot(self.train_loss,label="Loss")
        # plt.yscale('log')
        # plt.grid(True, which='both')
        # plt.savefig("loss.png")

        self.output.insert(tk.END, f"Treino completo em {tf-t0:.2f} segundos\n")
        #self.output.insert(tk.END, "Treino Executado com sucesso!\n")



    def lossFunction(self, dataset):
        # variables index
        X = 0
        Y = 1
        t = 2
        T0_ = 3
        #
        #***************************************************************
        # filter the initial conditions
        print("\n--- Initial Conditions ---")
        xt_0_idx = torch.where(dataset[:,t] == self.parameters['t domain'][0])[0]
        print(f"Shape of xt_0_idx: {xt_0_idx.shape}")
        print(f"dtype of xt_0_idx: {xt_0_idx.dtype}")
        print(f"Indices of initial conditions: {xt_0_idx}")
        if xt_0_idx.shape[0] != 0:
            # build the initial conditions dataset
            xt_0 = torch.index_select(dataset, dim=0, index=xt_0_idx).to(self.device)
            print(f"Shape of xt_0: {xt_0.shape}")
            print(f"dtype of xt_0: {xt_0.dtype}")
            print(f"First few rows of xt_0:\n{xt_0[:5]}")
            # build the initial conditions
            T0 = torch.index_select(dataset[:,[T0_]], dim=0, index=xt_0_idx).to(self.device)
            print(f"Shape of T0: {T0.shape}")
            print(f"dtype of T0: {T0.dtype}")
            print(f"First few values of T0:\n{T0[:5]}")
            #
            # evaluate the initial condition loss
            L0 = self.criteria(self.model(xt_0), T0)
            print(f"Initial condition loss (L0): {L0.item()}")
            print(f"dtype of L0: {L0.dtype}")
            # save the indices
            pde_indices = xt_0_idx.clone()
            print(f"Shape of pde_indices after IC: {pde_indices.shape}")
            print(f"dtype of pde_indices after IC: {pde_indices.dtype}")
            print(f"Indices in pde_indices after IC: {pde_indices}")
        else:
            l0 = torch.zeros(size=(1,1)).to(self.device)
            L0 = self.criteria(l0,l0)
            print("No initial condition points found. L0 set to zero.")
            print(f"dtype of L0 (zero case): {L0.dtype}")
            pde_indices = torch.empty(0, dtype=torch.int64) # Initialize as empty
            print(f"Shape of pde_indices (empty): {pde_indices.shape}")
            print(f"dtype of pde_indices (empty): {pde_indices.dtype}")
        # save the value
        self.train_loss_IC.append(L0.item())
        print(f"train_loss_IC: {self.train_loss_IC}")
        print(f"dtype of train_loss_IC item: {type(self.train_loss_IC[-1]) if self.train_loss_IC else None}")

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        for key in self.bc.var_dict.keys():
            print(f"\n--- Processing BC Key: {key} ---")
            print(f"Condition Type: {self.bc.var_dict[key].get()}")
            if self.bc.var_dict[key].get() != 3:  # Ignorar chaves do tipo PDE
                # Obter os pontos de fronteira definidos na BCGUI
                if key in self.bc.new_sets:
                    points = self.bc.new_sets[key]
                    print(f"BC Points for key '{key}':\n{points}")
                    print(f"dtype of BC Points: {points.dtype}")
                    print(f"BC Points shape: {points.shape}")
                    if points.numel() > 0 and points.ndim > 1 and points.shape[1] >= 2:
                        print(f"dataset[:, X].unsqueeze(1) shape: {dataset[:, X].unsqueeze(1).shape}")
                        print(f"dtype of dataset[:, X].unsqueeze(1): {dataset[:, X].unsqueeze(1).dtype}")
                        print(f"points[:, 0].unsqueeze(0) shape: {points[:, 0].unsqueeze(0).shape}")
                        print(f"dtype of points[:, 0].unsqueeze(0): {points[:, 0].unsqueeze(0).dtype}")
                        x_match = (dataset[:, X].unsqueeze(1) == points[:, 0].unsqueeze(0))
                        print(f"X Match (boolean):\n{x_match}")
                        print(f"dtype of X Match: {x_match.dtype}")
                        print(f"dataset[:, Y].unsqueeze(1) shape: {dataset[:, Y].unsqueeze(1).shape}")
                        print(f"dtype of dataset[:, Y].unsqueeze(1): {dataset[:, Y].unsqueeze(1).dtype}")
                        print(f"points[:, 1].unsqueeze(0) shape: {points[:, 1].unsqueeze(0).shape}")
                        print(f"dtype of points[:, 1].unsqueeze(0): {points[:, 1].unsqueeze(0).dtype}")
                        y_match = (dataset[:, Y].unsqueeze(1) == points[:, 1].unsqueeze(0))
                        print(f"Y Match (boolean):\n{y_match}")
                        print(f"dtype of Y Match: {y_match.dtype}")

                        # Find indices where both x and y match
                        xt_T_idx = torch.where(torch.all(torch.stack((x_match, y_match), dim=-1), dim=-1))[0]
                        print(f"xt_T_idx shape: {xt_T_idx.shape}")
                        print(f"dtype of xt_T_idx: {xt_T_idx.dtype}")
                        print(f"xt_T_idx elements: {xt_T_idx}")

                        if xt_T_idx.shape[0] != 0:
                            xt_T_tensor = torch.index_select(dataset, dim=0, index=xt_T_idx).to(self.device)
                            print(f"xt_{key} shape: {xt_T_tensor.shape}")
                            print(f"dtype of xt_{key}: {xt_T_tensor.dtype}")
                            print(f"xt_{key} first few rows:\n{xt_T_tensor[:5]}")

                            setattr(self, f"xt_{key}", xt_T_tensor)

                            T_target = torch.ones(size=(xt_T_tensor.shape[0], 1), dtype= torch.float32) * self.bc_adim[key]
                            print(f"dtype of T_target: {T_target.dtype}")
                            setattr(self, f"T_{key}", T_target.to(self.device))
                            print(f"T_{key} shape: {getattr(self, f'T_{key}').shape}")
                            print(f"dtype of T_{key}: {getattr(self, f'T_{key}').dtype}")
                            print(f"T_{key} first few values:\n{getattr(self, f'T_{key}')[:5]}")

                            model_output_bc = self.model(getattr(self, f"xt_{key}"))
                            print(f"dtype of model output (BC): {model_output_bc.dtype}")
                            loss_bc = self.criteria(model_output_bc, getattr(self, f"T_{key}"))
                            print(f"dtype of loss_bc: {loss_bc.dtype}")
                            setattr(self, f"L_{key}", loss_bc)
                            print(f"L_{key}: {getattr(self, f'L_{key}').item()}")
                            print(f"dtype of L_{key}.item(): {type(getattr(self, f'L_{key}').item())}")

                            pde_indices = torch.cat((pde_indices, xt_T_idx.clone()), dim=0)
                            print(f"pde_indices shape after {key}: {pde_indices.shape}")
                            print(f"dtype of pde_indices after {key}: {pde_indices.dtype}")
                            print(f"pde_indices elements after {key}: {pde_indices}")
                        else:
                            ll = torch.zeros(size=(1, 1)).to(self.device)
                            print(f"dtype of ll (no match): {ll.dtype}")
                            setattr(self, f"L_{key}", self.criteria(ll, ll))
                            print(f"No matching points found for BC key '{key}'. Loss set to zero.")
                    else:
                        print(f"Warning: 'points' for key '{key}' is empty or has incorrect dimensions.")

                train_loss_BC_key = f"train_loss_BC_{key}"
                if not hasattr(self, train_loss_BC_key):
                    setattr(self, train_loss_BC_key, [])
                if hasattr(self, f"L_{key}"):
                    getattr(self, train_loss_BC_key).append(getattr(self, f"L_{key}").item())
                    print(f"dtype of train_loss_BC_{key} item: {type(getattr(self, train_loss_BC_key)[-1])}")
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        #***************************************************************
        # filter the domain conditions (all points)
        print("\n--- PDE Loss ---")
        full_dataset_index = torch.arange(end=dataset.shape[0])
        print(f"Shape of full_dataset_index: {full_dataset_index.shape}")
        print(f"dtype of full_dataset_index: {full_dataset_index.dtype}")
        #
        mask = ~torch.isin(full_dataset_index, pde_indices)
        print(f"Shape of mask: {mask.shape}")
        print(f"dtype of mask: {mask.dtype}")
        print(f"First few elements of mask:\n{mask[:10]}")
        #
        filtered_indices = full_dataset_index[mask]
        print(f"Shape of filtered_indices: {filtered_indices.shape}")
        print(f"dtype of filtered_indices: {filtered_indices.dtype}")
        print(f"First few indices for PDE: {filtered_indices[:10]}")
        #
        xt_domain = torch.index_select(dataset, dim=0, index=filtered_indices).to(self.device)
        print(f"Shape of xt_domain: {xt_domain.shape}")
        print(f"dtype of xt_domain: {xt_domain.dtype}")
        print(f"First few rows of xt_domain:\n{xt_domain[:5]}")
        xt_domain.requires_grad_(True)
        #
        x_pde = xt_domain[:,[X]]
        y_pde = xt_domain[:,[Y]]
        t_pde = xt_domain[:,[t]]
        T0_pde = xt_domain[:,[T0_]]
        print(f"Shape of x_pde: {x_pde.shape}")
        print(f"Shape of y_pde: {y_pde.shape}")
        print(f"Shape of t_pde: {t_pde.shape}")
        T_est = self.model(torch.concat((x_pde,y_pde,t_pde,T0_pde), axis=1))
        #
        # compute the derivatives
        dTdt = torch.autograd.grad(outputs=T_est, inputs=t_pde, grad_outputs=torch.ones_like(t_pde), create_graph=True, retain_graph=True, allow_unused=True)[0]
        dTdx = torch.autograd.grad(outputs=T_est, inputs=x_pde, grad_outputs=torch.ones_like(x_pde), create_graph=True, retain_graph=True, allow_unused=True)[0]
        dTdy = torch.autograd.grad(outputs=T_est, inputs=y_pde, grad_outputs=torch.ones_like(y_pde), create_graph=True, retain_graph=True, allow_unused=True)[0]
        dTdxx = torch.autograd.grad(outputs=dTdx, inputs=x_pde, grad_outputs=torch.ones_like(x_pde), create_graph=True, retain_graph=True, allow_unused=True)[0]
        dTdyy = torch.autograd.grad(outputs=dTdy, inputs=y_pde, grad_outputs=torch.ones_like(y_pde), create_graph=True, retain_graph=True, allow_unused=True)[0]
        #
        L_PDE = self.criteria(dTdt, (self.parameters['alphax'] * dTdxx) + (self.parameters['alphay'] * dTdyy))
        #
        self.train_loss_PDE.append(L_PDE.item())
        # compute the total loss
        total_loss = L0 + sum(getattr(self, f"L_{key}") for key in self.bc.var_dict.keys() if hasattr(self, f"L_{key}")) + L_PDE

        # return the total loss
        return total_loss