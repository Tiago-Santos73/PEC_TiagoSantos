

import torch
torch.set_default_dtype(torch.float32)
SEED = 1616
torch.manual_seed(SEED)
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import time
import tkinter as tk
import os
from tkinter import ttk
import random
import numpy as np


class Plate(torch.nn.Module):
    def __init__(self, root, params, bc, ic,output, verbose = False):
        super(Plate,self).__init__()
        
        self.parameters = params
        self.bc = bc  
        self.ic = ic  
        self.xt_pde = None
        self.xt_bc = None
        self.loader = None
        self.output = output
        self.root = root


        self.filename = self.parameters['nodes file']
        
        #Initialize max and min temperature
        tmax=float('-inf')
        tmin =float('inf')
        
        #Loop over Temperatures to get min and max
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

        #Convert BC to adimensional
        for key, var in self.bc.var_dict.items():
            
            if var.get() != 3:
                self.bc_adim[key] = ((float(self.bc.entry_values[key])-tmin)/(tmax-tmin)) * (self.parameters['adim dom'][1]-self.parameters['adim dom'][0])+self.parameters['adim dom'][0]

        #Convert IC to adimensional
        for key in self.ic.ic_values:
            self.ic_adim[key] = ((float(self.ic.ic_values[key])-tmin)/(tmax-tmin)) * (self.parameters['adim dom'][1]-self.parameters['adim dom'][0])+self.parameters['adim dom'][0]

            

        
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


            #############################################
            # Get PDE Keys
            pde_keys = [key for key, var in self.bc.var_dict.items() if var.get() == 3]

            all_pde_points = []

            #Get PDE Points
            for key in pde_keys:
                if key in self.bc.new_sets:
                    points = self.bc.new_sets[key]
                    all_pde_points.append(points)
            
            if all_pde_points:

                 # Concatenate all PDE points
                all_pde_points_tensor = torch.cat(all_pde_points, dim=0)


                # Select 8% of the points randomly from the PDE points
                num_points_to_select = int(0.08 * all_pde_points_tensor.shape[0])
                random_indices = random.sample(range(all_pde_points_tensor.shape[0]), num_points_to_select)
                selected_points = all_pde_points_tensor[random_indices]


                # Initialize IC
                self.IC = torch.zeros(size=(selected_points.shape[0], 4))


                # compile IC
                self.IC[:, 0] = selected_points[:, 0]  # x coordinates
                self.IC[:, 1] = selected_points[:, 1]  # y coordinates
                self.IC[:, 2] = self.parameters['t domain'][0]  # Initial time


                # Create a tensor to store the initial temperature values
                IC_values = torch.zeros(selected_points.shape[0], 1)


                # Loop through the selected points to assign initial temperature values
                for i, idx in enumerate(random_indices):
                    for key in pde_keys:
                        # Check if the selected point matches a point in the boundary conditions
                        if key in self.bc.new_sets and torch.all(all_pde_points_tensor[idx, :2] == self.bc.new_sets[key][:, :2], dim=1).any():
                            # Assign the initial temperature value from the boundary conditions
                            IC_values[i] = self.ic_adim[key]
                            break
                
                # Set the initial temperature values in the IC tensor
                self.IC[:, 3] = IC_values.squeeze()

            
            # Set the initial temperature value for each PDE key
            for key in pde_keys:
                if key in self.bc.new_sets:
                    global Temp0
                    Temp0 = self.ic_adim[key]

            #Concatenate IC
            self.xt = self.IC

            # Create the domain with time values different from 0
            
            # initialize the datapoints
            all_points = []

            for key, points in self.bc.new_sets.items():
                
                num_points = points.shape[0]
                
                # Generate random time values for each point within the defined time domain
                t_values = torch.empty((num_points, 1)).uniform_(self.parameters['t domain'][0],self.parameters['t domain'][1])
                
                # Assign the initial temperature value (Temp0) to each point
                temp_values = torch.full((num_points, 1), Temp0)
                
                # Concatenate the x, y coordinates, time, and temperature into a new data tensor
                new_points = torch.cat((points[:, :2], t_values, temp_values), dim=1)
                
                # Append the new data to the list of all points
                all_points.append(new_points)
            

            # Concatenate all the data into a tensor
            points_tensor = torch.cat(all_points, dim=0) if all_points else torch.empty((0, 4))
            
        
            # compile the dataset
            self.xt = torch.cat((self.xt, points_tensor), dim=0)
            self.xt = self.xt.to(torch.float32)

            # torch.set_printoptions(threshold=5000)
            # print(self.xt)  

            # Display message 
            self.output.insert(tk.END, "Domínio criado com sucesso!\n")


            print("Dataset created with {:,d} elements\n".format(self.xt.shape[0]))
            
            # Create the dataloader
            print("Creating the DATALOADER...")
            self.loader = torch.utils.data.DataLoader(
                list(self.xt),
                batch_size=self.parameters['batch size'],
                shuffle=True
                )
            
            # Save the generated domain data to a .txt file
            xt_numpy = self.xt.cpu().numpy()
            np.savetxt(f"dominio_gerado_{os.path.splitext(self.parameters['nodes file'])[0]}.txt", xt_numpy, delimiter="\t", header="x\ty\tt\ttemp", comments="")
            
            print("Dataloader created\n")
    

    def trainPINN(self):
        
        # Create a progress window for visual feedback during training
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Progresso")
        progress_window.geometry("400x100")

        label = tk.Label(progress_window, text="Treino a decorrer...")
        label.pack(pady=5)


        # Progress Bar
        progress_bar = ttk.Progressbar(
        progress_window,
        orient="horizontal",
        length=300,
        mode="determinate"
        )
        progress_bar.pack(pady=10)


         # Status label
        status_label = tk.Label(progress_window, text="Epoch: 0")
        status_label.pack()


        # Update the progress window immediately
        progress_window.update()  

        # set the model into training mode
        self.model.train()

        # create the train loss database
        self.train_loss = []

        # define the databases for the loss function
        self.train_loss_IC = []
        self.train_loss_PDE = []

         # Initialize Database for loss BC
        for key in self.bc.var_dict.keys():
            
            if self.bc.var_dict[key].get() != 3:  # Ignore PDE
    
                # Dynamically create loss lists for BC keys
                setattr(self, f"train_loss_BC_{key}", [])
        


        t0 = time.time()
        for i in range(self.parameters['epochs']):
            
            # Update progress Bar
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

            # Initialize the loss string
            loss_str = "Loss: {:.2e} - L0: {:.2e}".format(last_loss, self.train_loss_IC[-1])

            # Dynamically add the loss for each BC
            for key in self.bc.var_dict.keys():
                if self.bc.var_dict[key].get() != 3:  # Ignore PDE keys
                    loss_list_key = f"train_loss_BC_{key}"
                    
                    # Check if the list exists and is not empty
                    if hasattr(self, loss_list_key) and getattr(self, loss_list_key): 
                        loss_str += " - L{}: {:.2e}".format(key, getattr(self, loss_list_key)[-1])

            # Add the loss from the PDE calculations
            loss_str += " - PDE: {:.2e}".format(self.train_loss_PDE[-1])
            print(loss_str)

            #
            print("Epoch time: {:.2f} seconds\n".format(time.time() - epoch_time))
            #

        # Close the progress window after training
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
            if self.bc.var_dict[key].get() != 3: 
                loss_list_key = f"train_loss_BC_{key}"
        
                # Dynamically get the loss list for BC
                loss_list = getattr(self, loss_list_key, None)
                if loss_list is not None and len(loss_list) > 0:
                    f.create_dataset(f"loss BC {key}", data=loss_list, compression="gzip")


        f.create_dataset("loss PDE", data=self.train_loss_PDE, compression="gzip")
        f.flush()
        f.close()
        print("Recording data...DONE!!!")
        

        #Save the trained model to a file
        filename = os.path.splitext(self.filename)[0] + "_model.pth"
        torch.save(self.model, filename)
        print("Model saved...")

        # plt.plot(self.train_loss,label="Loss")
        # plt.yscale('log')
        # plt.grid(True, which='both')
        # plt.savefig("loss.png")


        self.output.insert(tk.END, f"Treino completo em {tf-t0:.2f} segundos\n")
        



    def lossFunction(self, dataset):
        # variables index
        X = 0       # x of the dataset
        Y = 1       # y of the dataset
        t = 2       # t of the dataset
        T0_ = 3     # T0's
        #
        #***************************************************************
        # filter the initial conditions
        xt_0_idx = torch.where(dataset[:,t] == self.parameters['t domain'][0])[0]
        if xt_0_idx.shape[0] != 0:
            # build the initial conditions dataset
            xt_0 = torch.index_select(dataset, dim=0, index=xt_0_idx).to(self.device)
            # build the initial conditions
            T0 = torch.index_select(dataset[:,[T0_]], dim=0, index=xt_0_idx).to(self.device)
            #
            # evaluate the initial condition loss
            L0 = self.criteria(self.model(xt_0), T0)
            # save the indices
            pde_indices = xt_0_idx.clone()
        else:
            l0 = torch.zeros(size=(1,1)).to(self.device)
            L0 = self.criteria(l0,l0)
        # save the value
        self.train_loss_IC.append(L0.item())
    
        ############################################################

        # Loop over BC

        for key in self.bc.var_dict.keys():
            if self.bc.var_dict[key].get() != 3: 

                if key in self.bc.new_sets:
                    points = self.bc.new_sets[key]  # Get the points defined for this BC

                # Create dynamic variables for each BC key
                    xt_key = f"xt_{key}"
                    T_key = f"T_{key}"
                    L_key = f"L_{key}"  
                    train_loss_BC_key = f"train_loss_BC_{key}"  

                    # Find the indices in the dataset that match the BC
                    xt_T_idx = torch.where(
                        (dataset[:, X].unsqueeze(1) == points[:, 0].unsqueeze(0)) & 
                        (dataset[:, Y].unsqueeze(1) == points[:, 1].unsqueeze(0))
                    )[0]


                    if xt_T_idx.shape[0] != 0:
                    
                        xt_T_tensor = torch.index_select(dataset, dim=0, index=xt_T_idx)
                        
                        # Store as dynamic attributes
                        setattr(self, f"xt_{key}", xt_T_tensor)
                        setattr(self, f"xt_{key}", getattr(self, f"xt_{key}").to(self.device))
                        setattr(self, T_key, torch.ones(size=(xt_T_tensor.shape[0], 1), dtype= torch.float32) * self.bc_adim[key])
                        setattr(self, T_key, getattr(self, T_key).to(self.device))
                        setattr(self, L_key, self.criteria(self.model(getattr(self, xt_key)), getattr(self, T_key)))
                        
                    
                        # Save the indices
                        pde_indices = torch.cat((pde_indices, xt_T_idx.clone()), dim=0)
                    
                    else:
                        ll = torch.zeros(size=(1, 1)).to(self.device)
                        setattr(self, L_key, self.criteria(ll, ll))

                    # Save the boundary condition loss in the dynamic loss list
                    if not hasattr(self, train_loss_BC_key):
                        setattr(self, train_loss_BC_key, [])

                    getattr(self, train_loss_BC_key).append(getattr(self, L_key).item())
        
        #***************************************************************
        # filter the domain conditions (all points)
        # create the filtered index for the PDE
        full_dataset_index = torch.arange(end=dataset.shape[0])
        #
        mask = ~torch.isin(full_dataset_index, pde_indices)
        #
        filtered_indices = full_dataset_index[mask]
        #
        xt_domain = torch.index_select(dataset, dim=0, index=filtered_indices).to(self.device)
        xt_domain.requires_grad_(True)
        #
        x_pde = xt_domain[:,[X]]
        y_pde = xt_domain[:,[Y]]
        t_pde = xt_domain[:,[t]]
        T0_pde = xt_domain[:,[T0_]]
        T_est = self.model(torch.concat((x_pde,y_pde,t_pde,T0_pde), axis=1))
        #
        # compute the derivatives
        dTdt = torch.autograd.grad(
            outputs=T_est,
            inputs=t_pde,
            grad_outputs=torch.ones_like(t_pde),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        #
        dTdx = torch.autograd.grad(
            outputs=T_est,
            inputs=x_pde,
            grad_outputs=torch.ones_like(x_pde),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        #
        dTdy = torch.autograd.grad(
            outputs=T_est,
            inputs=y_pde,
            grad_outputs=torch.ones_like(y_pde),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        #
        dTdxx = torch.autograd.grad(
            outputs=dTdx,
            inputs=x_pde,
            grad_outputs=torch.ones_like(x_pde),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        #
        dTdyy = torch.autograd.grad(
            outputs=dTdy,
            inputs=y_pde,
            grad_outputs=torch.ones_like(y_pde),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        #
        L_PDE = self.criteria(dTdt, (self.parameters['alphax'] * dTdxx) + (self.parameters['alphay'] * dTdyy))
        #
        self.train_loss_PDE.append(L_PDE.item())
        # compute the total loss
        total_loss = L0 + sum(getattr(self, f"L_{key}") for key in self.bc.var_dict.keys() if hasattr(self, f"L_{key}")) + L_PDE

        # return the total loss
        return total_loss