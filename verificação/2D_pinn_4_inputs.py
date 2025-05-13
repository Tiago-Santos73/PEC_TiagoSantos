# along this training algorithm, the boundaries are not
# accounted for the domain due to they are fixed solutions
# of the problem and not variables

# Constant temperatures 

# https://github.com/nkusla/heat-transfer-pinn

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
import numpy as np


class Plate(torch.nn.Module):
    def __init__(self, params, verbose = False):
        super(Plate,self).__init__()
        #
        self.filename = "pinnsgui(1)" #params['folder'] + params['architecture']
        #
        # get the parsed parameters
        self.parameters = params
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
        # Define the domain
        self.define_domain()
        # exit()
        #
        # Train the function
        self.trainPINN()



    def define_domain(self):
        print("Defining the domain...")
        #
        # get the initial conditions datapoints
        # initialize the IC datapoints
        self.IC = torch.zeros(size=(self.parameters['IC samples'],4))
        # generate the datapoints with an uniform distribution
        IC_x = torch.empty(self.parameters['IC samples'], 1).uniform_(
            self.parameters['x domain'][0],
            self.parameters['x domain'][1]
        )
        IC_y = torch.empty(self.parameters['IC samples'], 1).uniform_(
            self.parameters['y domain'][0],
            self.parameters['y domain'][1]
        )
        #
        # compile the IC position
        self.IC[:,0] = IC_x[:,0]
        self.IC[:,1] = IC_y[:,0]
        # compile the IC time
        self.IC[:,2] = self.parameters['t domain'][0]
        #
        # compile the initial temperature
        self.IC[:,3] = self.parameters['T0']
        #
        #***************************************************************
        #
        # boundary condition x=0 (constant temperature)
        # initialize the datapoints
        self.BC_L = torch.zeros(size=(self.parameters['BC samples'],4))
        # generate the datapoints with an uniform distribution
        BC_L_y = torch.empty(self.parameters['BC samples'], 1).uniform_(
            self.parameters['y domain'][0],
            self.parameters['y domain'][1]
        )
        BC_L_t = torch.empty(self.parameters['BC samples'],1).uniform_(
            self.parameters['t domain'][0],
            self.parameters['t domain'][1]
        )
        # compile the position
        self.BC_L[:,0] = self.parameters['x domain'][0]
        self.BC_L[:,1] = BC_L_y[:,0]
        # compile the boundary condition time
        self.BC_L[:,2] = BC_L_t[:,0]
        # compile the initial temperature
        self.BC_L[:,3] = self.parameters['T0']
        #
        #***************************************************************
        #
        # boundary condition x=1 (constant temperature)
        # initialize the datapoints
        self.BC_R = torch.zeros(size=(self.parameters['BC samples'],4))
        # generate the datapoints with an uniform distribution
        BC_R_y = torch.empty(self.parameters['BC samples'], 1).uniform_(
            self.parameters['y domain'][0],
            self.parameters['y domain'][1]
        )
        BC_R_t = torch.empty(self.parameters['BC samples'],1).uniform_(
            self.parameters['t domain'][0],
            self.parameters['t domain'][1]
        )
        # compile the position
        self.BC_R[:,0] = self.parameters['x domain'][1]
        self.BC_R[:,1] = BC_R_y[:,0]
        # compile the boundary condition time
        self.BC_R[:,2] = BC_R_t[:,0]
        # compile the initial temperature
        self.BC_R[:,3] = self.parameters['T0']
        #
        #***************************************************************
        #
        # boundary condition y=0 (constant temperature)
        # initialize the datapoints
        self.BC_D = torch.zeros(size=(self.parameters['BC samples'],4))
        # generate the datapoints with an uniform distribution
        BC_D_x = torch.empty(self.parameters['BC samples'], 1).uniform_(
            self.parameters['x domain'][0],
            self.parameters['x domain'][1]
        )
        BC_D_t = torch.empty(self.parameters['BC samples'],1).uniform_(
            self.parameters['t domain'][0],
            self.parameters['t domain'][1]
        )
        # compile the position
        self.BC_D[:,0] = BC_D_x[:,0]
        self.BC_D[:,1] = self.parameters['y domain'][0]
        # compile the boundary condition time
        self.BC_D[:,2] = BC_D_t[:,0]
        # compile the initial temperature
        self.BC_D[:,3] = self.parameters['T0']
        #
        #***************************************************************
        #
        # boundary condition y=0 (constant temperature)
        # initialize the datapoints
        self.BC_U = torch.zeros(size=(self.parameters['BC samples'],4))
        # generate the datapoints with an uniform distribution
        BC_U_x = torch.empty(self.parameters['BC samples'], 1).uniform_(
            self.parameters['x domain'][0],
            self.parameters['x domain'][1]
        )
        BC_U_t = torch.empty(self.parameters['BC samples'],1).uniform_(
            self.parameters['t domain'][0],
            self.parameters['t domain'][1]
        )
        # compile the position
        self.BC_U[:,0] = BC_U_x[:,0]
        self.BC_U[:,1] = self.parameters['y domain'][1]
        # compile the boundary condition time
        self.BC_U[:,2] = BC_U_t[:,0]
        # compile the initial temperature
        self.BC_U[:,3] = self.parameters['T0']
        #
        #***************************************************************
        #
        # get the domain points
        # initialize the datapoints
        self.domain = torch.zeros(size=(self.parameters['domain samples'],4))
        # generate the datapoints with an uniform distribution
        domain_x = torch.empty(self.parameters['domain samples'],1).uniform_(
            self.parameters['x domain'][0],self.parameters['x domain'][1]
        )
        # generate the datapoints with an uniform distribution
        domain_y = torch.empty(self.parameters['domain samples'],1).uniform_(
            self.parameters['y domain'][0],self.parameters['y domain'][1]
        )
        # generate the datapoints with an uniform distribution
        domain_t = torch.empty(self.parameters['domain samples'],1).uniform_(
            self.parameters['t domain'][0],self.parameters['t domain'][1]
        )
        # compile the domain
        self.domain[:,0] = domain_x[:,0]
        self.domain[:,1] = domain_y[:,0]
        self.domain[:,2] = domain_t[:,0]
        self.domain[:,3] = self.parameters['T0']
        #
        # compile the dataset
        self.xt = torch.concat((self.IC,
                                self.BC_L,
                                self.BC_R,
                                self.BC_D,
                                self.BC_U,
                                self.domain),
                                axis=0)
        print("Dataset created with {:,d} elements\n".format(self.xt.shape[0]))
        # Create the dataloader
        print("Creating the DATALOADER...")
        self.loader = torch.utils.data.DataLoader(
            list(self.xt),
            batch_size=self.parameters['batch size'],
            shuffle=True
            )
        #
        print("Dataloader created\n")
        #print(self.xt)
        # xt_numpy = self.xt.cpu().numpy()
        # np.savetxt("dominio_gerado_nogui.txt", xt_numpy, delimiter="\t", header="x\ty\tt\ttemp", comments="")

        # # Verificar se existem pontos repetidos
        # print("Checking for duplicate (x, y) points in the dataset...")
        # xy = self.xt[:, :2]  # pega apenas as colunas x e y
        # xy_unique = torch.unique(xy, dim=0)
        # has_xy_duplicates = xy_unique.shape[0] != xy.shape[0]
        # print("Has duplicate (x, y):", has_xy_duplicates)


        


    def trainPINN(self):
        # set the model into training mode
        self.model.train()
        # create the train loss database
        self.train_loss = []
        # define the databases for the loss function
        self.train_loss_IC = []
        self.train_loss_BCL = []
        self.train_loss_BCR = []
        self.train_loss_BCD = []
        self.train_loss_BCU = []
        self.train_loss_PDE = []
        #
        t0 = time.time()
        for i in range(self.parameters['epochs']):
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
            #
            print("Loss: {:.2e} - L0: {:.2e} - LL: {:.2e} - LR: {:.2e} - LD: {:.2e} - LU: {:.2e} - PDE: {:.2e}".format(last_loss,
                                                                                             self.train_loss_IC[-1],
                                                                                             self.train_loss_BCL[-1],
                                                                                             self.train_loss_BCR[-1],
                                                                                             self.train_loss_BCD[-1],
                                                                                             self.train_loss_BCU[-1],
                                                                                             self.train_loss_PDE[-1]))
            #
            print("Epoch time: {:.2f} seconds\n".format(time.time() - epoch_time))
            #
            # check the condition
                     
        #
        tf = time.time()
        print("Training took: {:.2f} seconds\n".format(tf-t0))
        # save the data
        print("Recording data...")
        #
        filename = self.filename + '_losses.h5'
        f = h5py.File(filename, "w")
        f.create_dataset("loss", data=self.train_loss, compression="gzip")
        f.create_dataset("loss IC", data=self.train_loss_IC, compression="gzip")
        f.create_dataset("loss BCL", data=self.train_loss_BCL, compression="gzip")
        f.create_dataset("loss BCR", data=self.train_loss_BCR, compression="gzip")
        f.create_dataset("loss BCD", data=self.train_loss_BCD, compression="gzip")
        f.create_dataset("loss BCU", data=self.train_loss_BCU, compression="gzip")
        f.create_dataset("loss PDE", data=self.train_loss_PDE, compression="gzip")
        f.flush()
        f.close()
        print("Recording data...DONE!!!")
        #
        filename = "pinnsgui(1)" + "_model.pth"
        torch.save(self.model, filename)
        print("Model saved...")

        plt.plot(self.train_loss,label="Loss")
        plt.yscale('log')
        plt.grid(True, which='both')
        plt.savefig("loss.png")



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
        #
        #***************************************************************
        # filter the left BC x = 0
        xt_L_idx = torch.where(dataset[:,X] == self.parameters['x domain'][0])[0]
        if xt_L_idx.shape[0] != 0:
            # build the left BC dataset
            xt_L = torch.index_select(dataset, dim=0, index=xt_L_idx).to(self.device)
            # build the condition
            T_L = torch.ones(size=(xt_L.shape[0],1)) * self.parameters['TL']
            T_L = T_L.to(self.device)
            #
            # evaluate the boundary condition loss
            LL = self.criteria(self.model(xt_L), T_L)
            # save the indices
            pde_indices = torch.cat((pde_indices, xt_L_idx.clone()), dim=0)
        else:
            ll = torch.zeros(size=(1,1)).to(self.device)
            LL = self.criteria(ll,ll)
        # save the value
        self.train_loss_BCL.append(LL.item())
        #
        #***************************************************************
        # filter the right BC x = L
        xt_R_idx = torch.where(dataset[:,X] == self.parameters['x domain'][-1])[0]
        if xt_R_idx.shape[0] != 0:
            # build the left BC dataset
            xt_R = torch.index_select(dataset, dim=0, index=xt_R_idx).to(self.device)
            # build the condition
            T_R = torch.ones(size=(xt_R.shape[0],1)) * self.parameters['TR']
            T_R = T_R.to(self.device)
            #
            # evaluate the boundary condition loss
            LR = self.criteria(self.model(xt_R), T_R)
            # save the indices
            pde_indices = torch.cat((pde_indices, xt_R_idx.clone()), dim=0)
        else:
            lr = torch.zeros(size=(1,1)).to(self.device)
            LR = self.criteria(lr,lr)
        # save the value
        self.train_loss_BCR.append(LR.item())
        #
        #***************************************************************
        # filter the down BC y = 0
        xt_D_idx = torch.where(dataset[:,Y] == self.parameters['y domain'][0])[0]
        if xt_D_idx.shape[0] != 0:
            # build the left BC dataset
            xt_D = torch.index_select(dataset, dim=0, index=xt_D_idx).to(self.device)
            # build the condition
            T_D = torch.ones(size=(xt_D.shape[0],1)) * self.parameters['TD']
            T_D = T_D.to(self.device)
            #
            # evaluate the boundary condition loss
            LD = self.criteria(self.model(xt_D), T_D)
            # save the indices
            pde_indices = torch.cat((pde_indices, xt_D_idx.clone()), dim=0)
        else:
            ld = torch.zeros(size=(1,1)).to(self.device)
            LD = self.criteria(ld,ld)
        # save the value
        self.train_loss_BCD.append(LD.item())
        #
        #***************************************************************
        # filter the up BC y = 1
        xt_U_idx = torch.where(dataset[:,Y] == self.parameters['y domain'][1])[0]
        if xt_U_idx.shape[0] != 0:
            # build the left BC dataset
            xt_U = torch.index_select(dataset, dim=0, index=xt_U_idx).to(self.device)
            # build the condition
            T_U = torch.ones(size=(xt_U.shape[0],1)) * self.parameters['TU']
            T_U = T_U.to(self.device)
            #
            # evaluate the boundary condition loss
            LU = self.criteria(self.model(xt_U), T_U)
            # save the indices
            pde_indices = torch.cat((pde_indices, xt_U_idx.clone()), dim=0)
        else:
            lu = torch.zeros(size=(1,1)).to(self.device)
            LU = self.criteria(lu,lu)
        # save the value
        self.train_loss_BCU.append(LU.item())
        #
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
        L_PDE = self.criteria(dTdt, (self.parameters['alpha'] * dTdxx) + (self.parameters['alpha'] * dTdyy))
        #
        self.train_loss_PDE.append(L_PDE.item())
        # compute the total loss
        total_loss = L0 + LL + LR + LD + LU + L_PDE
        # return the total loss
        
        return total_loss
        

class PlotLoss:
    def __init__(self,name):
        self.name = name

    def plot_loss(self):
        # Load the h5 file into a pandas DataFrame
        # filename
        filename = self.name + "_losses.h5"
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
            plot_name = self.name + "_loss.png"
            plt.savefig(plot_name)

        return mean_last_100_loss
        


if __name__ == '__main__':
    #
    neurons = [50]
    layers = 7
    points = 5000
    lr = 0.001
    #
    FOLDER = "res_numerical"
    #
    d_total = {}
    index = 0    
    # create a temporarty dictionary
    D = {}
    # define the architecture
    #arch = "architecture_{:03d}".format(index)
    D['layers'] = layers
    D['neurons'] = neurons
    D['points'] = points
    D['learning rate'] = lr
    #
    # create the problem
    d = {}
    d['folder'] = FOLDER + "/"
    d['inputs'] = 4
    d['layers'] = [50]*7
    d['outputs'] = 1
    d['epochs'] = 20000
    d['lr'] = lr
    d['x domain'] = [0.0, 1.0]
    d['y domain'] = [0.0, 1.0]
    d['t domain'] = [0.0, 1.0]
    d['conditions'] = 6
    d['domain samples'] = points
    d['IC samples'] = int(points*0.08)
    d['BC samples'] = int(points*0.2/4.0)
    d['batch size'] = 500_000
    d['T0'] = 0.5
    d['TL'] = 0.5
    d['TR'] = 0.25
    d['TD'] = 0.5
    d['TU'] = 0.25
    d['alpha'] = 1.0
    #d['architecture'] = arch
    # run the problem
    NN = Plate(params=d)
    # plot the results
    #LM = PlotLoss(name=FOLDER + "/" + arch)
    #loss_mean = LM.plot_loss()