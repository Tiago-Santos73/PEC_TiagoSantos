

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
                self.bc_adim[key] = ((float(self.bc.entry_values[key])-tmin)/(tmax-tmin)) * (0.5-0.25) + 0.25 #####

        for key in self.ic.ic_values:
            self.ic_adim[key] = ((float(self.ic.ic_values[key])-tmin)/(tmax-tmin)) * (0.5-0.25) + 0.25 #####

            

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
            #
            # get the initial conditions datapoints
            # initialize the IC datapoints


            """"""""""""""""""""""""""""""""""""""
            # Filtrar as chaves em var_dict que têm valor 3 (PDE)
            pde_keys = [key for key, var in self.bc.var_dict.items() if var.get() == 3]

            # Contar o número total de pontos internos (com condição PDE)
            num_pde_points = sum(self.bc.new_sets[key].shape[0] for key in pde_keys if key in self.bc.new_sets)

            # Criar o tensor IC com o número correto de pontos
            self.IC = torch.zeros(size=(num_pde_points, 4))  # Ajuste o número de colunas conforme necessário
            """"""""""""""""""""""""""
            # # generate the datapoints with an uniform distributio

            # Assumindo que self.bc.new_sets tem as chaves que representam pontos, por exemplo, x, y, etc.
            IC_x = torch.empty(0, 1)  # Inicializar tensor vazio
            IC_y = torch.empty(0, 1)  # Inicializar tensor vazio

            # Gerar pontos apenas para chaves com condição "PDE"
            for key, var in self.bc.var_dict.items():
                if var.get() == 3:  # Se a chave está marcada como "PDE"
            # Obter os pontos correspondentes à condição PDE
                    points = self.bc.new_sets[key]
            
            # Concatenar os pontos no IC_x e IC_y
                    IC_x = torch.cat((IC_x, points[:, 0].unsqueeze(1)), dim=0)  # Pega os valores de x
                    IC_y = torch.cat((IC_y, points[:, 1].unsqueeze(1)), dim=0)  # Pega os valores de y

            # Agora IC_x e IC_y contêm os pontos válidos para as condições iniciais
            # Podemos também gerar valores para IC (temperaturas, etc.) para esses pontos
            
            IC_values = torch.zeros(IC_x.shape[0], 1)  # Para armazenar as condições iniciais
            

            
            idx = 0  # Índice para percorrer os pontos no IC_x e IC_y
            for key in pde_keys:
                if key in self.bc.new_sets:
                    points = self.bc.new_sets[key]
                    
                    # Pega o valor da temperatura inicial definida para o ponto
                    # initial_temperature = self.ic.ic_values.get(key, self.parameters['T0'])  # Padrão para T0 se não definido
                    global Temp0
                    Temp0 = self.ic_adim[key]
                    # Temp0 = float(initial_temperature)
                    # Atribui a temperatura aos pontos correspondentes
                    for _ in range(points.shape[0]):
                        IC_values[idx] = Temp0  # Atribui a temperatura ao ponto correspondente
                        idx += 1  # Avança para o próximo ponto


            # Imprimir para verificar
            # print(f"IC_x: {IC_x.shape}")
            # print(f"IC_y: {IC_y.shape}")
            # print(f"IC_values: {IC_values.shape}")


            """"""""""""""""""""""""""""""""
            # compile the IC position
            self.IC[:,0] = IC_x[:,0]
            self.IC[:,1] = IC_y[:,0]
            # compile the IC time
            self.IC[:,2] = self.parameters['t domain'][0]
            #
            # compile the initial temperature
            self.IC[:, 3] = IC_values.squeeze()
            #
            """"""""""""""""""""""""""""""""""""""""""""""""""

            # se nao convergir o problema pode estar aqui
            for key, var in self.bc.var_dict.items():
                if var.get() != 3:  # Ignorar as chaves do tipo PDE
                # Verificar se a chave existe em self.bc.new_sets (pontos de fronteira)
                    if key in self.bc.new_sets:
                # Obter os pontos da chave em new_sets
                        points = self.bc.new_sets[key]
                
                # Criar o tensor vazio para armazenar os dados de BC [x, y, t, bc_value]
                        bc_tensor = torch.zeros((points.shape[0], 4))  # 4 colunas (x, y, t, bc_value)
                
                # Preencher a primeira coluna com as coordenadas x
                        bc_tensor[:, 0] = points[:, 0]
                
                # Preencher a segunda coluna com as coordenadas y
                        bc_tensor[:, 1] = points[:, 1]
                
                # Preencher a terceira coluna com os valores de t 
                        bc_tensor[:, 2] = self.parameters['t domain'][0] # Preencher o domínio de tempo

                # Preencher a quarta coluna com o valor da condição de fronteira
                        bc_tensor[:, 3] = Temp0  # Preencher o valor da inicial
                
                # Criar um atributo no objeto para armazenar o tensor específico de cada chave
                        setattr(self, f"BC_tensor_{key}", bc_tensor)
            

            """"""""""""""""""""""""""""""""""""""""""""""""""""""""
            # # compile the dataset
          

            # Inicializando uma lista para armazenar todos os tensores
            bc_tensors = [self.IC]  # Comece com self.IC como primeiro tensor

            # Percorrer todas as chaves em self e adicionar os tensores BC_tensor_{key} à lista
            for key in self.bc.var_dict.keys():
                if hasattr(self, f"BC_tensor_{key}"):  # Verificar se o tensor existe
                    bc_tensor = getattr(self, f"BC_tensor_{key}")  # Obter o tensor correspondente à chave
                    bc_tensors.append(bc_tensor)  # Adicionar à lista de tensores

            # Concatenar todos os tensores na lista
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

            torch.set_printoptions(threshold=5000)
            print(self.xt)  
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
            # T0 = torch.index_select(dataset[:,[T0_]], dim=0, index=xt_0_idx).to(self.device)
            T0 = (torch.ones(size=(xt_0_idx.shape[0], 1), dtype= torch.float32) * Temp0).to(self.device)
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
    
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        for key in self.bc.var_dict.keys():
            if self.bc.var_dict[key].get() != 3:  # Ignorar chaves do tipo PDE
            # Obter os pontos de fronteira definidos na BCGUI
                if key in self.bc.new_sets:
                    points = self.bc.new_sets[key]  # Obtém os pontos definidos para esta condição de fronteira

                # Criar variáveis dinâmicas para cada chave de BC
                    xt_key = f"xt_{key}"
                    T_key = f"T_{key}"
                    L_key = f"L_{key}"  # Nome dinâmico para a perda
                    train_loss_BC_key = f"train_loss_BC_{key}"  # Lista dinâmica para perdas

                    xt_T_idx = torch.where(
                        (dataset[:, X].unsqueeze(1) == points[:, 0].unsqueeze(0)) & 
                        (dataset[:, Y].unsqueeze(1) == points[:, 1].unsqueeze(0))
                    )[0]

                    # xt_L_idx = torch.where(torch.all(dataset[:, [X, Y]].unsqueeze(1) == points.unsqueeze(0), dim=2))[0]


                    if xt_T_idx.shape[0] != 0:
                        # Construir o dataset das condições de fronteira
                        # 
                        xt_T_tensor = torch.index_select(dataset, dim=0, index=xt_T_idx)
                        setattr(self, f"xt_{key}", xt_T_tensor)
                        setattr(self, f"xt_{key}", getattr(self, f"xt_{key}").to(self.device))
                        setattr(self, T_key, torch.ones(size=(xt_T_tensor.shape[0], 1), dtype= torch.float32) * self.bc_adim[key])
                        setattr(self, T_key, getattr(self, T_key).to(self.device))
                        

                    
                        setattr(self, L_key, self.criteria(self.model(getattr(self, xt_key)), getattr(self, T_key)))
                        
                    
                    
                    
                    # Salvar os índices
                        pde_indices = torch.cat((pde_indices, xt_T_idx.clone()), dim=0)
                    else:
                        ll = torch.zeros(size=(1, 1)).to(self.device)
                        setattr(self, L_key, self.criteria(ll, ll))

                # Salvar o valor da perda dinâmica
                loss_value = getattr(self, L_key).item()

                # Usar getattr para acessar a lista de perdas dinâmica e armazenar o valor
                if not hasattr(self, train_loss_BC_key):
                    setattr(self, train_loss_BC_key, [])  # Se não existir, cria a lista
                getattr(self, train_loss_BC_key).append(loss_value)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
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