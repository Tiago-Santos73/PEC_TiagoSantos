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


class Plate(torch.nn.Module):
    def __init__(self, params, bc, ic,output, verbose = False):
        super(Plate,self).__init__()
        self.parameters = params
        self.bc = bc  # Instância da GUI das condições de fronteira
        self.ic = ic  # Instância da GUI das condições iniciais
        self.xt_pde = None
        self.xt_bc = None
        self.loader = None
        self.output = output

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
            

            #v2
            idx = 0  # Índice para percorrer os pontos no IC_x e IC_y
            for key in pde_keys:
                if key in self.bc.new_sets:
                    points = self.bc.new_sets[key]
                    
                    # Pega o valor da temperatura inicial definida para o ponto
                    initial_temperature = self.ic.ic_values.get(key, self.parameters['T0'])  # Padrão para T0 se não definido
                    Temp0 = float(initial_temperature)
                    # Atribui a temperatura aos pontos correspondentes
                    for _ in range(points.shape[0]):
                        IC_values[idx] = float(initial_temperature)  # Atribui a temperatura ao ponto correspondente
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
            #self.IC[:,3] = self.parameters['T0']
            self.IC[:, 3] = IC_values.squeeze()
            #
            """"""""""""""""""""""""""""""""""""""""""""""""""

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
                        bc_tensor[:, 3] = Temp0  # Preencher o valor da condição de fronteira
                
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