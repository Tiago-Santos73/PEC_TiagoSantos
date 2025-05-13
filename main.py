import tkinter as tk
import nodes_physics
import torch
torch.set_default_dtype(torch.float32)
import parametros
import BC
import IC
import NNet
import read_nodes
import plot
import loadmodel
import loadmodel_singlepoint

# Initialize main Tkinter window
root = tk.Tk()
root.title("GUI")

# Physical Groups Function
def physic_grp_outpt(): 
    N_pe = nodes_physics.nodes_pe(filevar,output)
    N_pe.phygrp()


# Read the mesh and initialize BC and IC 
def ler():
    global sets, BCw, ICw
    R = read_nodes.Read(filevar.get())
    sets = R.read()
    BCw = BC.BCGUI(root,sets,output)
    ICw = IC.ICGUI(root, sets, output, BCw)
    output.insert(tk.END, "Malha lida com sucesso!\n")


# Create NN function
def create_NN():
    param_ui.guardar_parametros()
    global NN
    #print(d)
    NN = NNet.Plate(root, param_ui.d, BCw, ICw, output)
    output.insert(tk.END, "NN criada com sucesso!\n")


#Train NN Function
def train_network():
    if 'NN' in globals():
        NN.trainPINN()
        plt = plot.PlotLoss(filevar.get(),output)
        plt.plot_loss()

    else:
        output.insert(tk.END, "Erro: Crie a rede neuronal primeiro!\n", "red")


#Evaluate Model Function
def eval():
    global ev
    ev = loadmodel.Inference(root,param_ui.d,output,ICw,BCw)
    ev.janela()


#Evaluate Model Point Function
def evalponto():
    global evp
    evp = loadmodel_singlepoint.SinglePointInference(root,param_ui.d,output,ICw,BCw)
    evp.janela()


#Frame Entry + Button
file_frame = tk.Frame(root)
file_frame.pack()


#Entry
filevar=tk.StringVar(value="Insert file name.inp")
entry=tk.Entry(file_frame, textvariable=filevar)
entry.pack(side=tk.LEFT,pady=5,padx=5)


#Button
g_btn = tk.Button(file_frame, text= "Guardar", command=lambda: param_ui.guardar_parametros())
g_btn.pack(side=tk.RIGHT)


# Frame Buttons + Output
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)


# Textbox Output
output = tk.Text(main_frame,height=20, width=40)
output.tag_configure("red", foreground="red")
output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5, padx = 5)


# Frame Buttons
button_frame = tk.Frame(main_frame)
button_frame.pack(side=tk.RIGHT, anchor='n', padx=5, pady=5)

    
#Parameter
param_ui = parametros.ParametrosNN(root,output,filevar)
btn_p = tk.Button(button_frame, text="Parametrizar", command=param_ui.parametrizar)
btn_p.pack(pady=5)


#Physical Groups
btn_run = tk.Button(button_frame, text="Informação do \n Ficheiro", command=physic_grp_outpt)
btn_run.pack(pady=5)


#Read Mesh
btn_RNN = tk.Button(button_frame, text="Ler Malha", command=ler)
btn_RNN.pack(pady=5)


#BC
btn_bc = tk.Button(button_frame, text="BC", command=lambda: BCw.create_window() if 'BCw' in globals() else output.insert(tk.END, "Erro: Leia a malha primeiro!\n","red"))
btn_bc.pack(pady=5)


#IC
btn_Ic = tk.Button(button_frame, text="IC", command=lambda: ICw.create_window() if 'BCw' in globals() else output.insert(tk.END, "Erro: Defina primeiro as condições de fronteira!\n","red"))
btn_Ic.pack(pady=5)


#Create NN
btn_NN = tk.Button(button_frame, text="Criar NN", command=create_NN)
btn_NN.pack(pady=5)


#Domain
btn_domain = tk.Button(button_frame, text="Domínio", command=lambda: NN.define_domain() if 'BCw' in globals() else output.insert(tk.END, "Erro: Defina primeiro as condições de fronteira!\n","red"))
btn_domain.pack(pady=5)


#Train NN
btn_train = tk.Button(button_frame, text="Treinar Rede", command = train_network)
btn_train.pack(pady=5)


#Evaluate Model
btn_eval = tk.Button(button_frame, text="Avaliar Rede", command = eval)
btn_eval.pack(pady=5)


#Evaluate Model Point
btn_pltevalp = tk.Button(button_frame, text="Avaliar Ponto", command = evalponto)
btn_pltevalp.pack(pady=5)


root.mainloop()