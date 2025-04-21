import tkinter as tk
import nodes_physics
import _2D_pinn_4_inputs
import torch
torch.set_default_dtype(torch.float32)
import parametros
import BC
import IC
import multiprocessing
import NNet
import read_nodes

#inicializar janela e parametros
root = tk.Tk()
root.title("GUI")


def ler():
    global sets, BCw, ICw
    R = read_nodes.Read(filevar.get())
    sets = R.read()
    BCw = BC.BCGUI(root,sets,output)
    ICw = IC.ICGUI(root, sets, output, BCw)
    output.insert(tk.END, "Malha lida com sucesso!\n")

def create_NN():
    param_ui.guardar_parametros()
    global NN
    #print(d)
    NN = NNet.Plate(param_ui.d, BCw, ICw, output)
    output.insert(tk.END, "NN criada com sucesso!\n")
    


#frame entry+label
file_frame = tk.Frame(root)
file_frame.pack()

#entrada e variavel
#filevar=tk.StringVar(value="Insert file name.inp")
filevar=tk.StringVar(value="file.inp")#################################
entry=tk.Entry(file_frame, textvariable=filevar)
entry.pack(side=tk.LEFT,pady=5,padx=5)

#label
g_btn = tk.Button(file_frame, text= "Guardar", command=lambda: param_ui.guardar_parametros())
g_btn.pack(side=tk.RIGHT)

# Frame butons+text
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# textbox output
output = tk.Text(main_frame,height=20, width=40)
output.tag_configure("red", foreground="red")
output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5, padx = 5)

# Frame buttons
button_frame = tk.Frame(main_frame)
button_frame.pack(side=tk.RIGHT, anchor='n', padx=5, pady=5)

#chamar classe nos e entidades fisicas
N_pe = nodes_physics.nodes_pe(filevar,output)

#parametrizar
param_ui = parametros.ParametrosNN(root,output,filevar)
btn_p = tk.Button(button_frame, text="Parametrizar", command=param_ui.parametrizar)
btn_p.pack(pady=5)

#botao run
btn_run = tk.Button(button_frame, text="Informação do \n Ficheiro", command=N_pe.runf)
btn_run.pack(pady=5)

#botao ler mesh
btn_RNN = tk.Button(button_frame, text="Ler Malha", command=ler)
btn_RNN.pack(pady=5)


#botao bc
btn_bc = tk.Button(button_frame, text="BC", command=lambda: BCw.create_window() if 'BCw' in globals() else output.insert(tk.END, "Erro: Leia a malha primeiro!\n","red"))
btn_bc.pack(pady=5)

#botao ic
btn_Ic = tk.Button(button_frame, text="IC", command=lambda: ICw.create_window() if 'BCw' in globals() else output.insert(tk.END, "Erro: Defina primeiro as condições de fronteira!\n","red"))
btn_Ic.pack(pady=5)

#botao crate NN
btn_NN = tk.Button(button_frame, text="Criar NN", command=create_NN)
btn_NN.pack(pady=5)

#botao Domain
btn_domain = tk.Button(button_frame, text="Domínio", command=lambda: NN.define_domain() if 'BCw' in globals() else output.insert(tk.END, "Erro: Defina primeiro as condições de fronteira!\n","red"))
btn_domain.pack(pady=5)

def train_network():
    if 'NN' in globals():
        NN.trainPINN()
    else:
        output.insert(tk.END, "Erro: Crie a rede neural primeiro!\n", "red")

#botao treinar
btn_train = tk.Button(button_frame, text="Treinar Rede", command = train_network)
btn_train.pack(pady=5)

root.mainloop()