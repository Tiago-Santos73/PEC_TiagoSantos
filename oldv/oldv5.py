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

#inicializar janela e parametros
root = tk.Tk()
root.title("GUI")



def create_NN():
    param_ui.guardar_parametros()
    global NNet
    #print(d)
    NNet = _2D_pinn_4_inputs.Plate(params=param_ui.d)
    output.insert(tk.END, "NN criada com sucesso!\n")

def ler():
    global sets, BCw
    sets = NNet.read_mesh()
    BCw = BC.BCGUI(root,sets,output)
    output.insert(tk.END, "Rede lida com sucesso!\n")
    #print(sets)


#frame entry+label
file_frame = tk.Frame(root)
file_frame.pack()

#entrada e variavel
filevar=tk.StringVar(value="Insert file name.inp")
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
btn_run = tk.Button(button_frame, text="File Info", command=N_pe.runf)
btn_run.pack(pady=5)

#botao crate NN
btn_NN = tk.Button(button_frame, text="Create NN", command=create_NN)
btn_NN.pack(pady=5)

#botao reade NN
btn_RNN = tk.Button(button_frame, text="Read Mesh", command=ler)
btn_RNN.pack(pady=5)


#botao bc
btn_bc = tk.Button(button_frame, text="BC", command=lambda: BCw.create_window() if 'BCw' in globals() else output.insert(tk.END, "Erro: Leia a malha primeiro!\n","red"))
btn_bc.pack(pady=5)


#botao ic
btn_Ic = tk.Button(button_frame, text="IC", command=lambda: IC.ICGUI(root, sets, output, BCw) if 'BCw' in globals() else output.insert(tk.END, "Erro: Defina primeiro as condições de fronteira!\n","red"))
btn_Ic.pack(pady=5)



root.mainloop()


