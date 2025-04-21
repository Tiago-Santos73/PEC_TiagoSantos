import tkinter as tk
import nodes_physics
import oldv.job as job
import torch
torch.set_default_dtype(torch.float32)

#inicializar janela e parametros
root = tk.Tk()
#root.geometry("500x500")
root.title("GUI")


#variaveis de paramtrização
d_inputs = tk.IntVar()
d_inputs.set(3)

d_layers_nnumber = tk.IntVar()
d_layers_nnumber.set(10)

d_layersnum= tk.IntVar()
d_layersnum.set(4)

d_outputs = tk.IntVar()
d_outputs.set(1)

d_epochs = tk.IntVar()
d_epochs.set(4000)

d_lr = tk.StringVar()
d_lr.set("1e-04")

d_domain_samples = tk.IntVar()
d_domain_samples.set(500)

d_Ic = tk.IntVar()
d_Ic.set(250)

d_Bc = tk.IntVar()
d_Bc.set(250)

filevar=tk.StringVar()
filevar.set("Insert file name.inp")


def create_NN():
    guardar_parametros()
    global NN
    NN = job.Plate(params=d)
    output.insert(tk.END, "NN criada com sucesso!\n")

def ler():
    global sets
    sets = NN.read_mesh()
    output.insert(tk.END, "Rede lida com sucesso!\n")
    #print(sets)

def guardar_parametros(janela=None):
    global d
    d = {}
    d['inputs'] = d_inputs.get()
    d['layers'] = [d_layers_nnumber.get()]*d_layersnum.get()
    d['outputs'] = d_outputs.get()
    d['epochs'] = d_epochs.get()
    d['lr'] = float(d_lr.get())
    d['domain samples'] = d_domain_samples.get()
    d['IC samples'] = d_Ic.get()
    d['BC samples'] = d_Bc.get()
    d['nodes file'] = filevar.get()
    output.insert(tk.END, "Parametrização guardada!\n")
    
    if janela:
        janela.destroy()

def parametrizar():
    parametrizar_janela = tk.Toplevel(root)
    parametrizar_janela.title("Parametrizar")
    
    #label da janela
    label_param = tk.Label(parametrizar_janela, text="Parametrização", font=("Helvetica", 14, "bold"))
    label_param.pack(pady=10)

    #definicoes do frame das entradas
    entry_frame = tk.Frame(parametrizar_janela)
    entry_frame.pack(fill=tk.BOTH, expand=True)
    entry_frame.grid_rowconfigure(0, weight=1)  
    entry_frame.grid_rowconfigure(1, weight=1)  
    entry_frame.grid_rowconfigure(2, weight=1)  
    entry_frame.grid_rowconfigure(3, weight=1) 
    entry_frame.grid_rowconfigure(4, weight=1) 
    entry_frame.grid_rowconfigure(5, weight=1) 
    entry_frame.grid_rowconfigure(6, weight=1) 
    entry_frame.grid_rowconfigure(7, weight=1) 
    entry_frame.grid_rowconfigure(8, weight=1) 
    entry_frame.grid_columnconfigure(0, weight=1)
    entry_frame.grid_columnconfigure(1, weight=1)
    
    #inputs
    d_inputs_entry = tk.Entry(entry_frame, textvariable = d_inputs)
    d_inputs_entry.grid(row=0, column=0, padx=5,pady=5, sticky="ew")

    d_inputs_label = tk.Label(entry_frame, text="Inputs")
    d_inputs_label.grid(row=0, column=1, padx=5, pady= 5, sticky="ew")

    #frame para os layers
    layers_frame= tk.Frame(entry_frame)
    layers_frame.grid(row=1, sticky="ew")
    layers_frame.grid_rowconfigure(0, weight=1)
    layers_frame.grid_columnconfigure(0, weight=1)
    layers_frame.grid_columnconfigure(1, weight=1)
    layers_frame.grid_columnconfigure(2, weight=1)
    layers_frame.grid_columnconfigure(3, weight=1)


    #layers
    d_layers_nnumber_entry=tk.Entry(layers_frame,textvariable=d_layers_nnumber)
    d_layers_nnumber_entry.grid(row=0,column=0,pady=5, sticky="ew")

    d_layers_label1=tk.Label(layers_frame,text="*")
    d_layers_label1.grid(row=0,column=1,pady=5, sticky="ew")
    
    d_layersnum_entry=tk.Entry(layers_frame,textvariable=d_layersnum)
    d_layersnum_entry.grid(row=0,column=2,pady=5, sticky="ew")

    d_layers_label2=tk.Label(layers_frame,text="Layers")
    d_layers_label2.grid(row=0,column=3,pady=5, sticky="ew")

    #outputs
    d_outputs_entry = tk.Entry(entry_frame, textvariable = d_outputs)
    d_outputs_entry.grid(row=2, column=0, padx=5,pady=5, sticky="ew")

    d_outputs_label = tk.Label(entry_frame, text="Outputs")
    d_outputs_label.grid(row=2, column=1, padx=5, pady= 5, sticky="ew")

    #epochs
    d_epochs_entry = tk.Entry(entry_frame, textvariable = d_epochs)
    d_epochs_entry.grid(row=3, column=0, padx=5,pady=5, sticky="ew")

    d_epochs_label = tk.Label(entry_frame, text="Epochs")
    d_epochs_label.grid(row=3, column=1, padx=5, pady= 5, sticky="ew")

    #lr
    d_lr_entry = tk.Entry(entry_frame, textvariable = d_lr)
    d_lr_entry.grid(row=4, column=0, padx=5,pady=5, sticky="ew")

    d_lr_label = tk.Label(entry_frame, text="lr")
    d_lr_label.grid(row=4, column=1, padx=5, pady= 5, sticky="ew")

    #domain_samples
    d_domain_samples_entry = tk.Entry(entry_frame, textvariable = d_domain_samples)
    d_domain_samples_entry.grid(row=5, column=0, padx=5,pady=5, sticky="ew")

    d_domain_samples_label = tk.Label(entry_frame, text="Domain Samples")
    d_domain_samples_label.grid(row=5, column=1, padx=5, pady= 5, sticky="ew")

    #Ic
    d_Ic_entry = tk.Entry(entry_frame, textvariable = d_Ic)
    d_Ic_entry.grid(row=6, column=0, padx=5,pady=5, sticky="ew")

    d_Ic_label = tk.Label(entry_frame, text="IC Samples")
    d_Ic_label.grid(row=6, column=1, padx=5, pady= 5, sticky="ew")

    #Bc
    d_Bc_entry = tk.Entry(entry_frame, textvariable = d_Bc)
    d_Bc_entry.grid(row=7, column=0, padx=5,pady=5, sticky="ew")

    d_Bc_label = tk.Label(entry_frame, text="BC Samples")
    d_Bc_label.grid(row=7, column=1, padx=5, pady= 5, sticky="ew")

    #Nodes File
    d_nf_entry = tk.Entry(entry_frame, textvariable = filevar)
    d_nf_entry.grid(row=7, column=0, padx=5,pady=5, sticky="ew")

    d_nf_label = tk.Label(entry_frame, text="Nodes file")
    d_nf_label.grid(row=7, column=1, padx=5, pady= 5, sticky="ew")

    btn_fechar = tk.Button(parametrizar_janela, text="Guardar", command=lambda: guardar_parametros(parametrizar_janela))
    btn_fechar.pack(pady=5)

def BC():
    BC_janela = tk.Toplevel(root)
    BC_janela.title("BC")

    # Label da janela
    label_param = tk.Label(BC_janela, text="Condições de Fronteira", font=("Helvetica", 14, "bold"))
    label_param.pack(pady=10)

    var_dict = {}  # Dicionário para guardar as variáveis de cada conjunto
    entry_dict = {}  # Dicionário para guardar as entradas
    unit_dict = {}  # Dicionário para guardar os as unidades

    unidades = {  
        1: "°C",    # Temperatura 
        2: "W/m²"   # Fluxo de calor 
    }

    # Dicionário para guardar os valores de entrada
    entry_values = {}

    def update_entry(key, *args):
        selected_value = var_dict[key].get()
        if selected_value == 3:  # Se for PDE, remove a entrada e as unidades
            entry_dict[key].pack_forget()
            unit_dict[key].pack_forget()
        else:
            entry_dict[key].pack(side='left', padx=5, fill='x', expand=True)
            unit_dict[key].config(text=unidades[selected_value])
            unit_dict[key].pack(side='left', padx=5)

    # Atualizar o valor no dicionário com o valor da entrada
        entry_values[key] = entry_dict[key].get()

    def save_entry_values():
        # Guardar todos os valores do dicionário
        for key in sets.keys():
            entry_values[key] = entry_dict[key].get()
        #print(entry_values) 
        output.insert(tk.END, "BC guardadas com sucesso!") 

    var_dict = {}
    entry_dict = {}
    unit_dict = {}

    for key, value in sets.items():
        # Frame para cada item no dicionário
        frame = tk.Frame(BC_janela, padx=10, pady=10, relief="solid", borderwidth=1)
        frame.pack(fill='x', padx=10, pady=5)

        # Frame interno para organizar horizontalmente
        inner_frame = tk.Frame(frame)
        inner_frame.pack(fill='x')

        # Label com o nome do conjunto
        label = tk.Label(inner_frame, text=str(key), font=("Helvetica", 12, "bold"))
        label.pack(side='left', padx=5)

        # variável única para cada conjunto
        var_dict[key] = tk.IntVar(value=1)  # Valor padrão: Temperatura constante

        # Criando os RadioButtons
        radio1 = tk.Radiobutton(inner_frame, text="Temperatura Constante", variable=var_dict[key], value=1, command=lambda k=key: update_entry(k))
        radio2 = tk.Radiobutton(inner_frame, text="Fluxo de calor constante", variable=var_dict[key], value=2, command=lambda k=key: update_entry(k))
        radio3 = tk.Radiobutton(inner_frame, text="PDE", variable=var_dict[key], value=3, command=lambda k=key: update_entry(k))

        radio1.pack(side='left', padx=5)
        radio2.pack(side='left', padx=5)
        radio3.pack(side='left', padx=5)

        # Criando a Entry (inicialmente visível)
        entry_dict[key] = tk.Entry(inner_frame)
        entry_dict[key].pack(side='left', padx=5, fill='x', expand=True)

        # Criando o Label da unidade
        unit_dict[key] = tk.Label(inner_frame, text="°C")  # Padrão para temperatura constante
        unit_dict[key].pack(side='left', padx=5)

        # Adicionando trace para atualizar automaticamente
        var_dict[key].trace_add("write", lambda *args, k=key: update_entry(k))

    # Atualizar todas as entradas para o estado inicial correto
    for key in sets.keys():
        update_entry(key)

    # Criando um botão para salvar os valores
    btn_save = tk.Button(BC_janela, text="Guardar", command=save_entry_values)
    btn_save.pack(pady=10)

def IC():
    IC_janela = tk.Toplevel(root)
    IC_janela.title("IC")

    # Label da janela
    label_param = tk.Label(IC_janela, text="Condições Iniciais", font=("Helvetica", 10, "bold"))
    label_param.pack(pady=10)



#frame entry+label
file_frame = tk.Frame(root)
file_frame.pack()

#entrada e variavel
entry=tk.Entry(file_frame, textvariable=filevar)
entry.pack(side=tk.LEFT,pady=5,padx=5)

#label
g_btn = tk.Button(file_frame, text= "Guardar", command=lambda: guardar_parametros())
g_btn.pack(side=tk.RIGHT)

# Frame butons+text
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# textbox output
output = tk.Text(main_frame,height=20, width=40)
output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5, padx = 5)

# Frame buttons
button_frame = tk.Frame(main_frame)
button_frame.pack(side=tk.RIGHT, anchor='n', padx=5, pady=5)

#chamar classe nos e entidades fisicas
N_pe = nodes_physics.nodes_pe(filevar,output)

#parametrizar
btn_p = tk.Button(button_frame, text="Parametrizar", command=parametrizar)
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
btn_bc = tk.Button(button_frame, text="BC", command=BC)
btn_bc.pack(pady=5)

#botao ic
btn_Ic = tk.Button(button_frame, text="IC", command=IC)
btn_Ic.pack(pady=5)

# #btn set atribute
# att=tk.IntVar()
# entryatt= tk.Entry(root,textvariable=att)
# entryatt.pack(pady=5,padx=5)
# def setatt():
#     setattr(NN, "teste1", att.get())
#     NN.testesimples()
# btnatt = tk.Button(root, text="set atribute",command=setatt)
# btnatt.pack(padx=10,pady=5)

root.mainloop()

