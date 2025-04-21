import tkinter as tk
import nodes_physics
import oldv.job as job
import torch
torch.set_default_dtype(torch.float32)
import parametros


#inicializar janela e parametros
root = tk.Tk()
root.title("Teste")




def create_NN():
    param_ui.guardar_parametros()
    global NN
    #print(d)
    NN = job.Plate(params=param_ui.d)
    output.insert(tk.END, "NN criada com sucesso!\n")

def ler():
    global sets
    sets = NN.read_mesh()
    output.insert(tk.END, "Rede lida com sucesso!\n")
    #print(sets)




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
btn_bc = tk.Button(button_frame, text="BC", command=BC)
btn_bc.pack(pady=5)

#botao ic
btn_Ic = tk.Button(button_frame, text="IC", command=IC)
btn_Ic.pack(pady=5)


root.mainloop()

