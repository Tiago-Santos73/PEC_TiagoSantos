import tkinter as tk
import ast

class ParametrosNN:
    d= {}

    def __init__(self, root, output, filevar):
        
        self.root = root
        self.output = output
        self.filevar = filevar
        
        #Initialize Parameters with default values
        self.d_inputs = tk.IntVar(value=4)
        self.d_layers_nnumber = tk.IntVar(value=50)
        self.d_layersnum = tk.IntVar(value=7)
        self.d_outputs = tk.IntVar(value=1)
        self.d_epochs = tk.IntVar(value=20000)
        self.d_lr = tk.StringVar(value="1e-03")
        self.d_domain_samples = tk.IntVar(value=500)
        self.d_Ic = tk.IntVar(value=250)
        self.d_Bc = tk.IntVar(value=250)
        self.d_tdom = tk.StringVar(value="[0.0, 1.0]")
        self.d_batches = tk.IntVar(value=500_000)
        self.d_alphax = tk.StringVar(value="1.0")
        self.d_alphay = tk.StringVar(value="1.0")
        self.d_adim = tk.StringVar(value="[0.25, 0.50]")

    #Save the parameters
    def guardar_parametros(self, janela=None):

        ParametrosNN.d['inputs'] = self.d_inputs.get()
        ParametrosNN.d['layers'] = [self.d_layers_nnumber.get()]*self.d_layersnum.get()
        ParametrosNN.d['outputs'] = self.d_outputs.get()
        ParametrosNN.d['epochs'] = self.d_epochs.get()
        ParametrosNN.d['lr'] = float(self.d_lr.get())
        #ParametrosNN.d['domain samples'] = self.d_domain_samples.get()
        #ParametrosNN.d['IC samples'] = self.d_Ic.get()
        #ParametrosNN.d['BC samples'] = self.d_Bc.get()
        ParametrosNN.d['nodes file'] = self.filevar.get()
        ParametrosNN.d['t domain'] = ast.literal_eval(self.d_tdom.get())
        ParametrosNN.d['batch size'] = self.d_batches.get()
        # ParametrosNN.d['alphax'] = float(self.d_alphax.get())
        # ParametrosNN.d['alphay'] = float(self.d_alphay.get())
        ParametrosNN.d['T0'] = 10
        ParametrosNN.d['adim dom'] = ast.literal_eval(self.d_adim.get())
        
        #Display Output
        self.output.insert(tk.END, "Parametrização guardada!\n")
        
        #Close Window
        if janela:
            janela.destroy()


    #Open Window
    def parametrizar(self):
        parametrizar_janela = tk.Toplevel(self.root)
        parametrizar_janela.title("Parametrizar")

        #Window Label
        label_param = tk.Label(parametrizar_janela, text="Parametrização", font=("Helvetica", 14, "bold"))
        label_param.pack(pady=10)
        
        #Frame
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
        entry_frame.grid_rowconfigure(9, weight=1)
        entry_frame.grid_rowconfigure(10, weight=1)
        entry_frame.grid_rowconfigure(11, weight=1)
        entry_frame.grid_rowconfigure(12, weight=1)
        entry_frame.grid_columnconfigure(0, weight=1)
        entry_frame.grid_columnconfigure(1, weight=1)


         # Inputs
        d_inputs_entry = tk.Entry(entry_frame, textvariable=self.d_inputs)
        d_inputs_entry.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        d_inputs_label = tk.Label(entry_frame, text="Camada de Entrada")
        d_inputs_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Frame Layers
        layers_frame = tk.Frame(entry_frame)
        layers_frame.grid(row=1, column=0, sticky="ew")
        layers_frame.grid_rowconfigure(0, weight=1)
        layers_frame.grid_columnconfigure(0, weight=1)
        layers_frame.grid_columnconfigure(1, weight=1)
        layers_frame.grid_columnconfigure(2, weight=1)
        
        # Layers
        d_layers_nnumber_entry = tk.Entry(layers_frame, textvariable=self.d_layers_nnumber)
        d_layers_nnumber_entry.grid(row=0, column=0,padx=5, pady=5, sticky="ew")
        
        d_layers_label1 = tk.Label(layers_frame, text="*")
        d_layers_label1.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        d_layersnum_entry = tk.Entry(layers_frame, textvariable=self.d_layersnum)
        d_layersnum_entry.grid(row=0, column=2, pady=5, sticky="ew")
        
        d_layers_label2 = tk.Label(entry_frame, text="Camadas Ocultas")
        d_layers_label2.grid(row=1, column=1, pady=5, sticky="ew")

        # Outputs
        d_outputs_entry = tk.Entry(entry_frame, textvariable=self.d_outputs)
        d_outputs_entry.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        d_outputs_label = tk.Label(entry_frame, text="Camada de Saída")
        d_outputs_label.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Epochs
        d_epochs_entry = tk.Entry(entry_frame, textvariable=self.d_epochs)
        d_epochs_entry.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        
        d_epochs_label = tk.Label(entry_frame, text="Épocas de Treino ")
        d_epochs_label.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        # lr
        d_lr_entry = tk.Entry(entry_frame, textvariable=self.d_lr)
        d_lr_entry.grid(row=4, column=0, padx=5, pady=5, sticky="ew")
        
        d_lr_label = tk.Label(entry_frame, text="Taxa de Aprendizagem")
        d_lr_label.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        # # Domain Samples
        # d_domain_samples_entry = tk.Entry(entry_frame, textvariable=self.d_domain_samples)
        # d_domain_samples_entry.grid(row=5, column=0, padx=5, pady=5, sticky="ew")
        
        # d_domain_samples_label = tk.Label(entry_frame, text="Domain Samples")
        # d_domain_samples_label.grid(row=5, column=1, padx=5, pady=5, sticky="ew")

        # # IC Samples
        # d_Ic_entry = tk.Entry(entry_frame, textvariable=self.d_Ic)
        # d_Ic_entry.grid(row=6, column=0, padx=5, pady=5, sticky="ew")
        
        # d_Ic_label = tk.Label(entry_frame, text="IC Samples")
        # d_Ic_label.grid(row=6, column=1, padx=5, pady=5, sticky="ew")

        # # BC Samples (Boundary Condition Samples)
        # d_Bc_entry = tk.Entry(entry_frame, textvariable=self.d_Bc)
        # d_Bc_entry.grid(row=7, column=0, padx=5, pady=5, sticky="ew")
        
        # d_Bc_label = tk.Label(entry_frame, text="BC Samples")
        # d_Bc_label.grid(row=7, column=1, padx=5, pady=5, sticky="ew")

        # Nodes File
        d_nf_entry = tk.Entry(entry_frame, textvariable=self.filevar)
        d_nf_entry.grid(row=8, column=0, padx=5, pady=5, sticky="ew")
        
        d_nf_label = tk.Label(entry_frame, text="Ficheiro Nós")
        d_nf_label.grid(row=8, column=1, padx=5, pady=5, sticky="ew")

        #Batches
        d_b_entry = tk.Entry(entry_frame, textvariable=self.d_batches)
        d_b_entry.grid(row=9, column=0, padx=5, pady=5, sticky="ew")
        
        d_b_label = tk.Label(entry_frame, text="Tamanho do Lote")
        d_b_label.grid(row=9, column=1, padx=5, pady=5, sticky="ew")

        #Time Domain
        d_t_entry = tk.Entry(entry_frame, textvariable=self.d_tdom)
        d_t_entry.grid(row=10, column=0, padx=5, pady=5, sticky="ew")
        
        d_t_label = tk.Label(entry_frame, text="Domínio Temporal")
        d_t_label.grid(row=10, column=1, padx=5, pady=5, sticky="ew")

        #Adimensional DOmain
        d_adim_entry = tk.Entry(entry_frame, textvariable=self.d_adim)
        d_adim_entry.grid(row=11, column=0, padx=5, pady=5, sticky="ew")
        
        d_adim_label = tk.Label(entry_frame, text="Limites Adimensionais")
        d_adim_label.grid(row=11, column=1, padx=5, pady=5, sticky="ew")

        # #alphax
        # d_alphax_entry = tk.Entry(entry_frame, textvariable=self.d_alphax)
        # d_alphax_entry.grid(row=11, column=0, padx=5, pady=5, sticky="ew")
        
        # d_alphax_label = tk.Label(entry_frame, text="Difusividade x")
        # d_alphax_label.grid(row=11, column=1, padx=5, pady=5, sticky="ew")

        # #alphay
        # d_alphay_entry = tk.Entry(entry_frame, textvariable=self.d_alphay)
        # d_alphay_entry.grid(row=12, column=0, padx=5, pady=5, sticky="ew")
        
        # d_alphay_label = tk.Label(entry_frame, text="Difusividade y")
        # d_alphay_label.grid(row=12, column=1, padx=5, pady=5, sticky="ew")


        #Save and Close
        btn_fechar = tk.Button(parametrizar_janela, text="Guardar", command=lambda: self.guardar_parametros(parametrizar_janela))
        btn_fechar.pack(pady=5)

        


