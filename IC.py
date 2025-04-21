import tkinter as tk
import torch 
torch.set_default_dtype(torch.float32)

class ICGUI:
    def __init__(self, root, sets, output, bc_instance):
        self.root = root
        self.sets = sets
        self.output = output
        self.bc_instance = bc_instance
        self.entry_dict = {}
        self.ic_values = {}
    
    def create_window(self):
        self.IC_janela = tk.Toplevel(self.root)
        self.IC_janela.title("IC")
        
        label_param = tk.Label(self.IC_janela, text="Condições Iniciais", font=("Helvetica", 16, "bold"))
        label_param.pack(pady=10)
        
        self.create_ic_entries()
        
        btn_save = tk.Button(self.IC_janela, text="Guardar", command=self.save_ic_values)
        btn_save.pack(pady=10)
    
    def create_ic_entries(self):
        for key, var in self.bc_instance.var_dict.items():
            if var.get() == 3:  # Apenas para chaves definidas com PDE
                frame = tk.Frame(self.IC_janela, padx=10, pady=10, relief="solid", borderwidth=1)
                frame.pack(fill='x', padx=10, pady=5)
                
                inner_frame = tk.Frame(frame)
                inner_frame.pack(fill='x')
                
                label = tk.Label(inner_frame, text=f"{key}   Temperatura Inicial:", font=("Helvetica", 12, "bold"))
                label.pack(side='left', padx=5)
                
                entry = tk.Entry(inner_frame)
                entry.pack(side='left', padx=5, fill='x', expand=True)
                
                unit_label = tk.Label(inner_frame, text="°C", font=("Helvetica", 12))
                unit_label.pack(side='left', padx=5)
                
                self.entry_dict[key] = entry
    
    def save_ic_values(self):
        self.ic_values = {key: float(self.entry_dict[key].get()) for key in self.entry_dict}
        self.output.insert(tk.END, f"Condições iniciais guardadas guardadas com sucesso!\n")
        #print(self.ic_values)
        self.IC_janela.destroy()
