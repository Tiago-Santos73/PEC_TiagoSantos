import tkinter as tk
import numpy as np
import torch
torch.set_default_dtype(torch.float32)

class BCGUI:
    def __init__(self, root, sets, output):
        self.root = root
        self.sets = sets
        self.output = output
        self.entry_values = {}
        self.var_dict = {}
        self.entry_dict = {}
        self.unit_dict = {}
        self.unidades = {1: "°C", 2: "W/m²"}  # Temperatura e fluxo de calor
        self.dimension_var = tk.IntVar(value=3)
        self.new_sets = {}  
        #self.create_window()
    
    def create_window(self):
        self.BC_janela = tk.Toplevel(self.root)
        self.BC_janela.title("BC")
        
        label_param = tk.Label(self.BC_janela, text="Condições de Fronteira", font=("Helvetica", 16, "bold"))
        label_param.pack(pady=10)
        
        self.create_dimension_selector()
        self.create_bc_entries()
        
        btn_save = tk.Button(self.BC_janela, text="Guardar", command=self.save_entry_values)
        btn_save.pack(pady=10)
    
    def create_dimension_selector(self):
        dimension_frame = tk.Frame(self.BC_janela)
        dimension_frame.pack(pady=10)
        
        tk.Radiobutton(dimension_frame, text="1D", variable=self.dimension_var, value=1, font=("Helvetica", 12)).pack(side='left', padx=5)
        tk.Radiobutton(dimension_frame, text="2D", variable=self.dimension_var, value=2, font=("Helvetica", 12)).pack(side='left', padx=5)
        tk.Radiobutton(dimension_frame, text="3D", variable=self.dimension_var, value=3, font=("Helvetica", 12)).pack(side='left', padx=5)
    
    def create_bc_entries(self):
        for key, value in self.sets.items():
            frame = tk.Frame(self.BC_janela, padx=10, pady=10, relief="solid", borderwidth=1)
            frame.pack(fill='x', padx=10, pady=5)
            
            inner_frame = tk.Frame(frame)
            inner_frame.pack(fill='x')
            
            label = tk.Label(inner_frame, text=str(key), font=("Helvetica", 12, "bold"))
            label.pack(side='left', padx=5)
            
            self.var_dict[key] = tk.IntVar(value=1)
            
            for value, text in [(1, "Temperatura Constante"), (2, "Fluxo de Calor Constante"), (3, "PDE")]:
                tk.Radiobutton(inner_frame, text=text, variable=self.var_dict[key], value=value, command=lambda k=key: self.update_entry(k)).pack(side='left', padx=5)
            
            self.entry_dict[key] = tk.Entry(inner_frame)
            self.entry_dict[key].pack(side='left', padx=5, fill='x', expand=True)
            
            self.unit_dict[key] = tk.Label(inner_frame, text="°C")
            self.unit_dict[key].pack(side='left', padx=5)
            
            self.var_dict[key].trace_add("write", lambda *args, k=key: self.update_entry(k))
            self.update_entry(key)
    
    def update_entry(self, key, *args):
        selected_value = self.var_dict[key].get()
        if selected_value == 3:
            self.entry_dict[key].pack_forget()
            self.unit_dict[key].pack_forget()
        else:
            self.entry_dict[key].pack(side='left', padx=5, fill='x', expand=True)
            self.unit_dict[key].config(text=self.unidades[selected_value])
            self.unit_dict[key].pack(side='left', padx=5)
        self.entry_values[key] = self.entry_dict[key].get()
    
    def save_entry_values(self):
        for key in self.sets.keys():
            self.entry_values[key] = self.entry_dict[key].get()
        
        dimension = self.dimension_var.get()
        self.new_sets = {}
        non_pde_points = []  
        
        for key, value in self.sets.items():
            if self.var_dict[key].get() != 3:
                if dimension == 1:
                    self.new_sets[key] = value[:, :-2]
                elif dimension == 2:
                    self.new_sets[key] = value[:, :-1]
                else:
                    self.new_sets[key] = value
                non_pde_points.append(self.new_sets[key])

        if non_pde_points:
            all_non_pde_points = torch.cat(non_pde_points, dim=0)
        else:
            all_non_pde_points = torch.tensor([])

        for key, value in self.sets.items():
            if self.var_dict[key].get() == 3:  
                if dimension == 1:
                    processed_value = value[:, :-2]
                elif dimension == 2:
                    processed_value = value[:, :-1]
                else:
                    processed_value = value

                if all_non_pde_points.numel() > 0:
        
        
                    a = processed_value.unsqueeze(1)  
                    b = all_non_pde_points.unsqueeze(0)  

                    # Find matches
                    matches = torch.all(a == b, dim=2).any(dim=1)

                    # Keep only non-matching points
                    self.new_sets[key] = processed_value[~matches]
                else:
                    self.new_sets[key] = processed_value
            
         
        
        all_points = torch.cat(list(self.new_sets.values()), dim=0)
        unique_points, counts = torch.unique(all_points, dim=0, return_counts=True)
        duplicate_points = unique_points[counts > 1]

        if duplicate_points.numel() > 0:
            for key, points in self.new_sets.items():
                a = points.unsqueeze(1)
                b = duplicate_points.unsqueeze(0)
                matches = torch.all(a == b, dim=2).any(dim=1)
                self.new_sets[key] = points[~matches]

        # with open("new_sets.txt", "w") as f:
        #         for key, tensor in self.new_sets.items():
        #             f.write(f"Key: {key}\n")
        #             f.write(str(tensor.tolist()) + "\n\n")

        # for key in self.new_sets :
        #     print(f"dtype of self.new_sets['{key}'] {self.new_sets[key].dtype}")


        self.BC_janela.destroy()

        
        #print("new_sets criado com sucesso:\n", self.new_sets)
        # print(self.entry_values)
        # for key in self.sets.keys():
        #      print(self.var_dict[key].get())
        
        self.output.insert(tk.END, "BC guardadas com sucesso!\n")
