import tkinter as tk
import numpy as np
import torch
torch.set_default_dtype(torch.float32)

class BCGUI:
    def __init__(self, root, sets, output):

        self.root = root
        self.sets = sets        # Points sets
        self.output = output
        self.entry_values = {}      # Store the values from the entries      
        self.var_dict = {}          # Store Options variables  
        self.entry_dict = {}        # Store Text Entry Widgets
        self.unit_dict = {}
        self.unidades = {1: "°C", 2: "W/m²"} 
        self.dimension_var = tk.IntVar(value=3)
        self.new_sets = {}          #Store Points sets after BC
        #self.create_window()
    
    def create_window(self):
        self.BC_janela = tk.Toplevel(self.root)
        self.BC_janela.title("BC")
        
        
        #Label
        label_param = tk.Label(self.BC_janela, text="Condições de Fronteira", font=("Helvetica", 16, "bold"))
        label_param.pack(pady=10)
        
        
        #Call Functions
        self.create_dimension_selector()
        self.create_bc_entries()
        
        #Save
        btn_save = tk.Button(self.BC_janela, text="Guardar", command=self.save_entry_values)
        btn_save.pack(pady=10)
    

    #Crete Dimension Selectors
    def create_dimension_selector(self):
        dimension_frame = tk.Frame(self.BC_janela)
        dimension_frame.pack(pady=10)
        
        #Radio Buttons
        tk.Radiobutton(dimension_frame, text="1D", variable=self.dimension_var, value=1, font=("Helvetica", 12)).pack(side='left', padx=5)
        tk.Radiobutton(dimension_frame, text="2D", variable=self.dimension_var, value=2, font=("Helvetica", 12)).pack(side='left', padx=5)
        tk.Radiobutton(dimension_frame, text="3D", variable=self.dimension_var, value=3, font=("Helvetica", 12)).pack(side='left', padx=5)
    

    def create_bc_entries(self):

        #Create Entry for each Set
        for key, value in self.sets.items():

            #Frame
            frame = tk.Frame(self.BC_janela, padx=10, pady=10, relief="solid", borderwidth=1)
            frame.pack(fill='x', padx=10, pady=5)
            
            inner_frame = tk.Frame(frame)
            inner_frame.pack(fill='x')
            
            #Set Name
            label = tk.Label(inner_frame, text=str(key), font=("Helvetica", 12, "bold"))
            label.pack(side='left', padx=5)


            #Variable to store option
            self.var_dict[key] = tk.IntVar(value=1)
            

            # Creates Radio Buttons for each type of BC
            for value, text in [(1, "Temperatura Constante"), (2, "Fluxo de Calor Constante"), (3, "PDE")]:
                tk.Radiobutton(inner_frame, text=text, variable=self.var_dict[key], value=value, command=lambda k=key: self.update_entry(k)).pack(side='left', padx=5)
            

            #Entry
            self.entry_dict[key] = tk.Entry(inner_frame)
            self.entry_dict[key].pack(side='left', padx=5, fill='x', expand=True)

            
            #Unit Label
            self.unit_dict[key] = tk.Label(inner_frame, text="°C")
            self.unit_dict[key].pack(side='left', padx=5)
            
            #Trace variable changes and update interface
            self.var_dict[key].trace_add("write", lambda *args, k=key: self.update_entry(k))
            self.update_entry(key)
    

    def update_entry(self, key, *args):


        # Updates the entry field and unit label based on the user's selection
        selected_value = self.var_dict[key].get()
        if selected_value == 3:

            # If the choice is "PDE", hides the entry field and unit label
            self.entry_dict[key].pack_forget()
            self.unit_dict[key].pack_forget()

        else:

            #Show the entry field and updates the unit label
            self.entry_dict[key].pack(side='left', padx=5, fill='x', expand=True)
            self.unit_dict[key].config(text=self.unidades[selected_value])
            self.unit_dict[key].pack(side='left', padx=5)

         # Stores values   
        self.entry_values[key] = self.entry_dict[key].get()

    # Stores values
    def save_entry_values(self):

        for key in self.sets.keys():
            self.entry_values[key] = self.entry_dict[key].get()
        
        # Get dimension 
        dimension = self.dimension_var.get()

        self.new_sets = {}      # Store Points sets after BC
        non_pde_points = []     # List to store non-PDE points
        
        #Processes the points for BC
        for key, value in self.sets.items():

            # Reduces the dimensions
            if self.var_dict[key].get() != 3: # If not PDE

                if dimension == 1:
                    self.new_sets[key] = value[:, :-2]
                elif dimension == 2:
                    self.new_sets[key] = value[:, :-1]
                else:
                    self.new_sets[key] = value
                non_pde_points.append(self.new_sets[key])

        # Concatenates all non-PDE points
        if non_pde_points:
            all_non_pde_points = torch.cat(non_pde_points, dim=0)
        else:
            all_non_pde_points = torch.tensor([])

        #Processes the points for PDE
        for key, value in self.sets.items():
            
            # Reduces the dimensions
            if self.var_dict[key].get() == 3:  # If PDE
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
            
         
        # Concatenates all processed points
        all_points = torch.cat(list(self.new_sets.values()), dim=0)
        
        # Finds duplicate points
        unique_points, counts = torch.unique(all_points, dim=0, return_counts=True)
        duplicate_points = unique_points[counts > 1]

        # If duplicate points, removes them
        if duplicate_points.numel() > 0:
            for key, points in self.new_sets.items():
                a = points.unsqueeze(1)
                b = duplicate_points.unsqueeze(0)
                matches = torch.all(a == b, dim=2).any(dim=1)
                self.new_sets[key] = points[~matches]

        #
        # with open("new_sets.txt", "w") as f:
        #         for key, tensor in self.new_sets.items():
        #             f.write(f"Key: {key}\n")
        #             f.write(str(tensor.tolist()) + "\n\n")

        # for key in self.new_sets :
        #     print(f"dtype of self.new_sets['{key}'] {self.new_sets[key].dtype}")


        # Closes BC window
        self.BC_janela.destroy()

        
        #print("new_sets criado com sucesso:\n", self.new_sets)
        # print(self.entry_values)
        # for key in self.sets.keys():
        #      print(self.var_dict[key].get())
        

        # Displays message
        self.output.insert(tk.END, "BC guardadas com sucesso!\n")
