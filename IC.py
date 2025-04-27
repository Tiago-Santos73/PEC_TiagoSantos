import tkinter as tk
import torch 
torch.set_default_dtype(torch.float32)

class ICGUI:
    def __init__(self, root, sets, output, bc_instance):
        
        
        self.root = root
        self.sets = sets
        self.output = output
        self.bc_instance = bc_instance
        self.entry_dict = {}        # Store Entry widgets
        self.ic_values = {}         # Store IC values 
    
    def create_window(self):
        self.IC_janela = tk.Toplevel(self.root)
        self.IC_janela.title("IC")
        

        #Label
        label_param = tk.Label(self.IC_janela, text="Condições Iniciais", font=("Helvetica", 16, "bold"))
        label_param.pack(pady=10)
        

        # Creates entry 
        self.create_ic_entries()
        

        # Save Button
        btn_save = tk.Button(self.IC_janela, text="Guardar", command=self.save_ic_values)
        btn_save.pack(pady=10)

    
    def create_ic_entries(self):

        for key, var in self.bc_instance.var_dict.items():
            # Only create entries for boundary conditions that have a PDE (type 3)

            if var.get() == 3:  #If PDE

                #Frame
                frame = tk.Frame(self.IC_janela, padx=10, pady=10, relief="solid", borderwidth=1)
                frame.pack(fill='x', padx=10, pady=5)
                
                inner_frame = tk.Frame(frame)
                inner_frame.pack(fill='x')
                

                #Label
                label = tk.Label(inner_frame, text=f"{key}   Temperatura Inicial:", font=("Helvetica", 12, "bold"))
                label.pack(side='left', padx=5)
                
                #Entry
                entry = tk.Entry(inner_frame)
                entry.pack(side='left', padx=5, fill='x', expand=True)

                #Unit Label
                unit_label = tk.Label(inner_frame, text="°C", font=("Helvetica", 12))
                unit_label.pack(side='left', padx=5)
                
                #Store Entry Widget
                self.entry_dict[key] = entry
    

    def save_ic_values(self):
        
        #Save
        self.ic_values = {key: float(self.entry_dict[key].get()) for key in self.entry_dict}
        
        #Display output
        self.output.insert(tk.END, f"Condições iniciais guardadas guardadas com sucesso!\n")
        
        #print(self.ic_values)
        
        #close window
        self.IC_janela.destroy()
