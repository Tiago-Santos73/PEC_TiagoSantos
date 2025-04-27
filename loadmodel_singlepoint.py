import tkinter as tk
import torch
torch.set_default_dtype(torch.float32)
import os
import numpy as np
import matplotlib.pyplot as plt

class SinglePointInference:
    def __init__(self, root, params, output, ic, bc):
        self.root = root
        self.output = output
        self.ic = ic
        self.parameters = params
        self.bc = bc

        # Define the filename of the model
        self.filename = self.parameters['nodes file']
        self.modelfilenamestr = os.path.splitext(self.filename)[0] + "_model.pth"
        self.modelfilename = tk.StringVar(value=self.modelfilenamestr)

        # Set the initial temperature from initial conditions
        tvalue = (float(self.ic.ic_values[key]) for key in self.ic.entry_dict)
        self.initial_temperature = tk.DoubleVar(value=(next(tvalue)))

        # Initialize variables for x, y coordinates and time
        self.x_coord = tk.DoubleVar()
        self.y_coord = tk.DoubleVar()
        self.time_val = tk.DoubleVar()

        #Initialize max and min temperature
        tmax = float('-inf')
        tmin = float('inf')

        #Loop over Temperatures to get min and max
        for key, var in self.bc.var_dict.items():
            if var.get() != 3:
                if float(self.bc.entry_values[key]) < tmin:
                    tmin = float(self.bc.entry_values[key])
                if float(self.bc.entry_values[key]) > tmax:
                    tmax = float(self.bc.entry_values[key])

        for key in self.ic.ic_values:
            if float(self.ic.ic_values[key]) < tmin:
                tmin = float(self.ic.ic_values[key])
            if float(self.ic.ic_values[key]) > tmax:
                tmax = float(self.ic.ic_values[key])

        self.bc_adim = {}
        self.ic_adim = {}

        #Convert BC to adimensional
        for key, var in self.bc.var_dict.items():
            if var.get() != 3:
                self.bc_adim[key] = ((float(self.bc.entry_values[key])-tmin)/(tmax-tmin)) * (self.parameters['adim dom'][1]-self.parameters['adim dom'][0]) + self.parameters['adim dom'][0]

        #Convert IC to adimensional
        for key in self.ic.ic_values:
            self.ic_adim[key] = ((float(self.ic.ic_values[key])-tmin)/(tmax-tmin)) * (self.parameters['adim dom'][1]-self.parameters['adim dom'][0]) + self.parameters['adim dom'][0]

        self.tmin = tmin
        self.tmax = tmax
        self.model = None  

    
    # Create a single point tensor at a specific time
    def create_domain_at_time(self):
        
        #Adimensional
        tc_adim = ((float(self.initial_temperature.get())-self.tmin)/(self.tmax-self.tmin)) * (self.parameters['adim dom'][1]-self.parameters['adim dom'][0]) + self.parameters['adim dom'][0]
        
        point = torch.tensor([[self.x_coord.get(), self.y_coord.get(), self.time_val.get(), tc_adim]], dtype=torch.float32)
        self.domain = point

        print("Mesh domain created!!!")
        print(self.domain)

    # Create multiple points for a single (x, y) location across different times
    def create_domain_across_time(self):

        #Adimensional
        tc_adim = ((float(self.initial_temperature.get())-self.tmin)/(self.tmax-self.tmin)) * (self.parameters['adim dom'][1]-self.parameters['adim dom'][0]) + self.parameters['adim dom'][0]

        start_time = float(self.parameters['t domain'][0])
        end_time = float(self.parameters['t domain'][1])
        num_time_steps = 200  # Number of time steps for the time evolution

        #equal spaced time points
        time_points = torch.linspace(start_time, end_time, num_time_steps).unsqueeze(1)

        #get coordinates
        x_val = self.x_coord.get()
        y_val = self.y_coord.get()

        #concatenate
        x_tensor = torch.full((num_time_steps, 1), x_val)
        y_tensor = torch.full((num_time_steps, 1), y_val)
        tc_tensor = torch.full((num_time_steps, 1), tc_adim)
        points = torch.cat((x_tensor, y_tensor, time_points, tc_tensor), dim=1)
        self.domain = points
        
        print("Mesh domain created!!!")
        print(self.domain)

    def load_model(self):
        # Load model
        print("Loading model...")
        self.model = torch.load(self.modelfilename.get(), map_location='cpu',weights_only=False)
        #
        # Set to evaluation mode
        self.model.eval()
        print("Model loaded.\n")   

    @torch.no_grad()
    def inference(self):
        # include the time into the domain

        self.domain = self.domain.to(torch.float32)
        
        #adimensional results
        self.results_adim = self.model(self.domain)
        #torch.set_printoptions(threshold=float('inf'))
        #print(self.results_adim)

        #dimensional results
        self.results = ((self.results_adim - self.parameters['adim dom'][0])/(self.parameters['adim dom'][1]-self.parameters['adim dom'][0])) * (self.tmax - self.tmin) + self.tmin 
        print(self.results)


    # Evaluate a single point at a specific time
    def evaluate_single_point(self, janela=None):
        self.create_domain_at_time()
        self.load_model()
        self.inference()
        
        self.output.insert(tk.END, f"Temperatura no instante {self.time_val.get()} é {self.results.item():.2f}\n")
        
        if janela:
            janela.destroy()

    # Evaluate temperature evolution across time for a fixed point
    def evaluate_across_time(self, janela):
        self.create_domain_across_time()
        self.load_model()
        self.inference()


        # Save results to a .txt file
        domain_np = self.domain.detach().cpu().numpy()
        results_np = self.results.detach().cpu().numpy()
        xyt = domain_np[:, :3]
        save_data = np.hstack((xyt, results_np))
        np.savetxt(f"Resultados_{os.path.splitext(self.parameters['nodes file'])[0]}_X_{self.x_coord.get()}_Y_{self.y_coord.get()}.txt", save_data, 
                   header="x y tempo temperatura", fmt="%.6f", comments='')

        self.output.insert(tk.END, "Modelo Avaliado\n")


        # Sort results by time for correct plotting
        time = self.domain[:, -2].detach().cpu().numpy().flatten()
        temps = self.results.detach().cpu().numpy().flatten()

        sorted_indices = np.argsort(time)
        time_sorted = time[sorted_indices]
        temps_sorted = temps[sorted_indices]

        # Plot matplotlib temperature vs. time
        plt.figure(figsize=(8, 5))
        plt.plot(time_sorted, temps_sorted, color="red", linewidth=2)
        plt.title("Evolução da Temperatura ao Longo do Tempo")
        plt.xlabel("Tempo")
        plt.ylabel("Temperatura (°C)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        #close window
        if janela:
            janela.destroy()


    # Create a window for user to input evaluation parameters
    def janela(self):
        evaluate_janela = tk.Toplevel(self.root)
        evaluate_janela.title("Avaliar Ponto")

        label_eval = tk.Label(evaluate_janela, text="Avaliação", font=("Helvetica", 14, "bold"))
        label_eval.pack(pady=10)

        #frame
        entry_frame = tk.Frame(evaluate_janela)
        entry_frame.pack(fill=tk.BOTH, expand=True)
        entry_frame.grid_rowconfigure(0, weight=1)
        entry_frame.grid_rowconfigure(1, weight=1)
        entry_frame.grid_rowconfigure(2, weight=1)
        entry_frame.grid_rowconfigure(3, weight=1)
        entry_frame.grid_rowconfigure(4, weight=1)
        entry_frame.grid_rowconfigure(5, weight=1)
        entry_frame.grid_rowconfigure(6, weight=1)
        entry_frame.grid_columnconfigure(0, weight=1)
        entry_frame.grid_columnconfigure(1, weight=1)

        # Model File
        model_entry = tk.Entry(entry_frame, textvariable=self.modelfilename)#, width=40
        model_entry.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        model_label = tk.Label(entry_frame, text="Modelo")#width=15,, anchor="center"
        model_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # X Coordinate
        x_entry = tk.Entry(entry_frame, textvariable=self.x_coord)
        x_entry.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        x_label = tk.Label(entry_frame, text="Coordenada X")
        x_label.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Y Coordinate
        y_entry = tk.Entry(entry_frame, textvariable=self.y_coord)
        y_entry.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        y_label = tk.Label(entry_frame, text="Coordenada Y")
        y_label.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Initial Temperature
        ic_entry = tk.Entry(entry_frame, textvariable=self.initial_temperature)
        ic_entry.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        ic_label = tk.Label(entry_frame, text="Temperatura Inicial")
        ic_label.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        # Time
        time_entry = tk.Entry(entry_frame, textvariable=self.time_val)
        time_entry.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

        time_label = tk.Label(entry_frame, text="Instante de Tempo")
        time_label.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        # Buttons Frame 
        button_frame = tk.Frame(evaluate_janela)
        button_frame.pack(pady=5)  
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        # Button evaluate at a single point
        evaluate_btn = tk.Button(button_frame, text="Avaliar Instante de Tempo", command=lambda: self.evaluate_single_point(evaluate_janela))
        evaluate_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Button evaluate across time
        evaluate_time_across_btn = tk.Button(button_frame, text="Avaliar ao Longo do Tempo", command=lambda: self.evaluate_across_time(evaluate_janela))
        evaluate_time_across_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")