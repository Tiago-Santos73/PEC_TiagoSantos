import tkinter as tk
import torch
torch.set_default_dtype(torch.float32)
import read_nodes
import os
import read_nodes
import numpy as np
import pyvista as pv

class Inference:
    def __init__(self, root, params, output,ic,bc):

        self.root = root 
        self.output = output
        self.ic = ic
        self.parameters = params
        self.bc = bc

        self.filename = self.parameters['nodes file']

        self.modelfilenamestr = os.path.splitext(self.filename)[0] + "_model.pth"

        self.modelfilename = tk.StringVar(value=self.modelfilenamestr)
        self.nodefilename = tk.StringVar(value=self.filename)
        tvalue = (float(self.ic.ic_values[key]) for key in self.ic.entry_dict)
        self.tc = tk.DoubleVar(value=(next(tvalue)))
        self.t = tk.DoubleVar()

        tmax=float('-inf')
        tmin =float('inf')
        
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

        for key, var in self.bc.var_dict.items():
            if var.get() != 3:
                self.bc_adim[key] = ((float(self.bc.entry_values[key])-tmin)/(tmax-tmin)) * (self.parameters['adim dom'][1]-self.parameters['adim dom'][0]) + self.parameters['adim dom'][0]

        for key in self.ic.ic_values:
            self.ic_adim[key] = ((float(self.ic.ic_values[key])-tmin)/(tmax-tmin)) * (self.parameters['adim dom'][1]-self.parameters['adim dom'][0]) + self.parameters['adim dom'][0]

        self.tmin = tmin
        self.tmax = tmax

    def create_domain(self):

        all_points = []

        R = read_nodes.Read(self.nodefilename.get())
        self.sets = R.read()

        for key, points in self.sets.items():
            num_points = points.shape[0]
            
            tc_adim = ((float(self.tc.get())-self.tmin)/(self.tmax-self.tmin)) * (self.parameters['adim dom'][1]-self.parameters['adim dom'][0]) + self.parameters['adim dom'][0]

            t_values = torch.full((num_points, 1), float(self.t.get()), dtype=torch.float32)
            
            temp_values = torch.full((num_points, 1), tc_adim, dtype=torch.float32)
            
            new_data = torch.cat((points[:, :2], t_values, temp_values), dim=1)
            
            all_points.append(new_data)
        
        self.domain = torch.cat(all_points, dim=0) if all_points else torch.empty((0, 4), dtype=torch.float32)
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
        self.results_adim = self.model(self.domain)
        #torch.set_printoptions(threshold=float('inf'))
        #print(self.results_adim)
        self.results = ((self.results_adim - self.parameters['adim dom'][0])/(self.parameters['adim dom'][1]-self.parameters['adim dom'][0])) * (self.tmax - self.tmin) + self.tmin #############
        
        domain_np = self.domain.detach().cpu().numpy()
        results_np = self.results.detach().cpu().numpy()

        xy = domain_np[:, :2]

        save_data = np.hstack((xy, results_np))

        np.savetxt(f"Resultados{os.path.splitext(self.parameters['nodes file'])[0]}_.txt", save_data, 
                   header="x y temperatura", fmt="%.6f", comments='')
        
        print(self.results)

    def avaliar(self,janela=None):
        self.create_domain()
        self.load_model()
        self.inference()

        self.output.insert(tk.END, "Modelo Avaliado\n")

        # Extract coordinates (x,y) and temperatures
        pointsxy = self.domain[:, :2].detach().cpu().numpy()  # Shape: [N, 2]
        points_xyz = np.column_stack((pointsxy, np.zeros(len(pointsxy))))
        temps = self.results.detach().cpu().numpy().flatten()  # Shape: [N]

        # Create mesh and assign temperature data
        mesh = pv.PolyData(points_xyz)
        mesh["Temperature"] = temps

        # Create plotter
        plotter = pv.Plotter()
        plotter.add_mesh(
            mesh,
            scalars="Temperature",
            cmap="jet",  # Other options: "plasma", "viridis", "coolwarm", "hot", ,"turbo"
            point_size=10,  # Adjust based on your point density
            render_points_as_spheres=True,
            show_scalar_bar=True,
        )
        plotter.add_scalar_bar(
            title="Temperature (°C)",
            interactive=True,  # Allows adjusting color range
            vertical=True,  # Orientation
        )

        # Configure plotter
        plotter.background_color = "white"
        plotter.show_grid()  # Adds x/y axes
        #plotter.add_axes()  # Adds coordinate system indicator

        plotter.add_axes(
            xlabel='X Axis',
            ylabel='Y Axis',
            zlabel='',  # Empty for 2D plots
            line_width=2,
            labels_off=False,
            color='black'
        )
        plotter.view_xy()

        plotter.camera.SetViewUp(0, 1, 0)  # Y points upward
        plotter.camera.SetPosition(0, 0, 1)  # Looking straight down Z-axis

        plotter.show(auto_close=False)  # Keep plot open
            
            # Then take screenshot
        #plotter.screenshot("temperature_plot.png")

        #plotter.close()
            
            # Close the plotter when done
        #plotter.close()

        # import os
        # plot_dir = os.path.join(os.getcwd(), "plots")
        # os.makedirs(plot_dir, exist_ok=True)
        # plotter.screenshot(os.path.join(plot_dir, "temperature_plot.png"))

        if janela:
            janela.destroy()
        


    def janela(self):
        evaluate_janela = tk.Toplevel(self.root)
        evaluate_janela.title("Avaliar")

        #label da janela
        label_eval = tk.Label(evaluate_janela, text="Avaliação", font=("Helvetica", 14, "bold"))
        label_eval.pack(pady=10)

        entry_frame = tk.Frame(evaluate_janela)
        entry_frame.pack(fill=tk.BOTH, expand=True)
        entry_frame.grid_rowconfigure(0, weight=1)
        entry_frame.grid_rowconfigure(1, weight=1)
        entry_frame.grid_rowconfigure(2, weight=1)
        entry_frame.grid_rowconfigure(3, weight=1)
        entry_frame.grid_columnconfigure(0, weight=1)
        entry_frame.grid_columnconfigure(1, weight=1)

            #Modelo a carregar
        model_entry = tk.Entry(entry_frame, textvariable = self.modelfilename)
        model_entry.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        nodes_label = tk.Label(entry_frame, text="Modelo")
        nodes_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
            
            # Nós a avaliar
        nodes_entry = tk.Entry(entry_frame, textvariable=self.nodefilename)
        nodes_entry.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        nodes_label = tk.Label(entry_frame, text="Nós a avaliar")
        nodes_label.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

            # Temperatura
        tc_entry = tk.Entry(entry_frame, textvariable=self.tc)
        tc_entry.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        tc_label = tk.Label(entry_frame, text="Temperatura Inicial")
        tc_label.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

            #Tempo
        t_entry = tk.Entry(entry_frame, textvariable=self.t)
        t_entry.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        t_label = tk.Label(entry_frame, text="Instante de Tempo")
        t_label.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        btn_fechar = tk.Button(evaluate_janela, text="Avaliar", command= lambda: self.avaliar(evaluate_janela) )
        btn_fechar.pack(pady=5)





