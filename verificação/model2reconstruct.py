import torch
torch.set_default_dtype(torch.float32)
import matplotlib.pyplot as plt
import os
import pandas as pd

class Inference:
    def __init__(self,parameters,ti):
        self.parameters = parameters
        # set the model file name
        # folder = "./res_numerical/"
        self.file = self.parameters["model"]
        self.name, ext = os.path.splitext(self.file)
        self.ti=ti
        #
        self.read_numerical()


    def create_domain_from_file(self,ti=0.0,tinicial=0.0):
        # search for the file containing the desired data
        idx = torch.where(self.cvs_data_time[:,1] == ti)[0]
        # read the file
        self.domain = pd.read_csv("./Numerical/" + self.csv_files[idx])
        # convert the data into tensor
        self.domain = torch.from_numpy(self.domain.values)
        # get the true output
        self.TRUE_OUTPUT = self.domain[:,[-1]].clone().detach()
        self.TRUE_OUTPUT = self.TRUE_OUTPUT.type(torch.float32)
        # get the domain for evaluation (with T0)
        self.domain = self.domain.clone().detach()
        self.domain[:,-1] = tinicial
        self.domain = self.domain.type(torch.float32)

    def read_numerical(self):
        # folder = "./Numerical/"
        folder = "./" + self.parameters['numerical'] + "/"
        self.csv_files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])
        # create a database of the time and file index
        self.cvs_data_time = torch.ones(size=(len(self.csv_files), 2), dtype=torch.float32)
        # iterate over all the files
        idx = 0
        for i in self.csv_files:
            # read the first line of the file
            first_line = pd.read_csv(folder + i, nrows=1)
            # save the data
            self.cvs_data_time[idx,0] = idx
            self.cvs_data_time[idx,1] = first_line['Time'].iloc[0]
            #
            idx += 1        

    def load_model(self):
        # Load model
        print("Loading model...")
        self.model = torch.load(self.file, map_location='cpu',weights_only=False)
        #
        # Set to evaluation mode
        self.model.eval()
        print("Model loaded.\n")        
    
    @torch.no_grad()
    def inference(self):
        # include the time into the domain
        self.OUTPUT = self.model(self.domain)
        #
        div = 100.0*torch.absolute((self.OUTPUT - self.TRUE_OUTPUT))/self.TRUE_OUTPUT
        # print(div)
        print(torch.mean(div))
    
    def plot_result(self):
        x = self.domain[:,0]
        y = self.domain[:,1]
        T = self.OUTPUT[:,0]
        #
        cmap = "jet"
        #levels = torch.linspace(0.0, 1.0, 21)
        levels = torch.linspace(0.25, 0.55, 26)
        #
        contour = plt.tricontourf(x, y, T, levels=levels, cmap=cmap)
        #
        ticks = levels
        plt.colorbar(contour, label="Temperature", ticks=ticks)
        #
        plt.clim(0.25, 0.55)
        #
        plt.scatter(x, y, color="black", s=5, alpha=0.7)
        #
        plt.title("PINN result")
        #
        plt.savefig(f'sur_{self.name}_{self.ti}.png')
        #
        plt.close()

    def plot_error(self):
        x = self.domain[:,0]
        y = self.domain[:,1]
        T = torch.absolute((self.OUTPUT - self.TRUE_OUTPUT))/self.TRUE_OUTPUT
        T = T[:,0]        
        #
        cmap = "jet"
        #levels = torch.linspace(0, 1.0, 21)
        levels = torch.linspace(0.0, 0.5, 26)
        #
        contour = plt.tricontourf(x, y, T, levels=levels, cmap=cmap)
        #
        ticks = levels
        plt.colorbar(contour, label="Error", ticks=ticks)
        #
        #plt.clim(0, 1.0)
        plt.clim(0, 0.5)
        #
        plt.scatter(x, y, color="black", s=5, alpha=0.7)
        #
        div = 100.0*torch.absolute((self.OUTPUT - self.TRUE_OUTPUT))/self.TRUE_OUTPUT
        plt.title("PINN result - mean error: {:.2f}%".format(torch.mean(div).item()))
        #
        plt.savefig(f'sur_error_{self.name}_{self.ti}.png')
        #
        plt.close()

    def plot_numerical(self):
        x = self.domain[:,0]
        y = self.domain[:,1]
        T = self.TRUE_OUTPUT[:,0]
        #
        cmap = "jet"
        #levels = torch.linspace(0.0, 1.0, 21)
        levels = torch.linspace(0.25, 0.55, 26)
        #
        contour = plt.tricontourf(x, y, T, levels=levels, cmap=cmap)
        #
        ticks = levels
        plt.colorbar(contour, label="Temperature", ticks=ticks)
        #
        plt.clim(0.25, 0.55)
        #
        plt.scatter(x, y, color="black", s=5, alpha=0.7)
        #
        plt.title("Numerical result")
        #
        plt.savefig(f'sur_true_{self.ti}.png')
        #
        plt.close()



d = {}
d['folder'] = "res_numerical"
d['model'] = "Final_model.pth"
d['numerical'] = "Numerical"
d['domain samples'] = 21
d['x domain'] = [0.0, 1.0]
d['y domain'] = [0.0, 1.0]
d['t domain'] = [0.0, 1.0]

ti = 0.45
Y = Inference(parameters=d,ti=ti)
temp_inicial = 0.5
Y.create_domain_from_file(ti=ti,tinicial=temp_inicial)
Y.load_model()
Y.inference()
Y.plot_result()
Y.plot_error()
Y.plot_numerical()


