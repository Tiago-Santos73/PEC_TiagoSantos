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
        #print("Loading model...")
        self.model = torch.load(self.file, map_location='cpu',weights_only=False)
        #
        # Set to evaluation mode
        self.model.eval()
        #print("Model loaded.\n")        
    
    @torch.no_grad()
    def inference(self):
        # include the time into the domain
        self.OUTPUT = self.model(self.domain)


    def calculate_error(self):
        div = 100.0 * torch.absolute((self.OUTPUT - self.TRUE_OUTPUT)) / self.TRUE_OUTPUT
        #print(torch.mean(div))
        return torch.mean(div).item()

    def plot_error_over_time(self, errors, ti_list):
    
        plt.figure()

        plt.plot(ti_list, errors, marker='o')

        plt.xlabel("Time")

        plt.ylabel("Error (%)")

        plt.title(" Error over Time")

    
        plt.grid(True, which='major', linewidth=1.0) 
        plt.grid(True, which='minor', linestyle='--', alpha=0.5) 
        plt.minorticks_on()
        plt.ylim(bottom=0)

        plt.savefig(f'error_over_time_{self.name}.png')

        plt.close()


       
    

if __name__ == '__main__':

    d = {}
    d['folder'] = "res_numerical"
    d['model'] = "Final(1)_model.pth"
    d['numerical'] = "Numerical"
    d['domain samples'] = 21
    d['x domain'] = [0.0, 1.0]
    d['y domain'] = [0.0, 1.0]
    d['t domain'] = [0.0, 1.0]
    temp_inicial = 0.5


    beginning = 0.0
    end = 1.0
    step = 0.01
    ti_list = []
    value = beginning
    while value < end+step:
        ti_list.append(round(value, 2))
        value = value + step

    errors = []

    for ti in ti_list:
        Y = Inference(parameters=d,ti=ti)
        Y.create_domain_from_file(ti=ti,tinicial=temp_inicial)
        Y.load_model()
        Y.inference()
        error = Y.calculate_error()
        errors.append(error)
    
    Y.plot_error_over_time(errors, ti_list)




