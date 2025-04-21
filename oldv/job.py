import torch
torch.set_default_dtype(torch.float32)
from tqdm import tqdm
import meshio
import read_nodes
import multiprocessing

class Plate(torch.nn.Module):
    def __init__(self, params):
        super(Plate,self).__init__()
        #
        # get the parsed parameters
        self.parameters = params
        #
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        #
        # Activation function
        self.activation = torch.nn.Tanh
        #
        # Loss function criteria
        self.criteria = torch.nn.MSELoss(reduction='mean')
        #
        # Initialize the neural network
        print("Creating the NN...")
        L = []
        #
        # First hidden layer
        L.append(torch.nn.Linear(self.parameters['inputs'],self.parameters['layers'][0]))
        L.append(self.activation())
        #
        # Internal Hidden layers
        for i in range(len(self.parameters['layers'])-1):
            L.append(torch.nn.Linear(self.parameters['layers'][i], self.parameters['layers'][i+1]))
            L.append(self.activation())
        #
        # Output layer
        L.append(torch.nn.Linear(self.parameters['layers'][-1], self.parameters['outputs']))
        # deploy the neural network
        self.model = torch.nn.Sequential(*L).to(self.device)
        #
        # define the optimizer for the model parameters
        self.adam = torch.optim.Adam(params=self.model.parameters(), lr=self.parameters['lr'])
        #
        print("NN Created")

    def read_mesh(self):
        NODES = read_nodes.Read(file=self.parameters['nodes file'])
        sets = NODES.read()
        # print(sets)
        return sets
    
    def testesimples(self):
        #print(self.teste1)

        for i in range(100_000_000):
            print(i)





if __name__ == '__main__':
    #  create the problem
    d = {}
    d['inputs'] = 3
    d['layers'] = [10]*4
    d['outputs'] = 1
    d['epochs'] = 4000
    d['lr'] = 1e-04
    d['domain samples'] = int(500)
    d['IC samples'] = int(250)
    d['BC samples'] = int(250)
    d['nodes file'] = "file.inp"

    NN = Plate(params=d)
    NN.read_mesh()