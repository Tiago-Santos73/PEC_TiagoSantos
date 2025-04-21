import meshio
import torch
torch.set_default_dtype(torch.float32)

class Read:
    def __init__(self, file):
        self.file = file

    def read(self):
        print("Reading the nodes file...")
        # read the inp file
        mesh = meshio.read(self.file)
        #
        # get the nodes coordinates
        nodes = mesh.points
        # convert the nodes coordinates into a tensor
        Nodes = torch.from_numpy(nodes).float()  # Explicitly convert to float32
        #print(f"Nodes tensor dtype: {Nodes.dtype}")
        #
        # get the node sets
        node_sets = mesh.point_sets
        #
        # get the node set's coordinates
        node_set_coords = {}
        for set_name, node_ids in node_sets.items():
            # get the index
            node_indices = [nid for nid in node_ids]
            # Extract coordinates
            node_coords = Nodes.clone().detach()[node_indices]
            # extract the set coordinates
            node_set_coords[set_name] = node_coords
        #
        print("ByNodes read!!!")
        #print(node_set_coords)
        return node_set_coords

   


if __name__ == '__main__':
    file = "file.inp"
    R = Read(file)
    R.read()