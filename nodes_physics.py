import tkinter as tk
import read_nodes
import meshio
import torch
import os
torch.set_default_dtype(torch.float32)



class nodes_pe:
    def __init__(self, filevar, output):
        self.filevar = filevar
        self.output = output

    def runf(self):
        # Nome do ficheiro com e sem extensão
        filename = self.filevar.get()
        #if not filename.endswith(".inp"):
        #    filename += ".inp"

        # Apagar o anterior
        self.output.delete("1.0", tk.END)

        # Ler o inp
        R = read_nodes.Read(filename)
        node_set_coords = R.read()
        mesh = meshio.read(filename)
        nodes = mesh.points

        # Número de nós
        nodes_str = f"Número de nós: {len(nodes)}\n"
        self.output.insert(tk.END, nodes_str)

        # Grups fisicos
        for key, value in mesh.cell_sets.items():
            entity_types = [mesh.cells[i].type for i in range(len(mesh.cells)) if value[i].size > 0]
            if key in node_set_coords:    
                self.output.insert(tk.END, f"Grupo Físico: {key}, {entity_types}\n")

        #Guardade ficheiro
        
        base_name, ext = os.path.splitext(filename)

        
        node_set_coords_file = f"{base_name}_node_set_coords.txt"
        
        
        with open(node_set_coords_file, "w") as f:
            for set_name, coords in node_set_coords.items():
                f.write(f"Node Set: {set_name}\n")
                for coord in coords:
                    coord_str = " ".join(map(str, coord.tolist()))
                    f.write(f"{coord_str}\n")
                f.write("\n")

        self.output.insert(tk.END, f"Coordenadas dos nós gravadas em {node_set_coords_file}\n")

if __name__ == '__main__':

    root = tk.Tk()
    textbox=tk.Text(root)
    filename = tk.StringVar()
    filename.set("file.inp")
    npe = nodes_pe(filename,textbox)
    npe.runf()