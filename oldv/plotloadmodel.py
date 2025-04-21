import pyvista as pv
import numpy as np
import torch
torch.set_default_dtype(torch.float32)

class Plotmodel:
    def __init__(self, inference):
        self.inference = inference

    def plot(self):
        # Extract coordinates (x,y) and temperatures
        pointsxy = self.inference.domain[:, :2].detach().cpu().numpy()  # Shape: [N, 2]
        points_xyz = np.column_stack((pointsxy, np.zeros(len(pointsxy))))
        temps = self.inference.results.detach().cpu().numpy().flatten()  # Shape: [N]

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