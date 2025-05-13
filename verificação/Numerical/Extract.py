from odbAccess import *
from abaqusConstants import *


# Open the ODB file (output database file)
odb = openOdb(path='./Job-1.odb')  

# Define the step and output variable you want to extract (e.g., TEMP)
step = odb.steps['Step-1']  

# Get the surface or node set for the upper surface
nodes_set = odb.rootAssembly.instances['PART-1-1'].nodeSets['NODES_TODOS']  # Use the name of your surface node set

# Loop over each time frame (time increment) in the step
i = 0
for frame in step.frames:
    time_value = frame.frameValue

    filename = "{:04d}.csv".format(i)

    # Open a file to write the results
    output_file = open(filename, 'w')

    # Write header in the output file
    output_file.write('X,Y,Time,Temperature\n')
    
    # Get the temperature field output at this frame
    temp_field = frame.fieldOutputs['NT11'] 

    # Extract temperature values for nodes on the upper surface
    for node in nodes_set.nodes:  # [0] is the instance number, usually 0
        node_label = node.label
        coords = node.coordinates
        x_coord = coords[0]  
        y_coord = coords[1]  
        
        # Extract the temperature value at the node
        temperature = temp_field.getSubset(region=nodes_set).getSubset(region=node).values[0].data

        # Write the time increment, node label, x, y coordinates, temperature, and ambient temperature to the output file
        output_file.write('{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(x_coord,y_coord,time_value,temperature))
    #
    # close the file
    output_file.close()
    # increase the file number
    i += 1

# Close the ODB
odb.close()
