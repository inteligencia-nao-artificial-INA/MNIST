import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_neural_network():
    architecture = [784, 10, 10]             # neural net architecture
    visible_neurons = [15, 10, 10]           # visible nodes by layer
    neuron_radii = {0: 0.20, 1: 0.6, 2: 0.6} # node size for each layer 
    neuron_border_color = 'black'            # node border color
    connection_color = (0.2, 0.2, 0.2, 0.9)  # connections colors (normalized RGBA)
    connection_thickness = 0.5               # connections width 

    # create directed graph with networkx
    G = nx.DiGraph()
    pos = {}           # dic to store node positions
    layer_dist = 10.0  # dist between layers 
    neuron_dist = 1.5  # dist between nodes in the same layer

    for layer_idx, (num_neurons, num_visible) in enumerate(zip(architecture, visible_neurons)):
        y_offset = -(num_visible - 1) * neuron_dist / 2  # centralize nodes

        for i in range(num_visible):
            node_id = f"L{layer_idx}_N{i}"
            G.add_node(node_id)
            pos[node_id] = (layer_idx * layer_dist, y_offset + i * neuron_dist)

            # connect nodes to previous layer 
            if layer_idx > 0:
                prev_layer = layer_idx - 1
                for j in range(visible_neurons[prev_layer]):
                    prev_node_id = f"L{prev_layer}_N{j}"
                    G.add_edge(prev_node_id, node_id)

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # adjust connection to node border
    for edge in G.edges():
        start_pos = np.array(pos[edge[0]])  # start connection position (center)
        end_pos = np.array(pos[edge[1]])    # end connection position (center)

        # identify each node layer and get node radius
        start_layer = int(edge[0].split("_")[0][1])  # extract layer index
        end_layer = int(edge[1].split("_")[0][1])
        start_radius = neuron_radii[start_layer]     # radius of origin layer
        end_radius = neuron_radii[end_layer]         # radius of end layer

        # calculate vector to node direction
        direction = end_pos - start_pos
        norm = np.linalg.norm(direction)
        if norm != 0:
            direction = direction / norm 

        # adjust connection positions to node border
        start_pos = start_pos + direction * start_radius
        end_pos = end_pos - direction * end_radius

        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                color=connection_color, linewidth=connection_thickness, zorder=1)

    # custom node style for each layer
    for node, (x, y) in pos.items():
        layer_idx = int(node.split("_")[0][1])   # identify node layer
        neuron_radius = neuron_radii[layer_idx]  # set neuron radius

        circle = plt.Circle((x, y), neuron_radius, edgecolor=neuron_border_color, 
                            facecolor='none', linewidth=0.5, zorder=2)
        ax.add_patch(circle)

    ax.set_xlim(-1, layer_dist * (len(architecture) - 1) + 1)
    ax.set_ylim(-(max(visible_neurons) * neuron_dist) / 2, (max(visible_neurons) * neuron_dist) / 2)
    ax.set_aspect('equal')
    ax.axis('off')

    # save network
    plt.savefig("../images/architecture.svg", bbox_inches='tight', dpi=300)
    plt.show()

    # obs: final image was polished with inkscape to generate ../images/architecture.png

draw_neural_network()