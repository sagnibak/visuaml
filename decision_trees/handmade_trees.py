import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

G = nx.DiGraph()

# add the nodes
node_list = [
    (0, {"label": "Raining?"}),
    (1, {"label": "Have time?"}),
    (2, {"label": "Temperature?"}),
    (
        3,
        {
            "label": "Hiking",
            "shape": "box",
            "style": "filled",
            "fillcolor": "green",
        },
    ),
    (
        4,
        {
            "label": "No hiking",
            "shape": "box",
            "style": "filled",
            "fillcolor": "green",
        },
    ),
    (
        5,
        {
            "label": "No hiking",
            "shape": "box",
            "style": "filled",
            "fillcolor": "green",
        },
    ),
    (
        6,
        {
            "label": "No hiking",
            "shape": "box",
            "style": "filled",
            "fillcolor": "green",
        },
    ),
    (
        7,
        {
            "label": "No hiking",
            "shape": "box",
            "style": "filled",
            "fillcolor": "green",
        },
    ),
]
G.add_nodes_from(node_list)

# add the edges
edge_list = [
    (0, 1, {"label": "No"}),
    (0, 4, {"label": "Yes"}),
    (1, 2, {"label": "Yes"}),
    (1, 5, {"label": "No"}),
    (2, 6, {"label": "<45°F"}),
    (2, 7, {"label": ">90°F"}),
    (2, 3, {"label": "45-90°F"}),
]
G.add_edges_from(edge_list)

# save the decision tree
nx.nx_pydot.to_pydot(G).write_png("intro_tree.png")
