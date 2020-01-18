import matplotlib.pyplot as plt
import numpy as np
import pydot

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
intro_tree = pydot.Dot(graph_type="digraph", ordering="out")
for node_name, attrs in node_list:
    intro_tree.add_node(pydot.Node(node_name, **attrs))

# add the edges
edge_list = [
    (0, 4, {"label": "Yes"}),
    (0, 1, {"label": "No"}),
    (1, 2, {"label": "Yes"}),
    (1, 5, {"label": "No"}),
    (2, 6, {"label": "<45°F"}),
    (2, 3, {"label": "45-90°F"}),
    (2, 7, {"label": ">90°F"}),
]
for u, v, attrs in edge_list:
    intro_tree.add_edge(pydot.Edge(u, v, **attrs))

# save the decision tree
intro_tree.write_png("intro_tree.png")
