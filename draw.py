import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from graph import CharacterGraph
from embeddings import load_embeddings

G = nx.Graph()

characters = load_embeddings()
graph = CharacterGraph()

graph.build_graph(characters)

for name, data in graph.nodes.items():
    G.add_node(name, group=data["group"])

for name in graph.edges:
    for neighbor, weight in graph.edges[name]:
        if not G.has_edge(name, neighbor):
            G.add_edge(name, neighbor, weight=weight)

group_colors = {
    "Muse": "#FF6B6B", #red
    "Aqours": "#6BB5FF", #blue
    "Nijigasaki High School Idol Club": "#FFD93D", #yellow
    "Liella!": "#6BCB77", #green 
    "Hasunosora Girls' High School Idol Club": "#C77DFF" #purple
}

group_centers = {
    "Muse": np.array([-2, 2]),
    "Aqours": np.array([2, 2]),
    "Nijigasaki High School Idol Club": np.array([0, 0]),
    "Liella!": np.array([-2, -2]),
    "Hasunosora Girls' High School Idol Club": np.array([2, -2])
}

init_pos = {}

for name, data in G.nodes(data=True):
    center = group_centers.get(data["group"], np.array([0,0]))
    init_pos[name] = center + np.random.randn(2) * 0.3

colors = [group_colors.get(G.nodes[n]["group"], "gray") for n in G.nodes]
pos = nx.spring_layout(G, seed=42, k=3, pos=init_pos, fixed=None)
filtered = [(u, v) for u,v,d in G.edges(data=True) if d["weight"] > 0.6]

plt.figure(figsize=(15,13))
nx.draw_networkx(G, pos=pos, edgelist=filtered, with_labels=True, font_size=5, node_color=colors, node_size=1500)
plt.savefig("graph.png", dpi=200, bbox_inches="tight")
