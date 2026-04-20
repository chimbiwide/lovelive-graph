from community import community_louvain
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import defaultdict

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
        if not G.has_edge(name, neighbor) and weight > 0.55:
            G.add_edge(name, neighbor, weight=weight)

partition = community_louvain.best_partition(G)

# a dictionary that automatically creates an empty list when you access a key
clusters = defaultdict(list)

for name, cluster_id in partition.items():
    clusters[cluster_id].append(name)

for cluster_id, members in sorted(clusters.items()):
    print(f"\nCluster {cluster_id}:")
    for m in members:
        print(f" {m} ({graph.nodes[m]["group"]})")
