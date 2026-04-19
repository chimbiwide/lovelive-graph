import json
import embeddings

class CharacterGraph:
    def __init__(self):
        self.nodes = {} #character data
        self.edges = {} #(neighbor, weight)

    def add_character(self, name, data):
        self.nodes[name] = data
        self.edges[name] = []

    def add_edge(self, char1, char2, weight):
        self.edges[char1].append((char2, weight))
        self.edges[char2].append((char1, weight))

    def neighbors(self, name):
        return self.edges[name]

    def top_similar(self, name, n=5):
        sorted_neighbors = sorted(self.edges[name], key=lambda x: x[1], reverse=True)
        return sorted_neighbors[:n]

    def build_graph(self, characters):
        for char in characters:
            self.add_character(char["name"], char)
        for i, char1 in enumerate(characters):
            for j, char2 in enumerate(characters):
                if j <= i:
                    continue
                self.add_edge(char1["name"], char2["name"], embeddings.cosine_similarity(char1["embedding"], char2["embedding"]))

if __name__=="__main__":
    # create character graph
    graph = CharacterGraph()

    #load and create character nodes
    characters = embeddings.load_embeddings()

    graph.build_graph(characters)

    print(graph.top_similar("Ruby Kurosawa"))

