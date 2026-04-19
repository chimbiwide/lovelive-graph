from embeddings import load_embeddings
from graph import CharacterGraph

graph = CharacterGraph()
characters = load_embeddings()

graph.build_graph(characters)

while(True):
    name = input("Enter character name (Muse&Aquors): ")
    if name not in graph.nodes:
        print("Invalid Name")
        continue

    result = graph.top_similar(name)
    print(f"The top 5 character that matches with {name}:")
    for i in range(5):
        char = result[i]
        print(f"{i+1}. {char[0]} ({graph.nodes[char[0]]["group"]}) : weight: {char[1]:.3f}")
    print("-"*20)
