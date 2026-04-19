from sentence_transformers import SentenceTransformer
import json
import numpy as np

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"

def load_characters(path="./data/characters.json"):
    with open(path, 'r') as f:
        return json.load(f)

def load_embeddings(path="./data/embedding.json"):
    with open(path, 'r') as f:
        return json.load(f)

# model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def generate_embeddings(characters):
    model = SentenceTransformer(MODEL_NAME)
    for character in characters:
        character["embedding"] = model.encode(character["personality"]).tolist()
    return characters

def save_embbeddings(characters, path="./data/embedding.json"):
    with open(path, 'w') as f:
        json.dump(characters, f)
    print("saved")

def compute_similarity(characters):
    results = []
    for i, char1 in enumerate(characters):
        for j, char2 in enumerate(characters):
            if j <= i:
                continue
            score = cosine_similarity(char1["embedding"], char2["embedding"])
            results.append((char1["name"], char2["name"], score))
    return results



if __name__=="__main__":
    characters = load_characters()
    characters = generate_embeddings(characters)
    save_embbeddings(characters)
    
    similarities = compute_similarity(characters)

    with open("./results.txt", 'w') as f:
        for char1, char2, score in sorted(similarities, key=lambda x: x[2], reverse=True):
            f.write(f"{char1} <--> {char2} -- {score}\n")
