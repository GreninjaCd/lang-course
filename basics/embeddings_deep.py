from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

def basic_embedding():

    # single text
    text = "What is Machine Learning?"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")


def batch_embeddings():
    # batch of texts
    texts = [
        "What is Machine Learning?",
        "What is Deep Learning?",
        "What is Reinforcement Learning?",
    ]
    batch_embeddings = embeddings.embed_documents(texts)
    for i, emb in enumerate(batch_embeddings):
        print(f"Text {i + 1} - Vector dimensions: {len(emb)}")
        print(f"Text {i + 1} - First 5 values: {emb[:5]}")
        print(f"Text {i + 1} - Vector norm: {np.linalg.norm(emb):.4f}\n")


def similarity_search():
    # Documents 
    docs = [
        "Pythin is a programming language.",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist?"

    #embed documents and query
    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    ranked_docs = sorted(zip(docs, similarities), key = lambda x : x[1], reverse = True)

    print(f"Ouery: {query}\n")
    print("Ranked by similarity:")
    for doc, score in ranked_docs:
        print(f" {score:.4f}: {doc}")

if __name__ == "__main__":
    similarity_search()