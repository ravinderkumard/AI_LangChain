from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=300
)

documents = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome.",
    "The capital of Spain is Madrid.",
    "The capital of Japan is Tokyo."
]

query = "What is the capital of France?"
query_embedding = embeddings.embed_query(query)
document_embeddings = embeddings.embed_documents(documents)
# Calculate cosine similarity
similarities = cosine_similarity(
    [query_embedding],
    document_embeddings
)[0]

index,score = sorted(list(enumerate(similarities)), key=lambda x: x[1])[-1]
print(f"Query: {query}")
print(f"Most similar document: {documents[index]}")
print(f"Cosine similarity score: {score:.4f}")