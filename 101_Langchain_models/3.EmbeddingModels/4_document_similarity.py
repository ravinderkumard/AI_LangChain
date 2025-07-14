from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    
)

documents = [
    "Virat Kohli is an Indian international cricketer and the former captain of the Indian national cricket team. He is a right-handed batsman and an occasional medium-fast bowler. He currently represents Royal Challengers Bengaluru in the IPL and Delhi in domestic cricket.",
    "Shubman Gill is an Indian international cricketer who plays for the India national team in all formats. Gill captains the Test team, vice-captains the ODI side and has captained the T20I team. He is nicknamed the Prince of Indian cricket",
    "Anil Kumble is a former Indian cricketer, captain, coach and commentator who played Test and One Day International cricket for his national team over an international career of 18 years.",
    "Sachin Ramesh Tendulkar is an Indian former international cricketer who captained the Indian national team.",
    "Yuvraj Singh is a former Indian international cricketer who played in all formats of the game. An all-rounder who batted left-handed in the middle order and bowled slow left-arm orthodox, he has won 7",
    "Paris is the capital and most populous city of France. It is located in the north-central part of the country, along the Seine River. Paris is known for its art, fashion, gastronomy, and culture.",
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