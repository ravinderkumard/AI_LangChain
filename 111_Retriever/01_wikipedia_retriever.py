from langchain_community.retrievers import WikipediaRetriever


retriever = WikipediaRetriever(
    top_k_results=2,  # Number of top results to retrieve
    lang="en"  # Language of the Wikipedia articles
)

query = "What is LangChain?"
results = retriever.invoke(query)
print("Query Results:", results)