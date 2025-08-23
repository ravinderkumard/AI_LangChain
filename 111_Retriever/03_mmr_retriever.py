from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_doc_1", "author": "Alice"}
    ),
    Document(
        page_content="It provides tools to work with LLMs, manage prompts, and handle data.",
        metadata={"source": "langchain_doc_2", "author": "Bob"}
    ),
    Document(
        page_content="LangChain supports various vector stores for efficient retrieval.",
        metadata={"source": "langchain_doc_3", "author": "Charlie"}
    ),
    Document(
        page_content="MMR stands for Maximal Marginal Relevance, a technique used to improve the diversity of search results.",
        metadata={"source": "mmr_doc_1", "author": "David"}
    ),
    Document(
        page_content="MMR helps in selecting documents that are not only relevant to the query but also diverse from each other.",
        metadata={"source": "mmr_doc_2", "author": "Eve"}
    )
]

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)

retreiver = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.5})
query = "What is MMR?"
results = retreiver.invoke(query)
#print("Query Results:", results)
for i, doc in enumerate(results):
    print(f"Document {i+1}:")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print()