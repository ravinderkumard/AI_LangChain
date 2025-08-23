from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
documents = [
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
    )
]   

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="langchain_collection")

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query = "What is retriever?"
results = retriever.invoke(query)

print("Query Results:", results)
for i, doc in enumerate(results):
    print(f"Document {i+1}:")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print()
