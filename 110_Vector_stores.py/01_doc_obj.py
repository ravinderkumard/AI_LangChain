from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()
doc1 = Document(
    page_content="This is a sample document for testing.",
    metadata={"source": "sample_doc_1", "author": "John Doe"} 
)
doc2 = Document(
    page_content="This is another sample document for testing.",
    metadata={"source": "sample_doc_2", "author": "Jane Doe"}
)
doc3 = Document(
    page_content="This is a third sample document for testing.",
    metadata={"source": "sample_doc_3", "author": "Alice Smith"}
)
# Create a list of documents
documents = [doc1, doc2, doc3]

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="vector_store"  # Directory to persist the vector store
)
# Persist the vector store
vector_store.persist()
# Load the vector store from the persisted directory
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="vector_store"
)

metadata = vector_store.get(include=['embeddings','documents','metadatas'])  # Get metadata of all documents
print(metadata)
# # Example query to find similar documents
query = "What is the sample document about?"
results = vector_store.similarity_search(query, k=2)  # Get top 2 similar documents

print("Query Results:",results)