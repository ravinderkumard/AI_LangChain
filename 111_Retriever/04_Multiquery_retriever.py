from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()

all_docs = [
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
    ),
    Document(
        page_content="Multi-query retrieval involves generating multiple queries from a single input query to enhance retrieval performance.",
        metadata={"source": "multiquery_doc_1", "author": "Frank"}
    ),
    Document(
        page_content="This technique can capture different aspects of the input query, leading to more comprehensive search results.",
        metadata={"source": "multiquery_doc_2", "author": "Grace"}
    )   
]  

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(all_docs, embeddings)
multi_query_retriever = MultiQueryRetriever.from_llm(
    llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
    retriever=vector_store.as_retriever(search_kwargs={"k": 5})
)
query = "What is multi-query retrieval?"
results = multi_query_retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"Document {i+1}:")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print()