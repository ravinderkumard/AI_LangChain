from langchain.text_splitter import  CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


text = """LangChain is a framework for developing applications powered by language models. 

It provides a standard interface for working with language models, making it easier to build applications that 

can understand and generate human language. LangChain supports various tasks such as text generation, 
summarization, translation, and more."""
loader = PyPDFLoader("109_Langchain_Text_splitter/data/llm.pdf")

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # Number of characters per chunk
    chunk_overlap=0  # Number of characters to overlap between chunks
)

result = text_splitter.split_text(text)
print(f"Number of chunks: {len(result)}")
print(f"Number of chunks: {result}")