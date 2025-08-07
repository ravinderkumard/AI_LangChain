from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
# Initialize the OpenAI chat model
llm = ChatOpenAI(model="gpt-4", temperature=0.0)

prompt = PromptTemplate(
    template="Write a summary of the following text:\n\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()
# Create a chain that processes the text through the prompt, LLM, and parser

loader = TextLoader("107_Document_Loaders/data/atlanta.txt", encoding="utf-8")

documents = loader.load()
for doc in documents:
    print(f"Document ID: {doc.metadata.get('source', 'unknown')}")
    print(f"Document ID: {doc.metadata}")
    print(f"Content: {doc.page_content[:100]}...")  # Print first 100 characters of content
    print("-" * 40)  # Separator for readability
# This code loads a text file using LangChain's TextLoader and prints the document ID and the first 100 characters of its content.
# The document ID is derived from the metadata, and the content is truncated for display purposes.
# Ensure you have the necessary packages installed:
# pip install langchain-community
# Make sure to adjust the file path and encoding as needed for your specific use case.
# Note: The TextLoader is part of the langchain_community package, which may need to be installed separately.
# If you encounter any issues, ensure that the file exists and is accessible at the specified path.
# This code snippet is a simple example of how to use LangChain's TextLoader to load text documents.
# It demonstrates how to read a text file and print its content in a structured way.
# This is a basic example and can be extended to include more complex document processing or analysis tasks.
# The TextLoader is useful for loading text files into LangChain's document format, which can then be used for various NLP tasks.
# The documents loaded can be further processed, analyzed, or used in various LangChain workflows.
# This code is a straightforward example of using LangChain's TextLoader to read and display text documents.
# It serves as a foundation for building more complex document processing applications using LangChain.
# The TextLoader is a convenient way to handle text files in LangChain, allowing for easy integration into larger NLP pipelines.
# This code snippet is a simple demonstration of how to use LangChain's TextLoader to load and display text documents.
# It can be used as a starting point for more advanced document processing tasks in LangChain.
# The TextLoader is a powerful tool for reading text files and converting them into LangChain's document format.        

chain = prompt | llm | parser
result = chain.invoke({'text': documents[0].page_content})

print(result)
