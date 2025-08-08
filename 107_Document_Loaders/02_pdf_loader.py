from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("107_Document_Loaders/data/llm.pdf")
documents = loader.load()
for doc in documents:
    print(f"Document ID: {doc.metadata.get('source', 'unknown')}")
    print(f"Document ID: {doc.metadata}")
    print(f"Content: {doc.page_content[:100]}...")  # Print first 100 characters of content
    print("-" * 40)  # Separator for readability
# This code loads a PDF file using LangChain's PyPDFLoader and prints the document ID and the first 100 characters of its content.
# The document ID is derived from the metadata, and the content is truncated for display purposes.
# Ensure you have the necessary packages installed:
# pip install langchain-community pypdf
# Make sure to adjust the file path and encoding as needed for your specific use case.
# Note: The PyPDFLoader is part of the langchain_community package, which may need to be installed separately.
# If you encounter any issues, ensure that the file exists and is accessible at the specified path.
# This code snippet is a simple example of how to use LangChain's PyPDFLoader to load PDF documents.
# It demonstrates how to read a PDF file and print its content in a structured way.         
# This is a basic example and can be extended to include more complex document processing or analysis tasks.
# The PyPDFLoader is useful for loading PDF files into LangChain's document format, which can then be used for various NLP tasks.
# The documents loaded can be further processed, analyzed, or used in various LangChain workflows.
# This code is a straightforward example of using LangChain's PyPDFLoader to read and display PDF documents.
# It serves as a foundation for building more complex document processing applications using LangChain.         
# The PyPDFLoader is a convenient way to handle PDF files in LangChain, allowing for easy integration into larger NLP pipelines.
# This code snippet is a simple demonstration of how to use LangChain's PyPDFLoader to load and display PDF documents.
# It can be used as a starting point for more advanced document processing tasks in LangChain.          