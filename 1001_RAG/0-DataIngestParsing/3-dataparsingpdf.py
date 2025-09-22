from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader,
)

try:
    pypdf_loader = PyPDFLoader("data/pdf/llm.pdf")
    pypdf_docs = pypdf_loader.load()
    print(pypdf_docs)
    print(f"Number of pages in PDF: {len(pypdf_docs)}")
    print(f"First page content: {pypdf_docs[0].page_content[:500]}")  # Print first 500 characters of the first page
    print(f"Metadata of first page: {pypdf_docs[0].metadata}")

except Exception as e:
    print(f"PyPDFLoader Error: {e}")


try:
    pymupdf_loader = PyMuPDFLoader("data/pdf/llm.pdf")
    pymupdf_docs = pymupdf_loader.load()
    print(pymupdf_docs)
    print(f"Number of pages in PDF: {len(pymupdf_docs)}")
    print(f"First page content: {pymupdf_docs[0].page_content[:500]}")  # Print first 500 characters of the first page
    print(f"Metadata of first page: {pymupdf_docs[0].metadata}")
except Exception as e:
    print(f"PyMuPDFLoader Error: {e}")
    
