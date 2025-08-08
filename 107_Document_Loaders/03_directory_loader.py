from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from datetime import datetime
import time
loader = DirectoryLoader(
    "107_Document_Loaders/data",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)
start_time = time.time()
#documents =loader.load()
documents = loader.lazy_load()  # Use lazy_load for potentially large directories
end_time = time.time()
print(f"Time taken to load documents: {end_time - start_time:.2f} seconds")
for doc in documents:
    print(f"Document ID: {doc.metadata.get('source', 'unknown')}")
    print(f"Document ID: {doc.metadata}")
    # print(f"Content: {doc.page_content[:100]}...")  # Print first 100 characters of content
    # print("-" * 40)  # Separator for readability

# print(f"Total documents loaded: {len(documents)}")
