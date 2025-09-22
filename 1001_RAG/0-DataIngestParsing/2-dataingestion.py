import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter

print("Setup completed")

doc = Document(
    page_content = "This is a main text content that will be embedded and searched.",
    metadata={
        "source":"example.txt",
        "page":1,
        "author":"RKD",
        "date_created":"2025",
        "custom_field":"any_value"
    }
)
print("Document Structure")
print(f"Content: {doc.page_content}")
print(f"Metadata : {doc.metadata}")

import os
os.makedirs("data/text_files",exist_ok=True)

sample_texts={
    "data/text_files/python_intro.txt":"""What is Python?

Python is a high-level, interpreted programming language that is widely used for many types of software development. It was created by Guido van Rossum and first released in 1991.

Key Features of Python

Easy to Learn and Read

Python’s syntax is simple and resembles human language, which makes it beginner-friendly.

Example:

print("Hello, World!")


Interpreted Language

Python code is executed line by line by the Python interpreter.

No need to compile code before running it.

Dynamically Typed

You don’t need to declare variable types explicitly.

x = 10      # integer
x = "Hello" # string


Versatile and Multi-Paradigm

Supports object-oriented, procedural, and functional programming.

Large Standard Library and Ecosystem

Comes with built-in modules for tasks like file handling, networking, and math.

Thousands of third-party packages (like numpy, pandas, langchain) extend its functionality.

Cross-Platform

Python works on Windows, macOS, Linux, and even mobile platforms.

Where Python is Used

Web Development – frameworks like Django, Flask

Data Science & Machine Learning – pandas, scikit-learn, tensorflow

Automation / Scripting – automate repetitive tasks

Artificial Intelligence & NLP – langchain, OpenAI, spaCy

Game Development – pygame

Networking & Cybersecurity – writing scripts for network tools

Why Python is Popular

Easy to learn for beginners.

Strong community support.

Flexible and powerful for professionals.

Integrates well with other languages and tools.

In short, Python is a general-purpose programming language that is easy to read, highly versatile, and widely used across industries."""
}

for filepath,content in sample_texts.items():
    with open(filepath,'w',encoding="utf-8") as f:
        f.write(content)

print("Sample file has Create")

from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/text_files/python_intro.txt",encoding="utf-8")
loader

documents = loader.load()
print(type(documents))
print(documents)

from langchain_community.document_loaders import DirectoryLoader
dir_loader = DirectoryLoader(
     "data/text_files",
     glob="**/*.txt",
     loader_cls = TextLoader,
     loader_kwargs={'encoding':'utf-8'},
     show_progress=True
)

documents = dir_loader.load()
print(f"Loaded {len(documents)} documents")


text = documents[0].page_content

char_splitter = CharacterTextSplitter(
    chunk_size=300,  # Number of characters per chunk
    chunk_overlap=50,  # Number of characters to overlap between chunks 
    length_function=len
)

char_chunks = char_splitter.split_text(text)

print(f"Number of character chunks: {len(char_chunks)}")
print(f"First chunk: {char_chunks[0][:100]}")



recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Number of characters per chunk
    chunk_overlap=50,  # Number of characters to overlap between chunks
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

recursive_chunks = recursive_splitter.split_text(text)

print(f"Number of recursive chunks: {len(recursive_chunks)}")
print(f"First recursive chunk: {recursive_chunks[0][:100]}")