from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
#Load environment variable
load_dotenv()

# Step 1: Load and split the document
if not os.path.exists("/Users/ravinderkumar/Work/upskill/AI/AIAgent/AI_LangChain/AI_LangChain/106_LangChain_Runnables/sample_doc.txt"):
    raise FileNotFoundError("Sample_doc.txt not found")
loader = TextLoader("/Users/ravinderkumar/Work/upskill/AI/AIAgent/AI_LangChain/AI_LangChain/106_LangChain_Runnables/sample_doc.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs = splitter.split_documents(documents)

# Step 2: Embed and store with FAISS
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs,embedding)

# Step 3: Create Retrieval QA chain with chat open AI
llm = ChatOpenAI()
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever)

# Step 4: As Question
query = "What is the main purpuse of the document?"
result = qa_chain.invoke(query)

print("Q:",query)
print("A:",result['result'])

