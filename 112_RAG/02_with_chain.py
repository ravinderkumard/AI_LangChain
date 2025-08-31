from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

def format_docs(retriever_docs):
    context_text = "\n\n".join([doc.page_content for doc in retriever_docs])
    return context_text

## Step 1a. Indexing YouTube Video
#video_id = "5qap5aO4i9A"  # Example video ID (Lo-fi hip hop radio - beats to relax/study to)
video_id = "bSDprg24pEA"
try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id=video_id, languages=['en'])
    transcript = " ".join(snippet.text for snippet in transcript_list.snippets)
    
except TranscriptsDisabled:
    transcript = "Transcript is disabled for this video."
    print("Transcript is disabled for this video.")

# Step 1b. Splitting the transcript into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print(f"Total Chunks: {len(chunks)}")

start_time = datetime.now();

# Step 1c. Creating embeddings and storing them in a vector store
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = FAISS.from_documents(chunks, embeddings)
print("Vector Store created with embeddings.")
end_time = datetime.now();
print("33 Difference:", end_time - start_time)
# Step 2. Retriever
retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs={"k": 4})


parallel_chain = RunnableParallel({
    "context": retriever|RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

result = parallel_chain.invoke("What is the main topic of the video?")
parser = StrOutputParser()





# start_time = datetime.now();
# # Step 3. Generative AI (RAG)
model = ChatOpenAI(temperature=0, model_name="gpt-4")
prompt_template = PromptTemplate(
    template="""you are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If you can't find the answer, say "I don't know".
    Context: {context}
    Question: {question}""",
    input_variables=["context", "question"]
)
# end_time = datetime.now();
# print("48 Difference:", end_time - start_time)
main_chain = parallel_chain|prompt_template|model|parser
print(main_chain.invoke("Can you summarize the video"))

