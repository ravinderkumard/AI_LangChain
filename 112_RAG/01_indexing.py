from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

## Step 1a. Indexing YouTube Video
#video_id = "5qap5aO4i9A"  # Example video ID (Lo-fi hip hop radio - beats to relax/study to)
video_id = "Gfr50f6ZBvo"
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
start_time = datetime.now();
# Step 3. Generative AI (RAG)
model = ChatOpenAI(temperature=0, model_name="gpt-4")
prompt_template = PromptTemplate(
    template="""you are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If you can't find the answer, say "I don't know".
    Context: {context}
    Question: {question}""",
    input_variables=["context", "question"]
)
end_time = datetime.now();
print("48 Difference:", end_time - start_time)

start_time = datetime.now();
question = "Is the topic of Science discussed in the video? If yes what was discussed"
retriever_docs = retriever.invoke(question)

context_text = "\n".join([doc.page_content for doc in retriever_docs])

final_prompt = prompt_template.invoke({"context":context_text,"question":question})
end_time = datetime.now();
print("58 Difference:", end_time - start_time)
print(final_prompt);

# Generation
answer = model.invoke(final_prompt)
print("Answer:", answer)