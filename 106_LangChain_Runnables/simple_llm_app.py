from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0.0)

prompt = PromptTemplate(
    template="Suggest a catchy blog title about {topic}",
    input_variables=["topic"]
)

topic = input("Enter a topic:")

formatted_prompt = prompt.format(topic = topic)

blog_title = llm.invoke(formatted_prompt)

print(blog_title)