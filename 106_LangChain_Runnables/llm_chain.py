from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

prompt = PromptTemplate(
    template = "Suggest a catchy blog title about {topic}",
    input_variables=["topic"]
)

chain = LLMChain(llm=llm,prompt=prompt)

topic = input("enter a topic")
output = chain.run(topic)

print(output)