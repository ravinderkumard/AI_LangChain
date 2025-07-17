from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0.0)

template1 = PromptTemplate(
    template="Write a detailed report on topic:\n\n{topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary of foolwing text:\n\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | llm | parser | template2 | llm | parser

print(chain.invoke({"topic":"Black HOle"}))

