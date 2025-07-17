from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0.0)

class Person(BaseModel):
    name: str = Field(description="Name of some fictional Person")
    age: int = Field(gt=18,description="Age of the person")
    city: str = Field(description='Name of the place where this person lives in ')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me details of Fictional Person from {place}\n{format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.invoke({'place','US'})
# result = llm.invoke(prompt)
# final_result = parser.parse(result.content)

# print(final_result)

chain = template | llm | parser
print("*************************")
print(chain)
print("*************************")
result = chain.invoke({'place':'Australia'})

print(result)