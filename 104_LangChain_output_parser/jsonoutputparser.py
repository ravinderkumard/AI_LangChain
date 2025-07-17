from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0.0)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template="Give me the name, age and city of a fictional person\n\n{format_instructions}",
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()}
)

prompt = template1.format()
print(prompt)
result = llm.invoke(prompt)
print("-------------------------------- ")
final_result = parser.parse(result.content)
print(final_result)