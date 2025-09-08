from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv

load_dotenv()

# Tool creation
@tool
def multiply(a: int, b: int) -> int:
    """Returns the product of two numbers."""
    return a * b

# Tool binding
llm = ChatOpenAI(model="gpt-4")

llm_with_tools = llm.bind_tools([multiply])

query = HumanMessage("What is the product of 3 and 4?")

messages = [query]

result = llm_with_tools.invoke(messages)

messages.append(result)

tool_result = multiply.invoke(result.tool_calls[0])

messages.append(tool_result)

final = llm_with_tools.invoke(messages)
print(final.content)