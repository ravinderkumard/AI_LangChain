from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests

load_dotenv()

# Tools
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Returns the conversion factor from base_currency to target_currency."""
    url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
    response = requests.get(url)
    data = response.json()
    return data['rates'][target_currency]

@tool
def convert_currency(base_currency_value: int, conversion_factor: float) -> float:
    """Returns the converted currency value."""
    return base_currency_value * conversion_factor

# LLM + tools
llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools([get_conversion_factor, convert_currency])

# Initial user message
messages = [HumanMessage("What is the conversion factor between USD and INR and based on that convert 10 usd to inr")]

# Loop until final response
while True:
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    # If no tool calls -> final answer
    if not ai_message.tool_calls:
        print("Final Answer:", ai_message.content)
        break

    # Otherwise handle tool calls
    for tool_call in ai_message.tool_calls:
        if tool_call["name"] == "get_conversion_factor":
            result = get_conversion_factor.invoke(tool_call["args"])
        elif tool_call["name"] == "convert_currency":
            result = convert_currency.invoke(tool_call["args"])
        else:
            result = "Unknown tool"

        # Send tool result back
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
