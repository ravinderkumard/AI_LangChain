from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class MutliplyInput(BaseModel):
    a: int = Field(..., description="The first number to multiply")
    b: int = Field(..., description="The second number to multiply")

def multiply_func(a: int, b: int) -> int:
    """Returns the product of two numbers."""
    return a * b

multiply_tool = StructuredTool.from_function(func=multiply_func,
                                             name="multiply",
                                             description="Multiplies two numbers",
                                             args_schema=MutliplyInput
                                             )
result = multiply_tool.invoke({"a":3,"b":4})
print(result)
