from langchain.tools import BaseTool
from typing import Type

from pydantic import BaseModel, Field

class MutliplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiplies two numbers"

    args_schema: Type[BaseModel] = MutliplyInput

    def _run(self, a: int, b: int) -> int:
        """Returns the product of two numbers."""
        return a * b
    
multiply_tool = MultiplyTool()
result = multiply_tool.invoke({"a":3,"b":4})
print(result)
