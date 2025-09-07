from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Returns the sum of two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Returns the difference of two numbers."""
    return a - b

class MathToolKit:
    def get_tools(self):
        return [add, subtract]
    
math_toolkit = MathToolKit()
tools = math_toolkit.get_tools()
print(tools)