from langchain_core.tools import tool

# Step 2: add type hints
@tool
def multiply(a: int, b: int) -> int:
    """Returns the product of two numbers."""
    return a * b


result = multiply.invoke({"a":3,"b":4})
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)
